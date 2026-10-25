"""Offline checks: fake Segmenter + scripted LangChain chat model, so no GPU or API key is needed."""

import ast
import io
import json
from pathlib import Path

import numpy as np
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from PIL import Image
from pydantic import Field

from segagent.adapters.store import LocalStore
from segagent.core.agent import run_agent
from segagent.core.domain import Detection, Workspace
from segagent.core.formats import mask_to_polygon, yolo_det
from segagent.core.tools import Toolbox

W, H = 200, 100


def rect(x1, y1, x2, y2):
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


class FakeSam:
    """'person' = a tall box; 'hat' = a small box on its top part, plus a duplicate under a synonym."""

    def segment(self, image, phrases, threshold=0.5):
        table = {
            "person": [Detection("person", 0.9, [20, 20, 60, 100], rect(20, 20, 60, 100))],
            "hat": [Detection("hat", 0.8, [25, 20, 55, 35], rect(25, 20, 55, 35))],
            "helmet": [Detection("helmet", 0.7, [25, 20, 55, 35], rect(25, 20, 55, 35))],
        }
        dets = [d for p in phrases for d in table.get(p, [])]
        return dets, {p: (0.99 if p in table else 0.01) for p in phrases}

    def segment_examples(self, image, boxes, labels, phrase, threshold=0.5):
        """Finds the example itself again plus one look-alike hat."""
        found = [Detection(phrase, 0.9, list(b), rect(*map(int, b))) for b, label in zip(boxes, labels) if label]
        return found + [Detection(phrase, 0.7, [125, 20, 155, 35], rect(125, 20, 155, 35))], 0.98


class ScriptedChat(GenericFakeChatModel):
    """Replies with the given tool calls in order and records what it was sent."""

    seen: list = Field(default_factory=list)

    @classmethod
    def calling(cls, *calls):
        return cls(messages=iter(AIMessage("", tool_calls=[{"name": n, "args": a, "id": f"c{i}"}])
                                 for i, (n, a) in enumerate(calls)))

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def test_core_never_imports_outward():
    """The dependency rule: core/ knows nothing about adapters, model providers, HTTP or env config."""
    banned = ("segagent.adapters", "segagent.web", "torch", "transformers", "openai", "langchain_openai",
              "langchain_google", "fastapi", "dotenv")
    for f in (Path(__file__).parent.parent / "segagent" / "core").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not [n for n in names if n.startswith(banned)], f"{f.name} imports {names}"


def test_workspace_dedup_and_relations():
    ws = Workspace(Image.new("RGB", (W, H)), "img")
    added = ws.add(FakeSam().segment(None, ["person", "hat", "helmet"])[0])
    assert [i.id for i in added] == ["m1", "m2"]  # helmet merged into the hat
    assert ws.instances["m2"].phrases == ["hat", "helmet"]
    assert ws.relations() == ["m2 inside m1 (top part)"]
    assert ws.describe(ws.instances["m1"])["box"] == [100, 200, 300, 1000]


def test_yolo_det_and_split_mask_polygon():
    ws = Workspace(Image.new("RGB", (W, H)), "img")
    ws.add([Detection("x", 0.9, [0, 0, 100, 50], rect(0, 0, 100, 50))])
    ws.instances["m1"].class_name = "box"
    assert yolo_det(ws, [ws.instances["m1"]], ["box"]) == "0 0.250000 0.250000 0.500000 0.500000\n"

    poly = mask_to_polygon(rect(10, 10, 40, 40) | rect(120, 50, 180, 90))
    # One polygon that visits both pieces (bridged), not two separate objects.
    assert (poly[:, 0] < 50).any() and (poly[:, 0] > 110).any()


def test_select_all_keeps_unlisted_objects(tmp_path):
    class ManyBooks:
        def segment(self, image, phrases, threshold=0.5):
            return [Detection("book", 0.4 + i / 100, [i * 5, 0, i * 5 + 4, 10], rect(i * 5, 0, i * 5 + 4, 10))
                    for i in range(35)], {"book": 0.99}

    tools = Toolbox(ManyBooks(), Workspace(Image.new("RGB", (W, H)), "shelf"), LocalStore(tmp_path))
    result, _ = tools.segment_concepts(["book"])
    assert len(result["added"]) == 30 and result["more"] == 5 and "select_all" in result["note"]

    result, _ = tools.finalize(select_all=[{"phrase": "book", "class": "book", "min_score": 0.45}])
    assert result["selected"] == 30  # scores 0.45..0.74, including ones never listed to the LLM
    assert tools.finalize(select_all=[{"phrase": "lamp", "class": "x"}])[0]["error"]


def test_agent_loop_end_to_end(tmp_path):
    chat = ScriptedChat.calling(
        ("segment_concepts", {"phrases": ["person", "hat", "forklift"]}),
        ("finalize", {"selection": [{"id": "m9", "class": "person"}]}),  # bad id -> error, retry
        ("finalize", {"selection": [{"id": "m1", "class": "person"}, {"id": "m2", "class": "hat"}]}),
    )
    phases = []
    report = run_agent(Image.new("RGB", (W, H), "gray"), "site", "label people and hats", segmenter=FakeSam(),
                       chat=chat, store=LocalStore(tmp_path), classes=["person", "hat"],
                       on_progress=lambda r: phases.append(r["phase"]))

    assert report["status"] == "done"
    assert [(s["id"], s["class"]) for s in report["selected"]] == [("m1", "person"), ("m2", "hat")]
    assert phases[:2] == ["llm", "tool:segment_concepts"] and phases[-1] is None
    # The overlay follows the segment result as a user message; the bad id got a helpful error.
    tool_msg, image_msg = chat.seen[1][-2:]
    assert tool_msg.type == "tool" and "m2 inside m1 (top part)" in tool_msg.content and "forklift" in tool_msg.content
    assert image_msg.type == "human" and image_msg.content[1]["type"] == "image"
    assert "Valid ids" in chat.seen[2][-1].content

    seg = (tmp_path / report["files"]["yolo_seg"]).read_text().splitlines()
    assert [line.split()[0] for line in seg] == ["0", "1"]
    assert all(0 <= float(v) <= 1 for line in seg for v in line.split()[1:])
    assert [i["class_name"] for i in json.loads((tmp_path / "site.json").read_text())["instances"]] == ["person", "hat"]
    assert json.loads((tmp_path / "trace.json").read_text())["status"] == "done"
    for name in ("site.jpg", "site_overlay.jpg", "data.yaml", "seen_1.jpg"):
        assert (tmp_path / name).exists(), name


def test_segment_like_adds_lookalikes(tmp_path):
    tools = Toolbox(FakeSam(), Workspace(Image.new("RGB", (W, H)), "site"), LocalStore(tmp_path))
    tools.segment_concepts(["person", "hat"])
    result, _ = tools.segment_like(["m2"])
    assert [a["id"] for a in result["added"]] == ["m3"] and result["added"][0]["phrase"] == "like m2"
    assert result["also_matched_existing"] == ["m2"]  # the example itself is found again, not duplicated
    assert "Valid ids" in tools.segment_like(["m9"])[0]["error"]


def test_to_coco_loads_in_pycocotools(tmp_path):
    from pycocotools.coco import COCO

    from segagent.core.formats import to_coco

    ws = Workspace(Image.new("RGB", (W, H)), "img")
    ws.add([Detection("x", 0.9, [10, 10, 180, 90], rect(10, 10, 40, 40) | rect(120, 50, 180, 90))])
    ws.instances["m1"].class_name = "thing"
    (tmp_path / "coco.json").write_text(json.dumps(to_coco([(ws, [ws.instances["m1"]])], ["thing"])))
    coco = COCO(str(tmp_path / "coco.json"))
    ann = coco.loadAnns(coco.getAnnIds())[0]
    assert len(ann["segmentation"]) == 2 and ann["category_id"] == 1  # split mask keeps both pieces
    m, src = coco.annToMask(ann).astype(bool), ws.instances["m1"].mask
    assert (m & src).sum() / (m | src).sum() > 0.9


def test_parse_specs_reports_every_problem():
    import pytest

    from segagent.core.direct import parse_specs

    with pytest.raises(ValueError) as e:
        parse_specs([{"name": "a", "phrases": []}, {"name": "a", "phrases": ["x"], "min_score": 2}])
    assert all(s in str(e.value) for s in ("phrases must be", "min_score", "unique"))
    assert parse_specs([{"name": "hat", "phrases": [" Hard Hat "]}])[0].phrases == ["hard hat"]


def test_run_direct_first_spec_wins(tmp_path):
    from segagent.core.direct import ClassSpec, run_direct

    specs = [ClassSpec("helmet", ["hat"]), ClassSpec("headwear", ["hat", "helmet"]), ClassSpec("person", ["person"], 0.95)]
    report = run_direct(Image.new("RGB", (W, H)), "site", specs, segmenter=FakeSam(), store=LocalStore(tmp_path))
    assert report["selected"] == [{"id": "m2", "class": "helmet", "score": 0.8}]  # m1 = person, 0.9 < 0.95
    assert report["classes"] == ["helmet", "headwear", "person"] and report["steps"] == []
    assert {"coco", "yolo_seg", "overlay"} <= set(report["files"]) and (tmp_path / "trace.json").exists()


def test_v1_api_auth_isolation_and_runs(tmp_path, monkeypatch):
    import time as _time

    from fastapi.testclient import TestClient

    from segagent import web

    monkeypatch.setattr(web, "RUNS", tmp_path)
    monkeypatch.setattr(web, "API_KEYS", {"acme": "key-a", "beta": "key-b"})
    web.app.state.segmenter = FakeSam()
    web.app.state.chat = ScriptedChat.calling(
        ("segment_concepts", {"phrases": ["person"]}), ("finalize", {"selection": [{"id": "m1", "class": "person"}]}))
    client = TestClient(web.app)  # no `with`: the lifespan (real SAM 3 + LLM) doesn't run
    buf = io.BytesIO()
    Image.new("RGB", (W, H)).save(buf, "PNG")
    image = {"image": ("site.png", buf.getvalue(), "image/png")}
    a, b = {"X-API-Key": "key-a"}, {"X-API-Key": "key-b"}

    assert client.post("/v1/segment", files=image, data={"specs": "[]"}).status_code == 401
    assert client.post("/v1/segment", files=image, data={"specs": "[]"}, headers=a).status_code == 400
    r = client.post("/v1/segment", files=image, headers=a,
                    data={"specs": json.dumps([{"name": "person", "phrases": ["person"]}])}).json()
    assert r["selected"][0]["class"] == "person"
    assert client.get(f"/v1/runs/{r['id']}/files/{r['files']['yolo_det']}", headers=a).text.startswith("0 ")
    assert client.get(f"/v1/runs/{r['id']}", headers=b).status_code == 404  # other tenant: not found
    assert client.get(f"/v1/runs/{r['id']}/files/{r['files']['yolo_det']}", headers=b).status_code == 404
    assert client.get(f"/v1/runs/{r['id']}/files/..%2F..%2Fbeta", headers=a).status_code == 404  # no traversal
    assert client.get("/").status_code == 404 and client.get("/api/runs/x").status_code == 404  # demo not exposed

    run_id = client.post("/v1/agent/runs", files=image, data={"instruction": "label people"}, headers=a).json()["id"]
    for _ in range(50):
        report = client.get(f"/v1/runs/{run_id}", headers=a).json()
        if report["status"] != "running":
            break
        _time.sleep(0.1)
    assert report["status"] == "done" and report["selected"][0]["class"] == "person"
