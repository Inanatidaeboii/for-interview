"""Offline checks: fake Segmenter + scripted ChatModel, so no GPU or API key is needed."""

import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image

from segagent.adapters.store import LocalStore
from segagent.core.agent import run_agent
from segagent.core.domain import Detection, Workspace
from segagent.core.formats import mask_to_polygon, yolo_det
from segagent.core.ports import Reply, ToolCall
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


class ScriptedChat:
    name = "fake"

    def __init__(self, *calls):
        self.calls, self.seen = list(calls), []

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        name, args = self.calls.pop(0)
        call = ToolCall(f"c{len(self.seen)}", name, json.dumps(args))
        return Reply(None, [call], {"role": "assistant", "content": None, "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": name, "arguments": call.arguments}}]})


def test_core_never_imports_outward():
    """The dependency rule: core/ knows nothing about adapters, models, HTTP or env config."""
    banned = ("segagent.adapters", "segagent.web", "torch", "transformers", "openai", "fastapi", "dotenv")
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
    chat = ScriptedChat(
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
    assert tool_msg["role"] == "tool" and "m2 inside m1 (top part)" in tool_msg["content"] and "forklift" in tool_msg["content"]
    assert image_msg["role"] == "user" and image_msg["content"][1]["type"] == "image"
    assert "Valid ids" in chat.seen[2][-1]["content"]

    seg = (tmp_path / report["files"]["yolo_seg"]).read_text().splitlines()
    assert [line.split()[0] for line in seg] == ["0", "1"]
    assert all(0 <= float(v) <= 1 for line in seg for v in line.split()[1:])
    assert [i["class_name"] for i in json.loads((tmp_path / "site.json").read_text())["instances"]] == ["person", "hat"]
    assert json.loads((tmp_path / "trace.json").read_text())["status"] == "done"
    for name in ("site.jpg", "site_overlay.jpg", "data.yaml", "seen_1.jpg"):
        assert (tmp_path / name).exists(), name
