"""CLI entrypoint: python -m segagent IMAGE "instruction" [--classes a,b]. Output goes to runs/<time>-<image>/."""

import argparse
import json
import time
from pathlib import Path

from segagent.adapters.llm import OpenAIChat
from segagent.adapters.sam3 import Sam3
from segagent.adapters.store import LocalStore, load_image
from segagent.core.agent import run_agent

p = argparse.ArgumentParser(prog="python -m segagent", description="Label one image with SAM 3 driven by an LLM agent.")
p.add_argument("image")
p.add_argument("instruction", help='e.g. "label every worker without a hard hat"')
p.add_argument("--classes", help="comma-separated class names; fixes YOLO class ids and order")
a = p.parse_args()

image_id = Path(a.image).stem
run_dir = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{image_id}"
classes = [c.strip() for c in a.classes.split(",") if c.strip()] if a.classes else None
report = run_agent(load_image(a.image), image_id, a.instruction, segmenter=Sam3(), chat=OpenAIChat(),
                   store=LocalStore(run_dir), classes=classes)
report.pop("steps")  # full detail is in trace.json
print(json.dumps({"run_dir": str(run_dir), **report}, indent=2))
