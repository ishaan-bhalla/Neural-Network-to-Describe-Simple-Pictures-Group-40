"""Zero-shot Qwen3-VL baseline on tic-tac-toe boards (local Ollama).

Same prompt and settings as the 2026-10-19 run on the original 30 test boards:
the model is asked for the 9 cells as JSON (Ollama "format": "json"),
temperature 0, thinking off. The caption is rebuilt from the cells and scored
exactly like the MultiHead CNN (exact caption match and per-cell accuracy).

Also records the wall time of each request (image already on disk -> parsed
answer, model already loaded; one untimed warm-up request first) and Ollama's
own duration fields. Results are appended to a JSONL file as they arrive, so
an interrupted run can be resumed by running the same command again.

usage:
  python Evaluation/qwen3vl_tictactoe.py --split test --ids first:100 --out Evaluation/qwen3vl_runs_<date>.jsonl
  python Evaluation/qwen3vl_tictactoe.py --split test --ids-file ids.txt --out ...
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOTS = {
    "original_test": (PROJECT_ROOT / "data" / "processed" / "tictactoe", "test"),
    "test": (PROJECT_ROOT / "data" / "processed" / "tictactoe_heldout", "test"),
    "val": (PROJECT_ROOT / "data" / "processed" / "tictactoe_heldout", "val"),
}
CELLS = ["top left", "top middle", "top right", "middle left", "center",
         "middle right", "bottom left", "bottom middle", "bottom right"]

PROMPT = (
    "This image shows a 3x3 tic-tac-toe board. For each of the 9 cells, say "
    "whether it contains X, O, or is empty. Reply with only a JSON object with "
    "exactly these keys: " + ", ".join(f'"{c}"' for c in CELLS) +
    '. Each value must be "X", "O" or "empty".'
)


def caption(cells: dict) -> str:
    x = [c for c in CELLS if cells[c] == "X"]
    o = [c for c in CELLS if cells[c] == "O"]
    return f"X is in {' and '.join(x) or 'none'}; O is in {' and '.join(o) or 'none'}"


def ask(model: str, image: Path) -> dict:
    t0 = time.perf_counter()
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "images": [base64.b64encode(image.read_bytes()).decode()],
        "format": "json",
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    raw = resp["response"]
    try:
        got = json.loads(raw)
    except json.JSONDecodeError:
        got = {}
    cells = {}
    for c in CELLS:
        v = str(got.get(c, "")).strip().upper()
        cells[c] = v if v in ("X", "O") else "empty"
    wall = time.perf_counter() - t0
    durations = {k: resp.get(k, 0) / 1e9 for k in
                 ["total_duration", "load_duration", "prompt_eval_duration", "eval_duration"]}
    return {"raw": raw, "cells": cells, "wall_s": wall, "ollama_s": durations,
            "eval_count": resp.get("eval_count")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(ROOTS), default="test")
    ap.add_argument("--ids", default=None, help="'first:N' to take the first N boards")
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--model", default="qwen3-vl:32b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-boards", type=int, default=150, help="hard cap per invocation")
    args = ap.parse_args()

    root, split = ROOTS[args.split]
    rows = [json.loads(l) for l in open(root / "metadata" / f"{split}.jsonl", encoding="utf-8")]
    if args.ids_file:
        wanted = [l.strip() for l in open(args.ids_file) if l.strip()]
        by_id = {r["id"]: r for r in rows}
        rows = [by_id[i] for i in wanted]
    elif args.ids and args.ids.startswith("first:"):
        rows = rows[: int(args.ids.split(":")[1])]

    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
    todo = [r for r in rows if r["id"] not in done][: args.max_boards]
    print(f"{len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return

    # Untimed warm-up so the model load is not counted.
    w = ask(args.model, root / todo[0]["image_path"])
    print(f"warm-up {w['wall_s']:.1f}s (load {w['ollama_s']['load_duration']:.1f}s)", flush=True)

    with open(args.out, "a") as f:
        for r in todo:
            res = ask(args.model, root / r["image_path"])
            pred = caption(res["cells"])
            hits = sum(res["cells"][c] == r["canonical_label"][c] for c in CELLS)
            rec = {"id": r["id"], "split": args.split, "model": args.model,
                   "true_caption": r["caption"], "pred_caption": pred,
                   "exact_match": pred == r["caption"], "cells_correct": hits,
                   "wall_s": res["wall_s"], "ollama_s": res["ollama_s"],
                   "eval_count": res["eval_count"], "raw": res["raw"],
                   "loadavg": os.getloadavg(),
                   "ollama_ps": subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.strip(),
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{r['id']}  {'OK ' if rec['exact_match'] else 'BAD'}  cells {hits}/9  "
                  f"{res['wall_s']:.1f}s  load {rec['loadavg'][0]:.1f}", flush=True)


if __name__ == "__main__":
    main()
