"""Evaluate the committed tic-tac-toe MultiHead CNN (the weights the demo serves).

No retraining. Scores the original 30-board test split and the held-out set from
make_tictactoe_heldout.py, with 95% confidence intervals, an abstention rule
("I don't know" when the least confident of the 9 cell heads is below a
threshold chosen on the held-out validation split), and CPU latency.

usage: python Evaluation/eval_multihead_tictactoe.py [--date 2026-10-23]
writes Evaluation/results_multihead_tictactoe_<date>.json
       Evaluation/predictions_multihead_tictactoe_<date>.json (per board, used by cascade_tictactoe.py)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.multihead_inference import _TRANSFORM, POSITION_ORDER, load_model  # noqa: E402

ORIGINAL = PROJECT_ROOT / "data" / "processed" / "tictactoe"
HELDOUT = PROJECT_ROOT / "data" / "processed" / "tictactoe_heldout"
WEIGHTS = PROJECT_ROOT / "artifacts" / "tictactoe" / "multihead_cnn" / "best_model.pth"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]
# Rule fixed before looking at the test split: the lowest threshold whose
# answered boards on the validation split are all correct (100% selective
# sentence accuracy), i.e. the most coverage with no validation error.
TARGET_VAL_SELECTIVE_ACC = 1.0
N_BOOT = 10_000


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def bootstrap(per_board: np.ndarray, seed: int = 0) -> list[float]:
    """95% percentile bootstrap CI of the mean, resampling boards."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_board), size=(N_BOOT, len(per_board)))
    means = per_board[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def caption(cells: dict) -> str:
    x = [p for p in POSITION_ORDER if cells[p] == "X"]
    o = [p for p in POSITION_ORDER if cells[p] == "O"]
    return f"X is in {' and '.join(x) or 'none'}; O is in {' and '.join(o) or 'none'}"


def load_rows(root: Path, split: str) -> list[dict]:
    return [json.loads(l) for l in open(root / "metadata" / f"{split}.jsonl", encoding="utf-8")]


@torch.no_grad()
def predict_rows(model, label_maps, rows, root: Path) -> list[dict]:
    inverse = {k: {v: lbl for lbl, v in m.items()} for k, m in label_maps.items()}
    out = []
    for r in rows:
        x = _TRANSFORM(Image.open(root / r["image_path"]).convert("RGB")).unsqueeze(0)
        logits = model(x)
        cells, conf = {}, {}
        for head in POSITION_ORDER:
            probs = F.softmax(logits[head][0], dim=0)
            i = int(probs.argmax())
            cells[head] = inverse[head][i]
            conf[head] = float(probs[i])
        hits = sum(cells[p] == r["canonical_label"][p] for p in POSITION_ORDER)
        pred_caption = caption(cells)
        out.append({
            "id": r["id"],
            "true_caption": r["caption"],
            "pred_caption": pred_caption,
            "exact_match": pred_caption == r["caption"],
            "cells_correct": hits,
            "min_cell_confidence": min(conf.values()),
            "cell_confidence": conf,
        })
    return out


def score(preds: list[dict]) -> dict:
    n = len(preds)
    exact = np.array([p["exact_match"] for p in preds], dtype=float)
    cells = np.array([p["cells_correct"] for p in preds], dtype=float) / 9
    k = int(exact.sum())
    c = int(round(cells.sum() * 9))
    return {
        "n_boards": n,
        "sentence_correct": k,
        "sentence_accuracy": k / n,
        "sentence_accuracy_wilson95": wilson(k, n),
        "sentence_accuracy_bootstrap95": bootstrap(exact),
        "cells_correct": c,
        "n_cells": 9 * n,
        "cell_accuracy": c / (9 * n),
        # Cells on the same board are not independent, so the CI resamples boards.
        "cell_accuracy_bootstrap95": bootstrap(cells),
    }


def coverage_table(preds: list[dict]) -> list[dict]:
    n = len(preds)
    rows = []
    for t in THRESHOLDS:
        answered = [p for p in preds if p["min_cell_confidence"] >= t]
        k = sum(p["exact_match"] for p in answered)
        rows.append({
            "threshold": t,
            "answered": len(answered),
            "coverage": len(answered) / n,
            "answered_correct": k,
            "selective_sentence_accuracy": k / len(answered) if answered else None,
            "selective_accuracy_wilson95": wilson(k, len(answered)) if answered else None,
            "errors_answered": len(answered) - k,
            "abstained": n - len(answered),
            "abstained_that_were_wrong": sum(not p["exact_match"] for p in preds if p["min_cell_confidence"] < t),
        })
    return rows


@torch.no_grad()
def latency(model, rows, root: Path, warmup: int = 50) -> dict:
    paths = [root / r["image_path"] for r in rows]
    tensors = [_TRANSFORM(Image.open(p).convert("RGB")).unsqueeze(0) for p in paths]
    for x in tensors[:warmup]:
        model(x)
    fwd = []
    for x in tensors:
        t0 = time.perf_counter()
        logits = model(x)
        _ = {h: F.softmax(v[0], dim=0).argmax() for h, v in logits.items()}
        fwd.append(time.perf_counter() - t0)
    e2e = []
    for p in paths:
        t0 = time.perf_counter()
        x = _TRANSFORM(Image.open(p).convert("RGB")).unsqueeze(0)
        logits = model(x)
        _ = {h: F.softmax(v[0], dim=0).argmax() for h, v in logits.items()}
        e2e.append(time.perf_counter() - t0)

    def stats(xs):
        ms = np.array(xs) * 1000
        return {"n": len(ms), "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
                "mean_ms": float(ms.mean()), "max_ms": float(ms.max())}

    return {
        "device": "cpu",
        "batch_size": 1,
        "torch_threads": torch.get_num_threads(),
        "warmup_images": warmup,
        "forward_only": stats(fwd),
        "end_to_end_png_to_prediction": stats(e2e),
    }


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    device = torch.device("cpu")
    model, label_maps = load_model("tictactoe", PROJECT_ROOT, device)
    n_params = sum(p.numel() for p in model.parameters())

    preds = {
        "original_test_30": predict_rows(model, label_maps, load_rows(ORIGINAL, "test"), ORIGINAL),
        "heldout_val": predict_rows(model, label_maps, load_rows(HELDOUT, "val"), HELDOUT),
        "heldout_test": predict_rows(model, label_maps, load_rows(HELDOUT, "test"), HELDOUT),
    }

    val_table = coverage_table(preds["heldout_val"])
    ok = [r for r in val_table if r["selective_sentence_accuracy"] is not None
          and r["selective_sentence_accuracy"] >= TARGET_VAL_SELECTIVE_ACC]
    chosen = min(r["threshold"] for r in ok) if ok else None
    test_table = coverage_table(preds["heldout_test"])

    load_before = sh("uptime")
    lat = latency(model, load_rows(HELDOUT, "test"), HELDOUT)
    load_after = sh("uptime")

    results = {
        "date": args.date,
        "model": "MultiHead CNN (StructuredCNN), committed weights",
        "weights": str(WEIGHTS.relative_to(PROJECT_ROOT)),
        "weights_file_bytes": WEIGHTS.stat().st_size,
        "parameters": n_params,
        "machine": {
            "cpu": sh("sysctl -n machdep.cpu.brand_string"),
            "ram_bytes": int(sh("sysctl -n hw.memsize") or 0),
            "logical_cpus": int(sh("sysctl -n hw.ncpu") or 0),
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "load_before_timing": load_before,
            "load_after_timing": load_after,
            "ollama_ps_during_timing": sh("ollama ps"),
        },
        "confidence": "min over the 9 cell heads of the top softmax probability",
        "metrics": {k: score(v) for k, v in preds.items()},
        "abstention": {
            "threshold_rule": "lowest threshold with 100% selective sentence accuracy on heldout_val",
            "chosen_threshold": chosen,
            "val_coverage_table": val_table,
            "test_coverage_table": test_table,
        },
        "latency": lat,
    }
    out = PROJECT_ROOT / "Evaluation" / f"results_multihead_tictactoe_{args.date}.json"
    out.write_text(json.dumps(results, indent=1))
    (PROJECT_ROOT / "Evaluation" / f"predictions_multihead_tictactoe_{args.date}.json").write_text(
        json.dumps(preds, indent=1))
    print(json.dumps({k: results[k] for k in ["parameters", "weights_file_bytes", "metrics", "latency"]}, indent=1))
    print("chosen threshold:", chosen)
    for name, table in [("val", val_table), ("test", test_table)]:
        print(name)
        for r in table:
            print(f"  t={r['threshold']:<6} answered {r['answered']:4d} ({r['coverage']:.1%})  "
                  f"acc {r['selective_sentence_accuracy']}  errors {r['errors_answered']}")


if __name__ == "__main__":
    main()
