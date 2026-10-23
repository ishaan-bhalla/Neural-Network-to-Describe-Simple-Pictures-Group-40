"""Compare the MultiHead CNN with Qwen3-VL and score a CNN -> Qwen3-VL cascade.

Inputs (from eval_multihead_tictactoe.py and qwen3vl_tictactoe.py):
  Evaluation/results_multihead_tictactoe_<date>.json
  Evaluation/predictions_multihead_tictactoe_<date>.json
  Evaluation/qwen3vl_tictactoe_runs_<date>.jsonl

Qwen3-VL was run on the first 100 held-out test boards, a random sample because
the held-out set is shuffled (the run was planned to continue with the CNN's
low-confidence boards but was stopped at 100 to free the GPU). The cascade is
scored on that 100-board sample for every threshold, and on all 1,000 boards
only for thresholds whose routed boards all have a Qwen3-VL answer.

Cascade: if the CNN's min cell confidence >= threshold, answer with the CNN;
otherwise send the board to Qwen3-VL. Latency per board = CNN end-to-end
mean + the measured Qwen3-VL wall time for routed boards.

usage: python Evaluation/cascade_tictactoe.py [--date 2026-10-23]
writes Evaluation/results_cascade_tictactoe_<date>.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_multihead_tictactoe import THRESHOLDS, bootstrap, wilson  # noqa: E402

EVAL = Path(__file__).resolve().parent
QWEN_SIZE_BYTES = 20_910_274_496  # qwen3-vl:32b model blob (Q4_K_M), from the Ollama manifest
QWEN_PARAMS = 33.4e9              # `ollama show qwen3-vl:32b`


def pct(xs, q):
    return float(np.percentile(np.array(xs), q))


def acc_block(exact: list[bool], cells: list[int]) -> dict:
    n = len(exact)
    k = int(sum(exact))
    c = int(sum(cells))
    return {"n_boards": n, "sentence_correct": k, "sentence_accuracy": k / n,
            "sentence_accuracy_wilson95": wilson(k, n),
            "cells_correct": c, "n_cells": 9 * n, "cell_accuracy": c / (9 * n),
            "cell_accuracy_bootstrap95": bootstrap(np.array(cells, dtype=float) / 9)}


def cascade(cnn: list[dict], qwen: dict, t: float, cnn_ms: float) -> dict | None:
    routed = [p for p in cnn if p["min_cell_confidence"] < t]
    if any(p["id"] not in qwen for p in routed):
        return None
    exact, cells, lat = [], [], []
    for p in cnn:
        if p["min_cell_confidence"] < t:
            q = qwen[p["id"]]
            exact.append(q["exact_match"])
            cells.append(q["cells_correct"])
            lat.append(cnn_ms + q["wall_s"] * 1000)
        else:
            exact.append(p["exact_match"])
            cells.append(p["cells_correct"])
            lat.append(cnn_ms)
    block = acc_block(exact, cells)
    block.update({
        "threshold": t,
        "routed_to_qwen": len(routed),
        "routed_fraction": len(routed) / len(cnn),
        "cnn_errors_routed": sum(not p["exact_match"] for p in routed),
        "qwen_correct_on_routed": sum(qwen[p["id"]]["exact_match"] for p in routed),
        "mean_latency_ms_per_board": float(np.mean(lat)),
        "p50_latency_ms": pct(lat, 50),
        "p95_latency_ms": pct(lat, 95),
    })
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()

    cnn_res = json.loads((EVAL / f"results_multihead_tictactoe_{args.date}.json").read_text())
    preds = json.loads((EVAL / f"predictions_multihead_tictactoe_{args.date}.json").read_text())["heldout_test"]
    runs = [json.loads(l) for l in open(EVAL / f"qwen3vl_tictactoe_runs_{args.date}.jsonl") if l.strip()]
    qwen = {r["id"]: r for r in runs}
    sample_ids = [p["id"] for p in preds[:100]]
    sample_ids = [i for i in sample_ids if i in qwen]  # in case the run was stopped early
    sample = [p for p in preds if p["id"] in set(sample_ids)]
    cnn_ms = cnn_res["latency"]["end_to_end_png_to_prediction"]["mean_ms"]

    walls = [r["wall_s"] for r in runs]
    loads = [r["loadavg"][0] for r in runs]
    other_models = sorted({line.split()[0] for r in runs for line in r["ollama_ps"].splitlines()[1:]
                           if line.split() and line.split()[0] != "qwen3-vl:32b"})

    uncertain = [p for p in preds if p["min_cell_confidence"] < 0.7 and p["id"] in qwen]
    head_to_head = {
        "boards": "first 100 held-out test boards (random sample of the shuffled set)",
        "multihead_cnn": acc_block([p["exact_match"] for p in sample], [p["cells_correct"] for p in sample]),
        "qwen3_vl_32b": acc_block([qwen[i]["exact_match"] for i in sample_ids],
                                  [qwen[i]["cells_correct"] for i in sample_ids]),
    }

    head_to_head["cnn_uncertain_boards_run_on_qwen"] = {
        "n": len(uncertain),
        "cnn_correct": sum(p["exact_match"] for p in uncertain),
        "qwen_correct": sum(qwen[p["id"]]["exact_match"] for p in uncertain),
    }

    cnn_lat = cnn_res["latency"]["end_to_end_png_to_prediction"]
    latency = {
        "multihead_cnn_cpu_batch1": {**cnn_lat, "parameters": cnn_res["parameters"],
                                     "file_bytes": cnn_res["weights_file_bytes"]},
        "qwen3_vl_32b_ollama": {
            "n": len(walls),
            "p50_ms": pct(walls, 50) * 1000, "p95_ms": pct(walls, 95) * 1000,
            "mean_ms": float(np.mean(walls)) * 1000, "max_ms": float(np.max(walls)) * 1000,
            "mean_output_tokens": float(np.mean([r["eval_count"] for r in runs])),
            "parameters": QWEN_PARAMS, "file_bytes": QWEN_SIZE_BYTES, "quantization": "Q4_K_M",
            "measured": "wall time of one /api/generate call (base64 PNG -> parsed JSON), model loaded, "
                        "Apple GPU via Ollama",
        },
        "speedup_p50": pct(walls, 50) * 1000 / cnn_lat["p50_ms"],
        "speedup_p95": pct(walls, 95) * 1000 / cnn_lat["p95_ms"],
        "parameter_ratio": QWEN_PARAMS / cnn_res["parameters"],
        "file_size_ratio": QWEN_SIZE_BYTES / cnn_res["weights_file_bytes"],
        "contention": {
            "loadavg_1min_during_qwen_min_max": [min(loads), max(loads)],
            "other_ollama_models_seen_loaded": other_models,
            "first_and_last_qwen_timestamp": [runs[0]["timestamp"], runs[-1]["timestamp"]],
        },
    }

    full = {str(t): cascade(preds, qwen, t, cnn_ms) for t in THRESHOLDS}
    on_sample = {str(t): cascade(sample, qwen, t, cnn_ms) for t in THRESHOLDS}
    out = {
        "date": args.date,
        "qwen_boards_run": len(runs),
        "qwen_prompt_settings": "same prompt as the 2026-10-19 baseline; format=json, temperature 0, think=false",
        "head_to_head_100": head_to_head,
        "latency_and_size": latency,
        "chosen_threshold_from_val": cnn_res["abstention"]["chosen_threshold"],
        # Second operating point, added after seeing that the 100% rule keeps only ~40% of boards:
        # lowest threshold with >= 99.5% selective sentence accuracy on heldout_val.
        "secondary_threshold_from_val": min(r["threshold"] for r in cnn_res["abstention"]["val_coverage_table"]
                                            if r["selective_sentence_accuracy"] is not None
                                            and r["selective_sentence_accuracy"] >= 0.995),
        "cascade_full_1000": {k: v for k, v in full.items() if v is not None},
        "cascade_full_1000_not_computable_for": [k for k, v in full.items() if v is None],
        "cascade_on_100_sample": on_sample,
        # Without Qwen answers for every routed board, the full-set cascade accuracy is only bounded:
        # lower = Qwen gets every routed board wrong, upper = Qwen gets every routed board right.
        "cascade_full_1000_bounds": {str(t): {
            "routed": sum(p["min_cell_confidence"] < t for p in preds),
            "routed_with_qwen_answer": sum(p["min_cell_confidence"] < t and p["id"] in qwen for p in preds),
            "cnn_correct_on_kept": sum(p["exact_match"] for p in preds if p["min_cell_confidence"] >= t),
            "sentence_accuracy_lower": sum(p["exact_match"] for p in preds if p["min_cell_confidence"] >= t) / len(preds),
            "sentence_accuracy_upper": (sum(p["exact_match"] for p in preds if p["min_cell_confidence"] >= t)
                                        + sum(p["min_cell_confidence"] < t for p in preds)) / len(preds),
        } for t in THRESHOLDS},
    }
    path = EVAL / f"results_cascade_tictactoe_{args.date}.json"
    path.write_text(json.dumps(out, indent=1))

    def short(b):
        return (f"acc {b['sentence_correct']}/{b['n_boards']} ({b['sentence_accuracy']:.2%}) cells {b['cell_accuracy']:.2%}"
                f" routed {b['routed_to_qwen']} ({b['routed_fraction']:.1%}) mean {b['mean_latency_ms_per_board']:.0f} ms")
    print(json.dumps({"head_to_head": {k: {kk: v[kk] for kk in ["n_boards", "sentence_correct", "sentence_accuracy",
                                                                 "sentence_accuracy_wilson95", "cell_accuracy"]}
                                       for k, v in head_to_head.items() if isinstance(v, dict) and "n_boards" in v},
                      "latency": latency}, indent=1))
    print("cascade, all 1000:")
    for k, b in out["cascade_full_1000"].items():
        print(" ", k, short(b))
    print("cascade, 100 sample:")
    for k, b in on_sample.items():
        print(" ", k, short(b))


if __name__ == "__main__":
    main()
