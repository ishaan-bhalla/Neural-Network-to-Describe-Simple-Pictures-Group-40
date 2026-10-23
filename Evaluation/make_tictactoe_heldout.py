"""Generate a larger held-out tic-tac-toe set for evaluation (no retraining).

Uses the same generator, renderer and settings as src/data/generate_tictactoe.py
(256x256, 2-6 marks, both players present) with a different seed. A board is
kept only if its state (which cell holds X, O or nothing) does not appear in
the original 300-board dataset (train, val or test) and has not been drawn
already, so every held-out board is unseen and distinct.

The accepted boards are shuffled and split into --n-val validation boards (used
only to pick the abstention threshold) and --n-test test boards.

usage: python Evaluation/make_tictactoe_heldout.py [--seed 2026 --n-val 500 --n-test 1000]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate_tictactoe import (  # noqa: E402
    POSITION_ORDER,
    board_to_symbolic_state,
    build_canonical_label,
    build_caption,
    random_board,
    render_board,
    symbolic_state_to_board,
    to_notation,
    write_jsonl,
)

ORIGINAL = PROJECT_ROOT / "data" / "processed" / "tictactoe" / "metadata"
OUT = PROJECT_ROOT / "data" / "processed" / "tictactoe_heldout"


def state_key(label: dict) -> tuple:
    return tuple(label[p] for p in POSITION_ORDER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--output-root", type=Path, default=OUT)
    args = ap.parse_args()

    original = {}
    for split in ["train", "val", "test"]:
        for line in open(ORIGINAL / f"{split}.jsonl", encoding="utf-8"):
            original.setdefault(state_key(json.loads(line)["canonical_label"]), set()).add(split)

    rng = random.Random(args.seed)
    seen: set[tuple] = set()
    boards = []
    rejected = Counter()
    draws = 0
    while len(boards) < args.n_val + args.n_test:
        draws += 1
        board = random_board(rng, 2, 6, allow_single_player=False)
        key = state_key(build_canonical_label(board_to_symbolic_state(board)))
        if key in original:
            rejected["in_original_dataset"] += 1
            continue
        if key in seen:
            rejected["duplicate_in_heldout"] += 1
            continue
        seen.add(key)
        boards.append(board)

    # Shuffle before splitting so val and test follow the same distribution.
    random.Random(args.seed + 1).shuffle(boards)
    rows = {"val": [], "test": []}
    for i, board in enumerate(boards):
        split = "val" if i < args.n_val else "test"
        symbolic = board_to_symbolic_state(board)
        sample_id = f"tictactoe_heldout_{split}_{len(rows[split]) + 1:06d}"
        rel = Path("images") / split / f"{sample_id}.png"
        render_board(board, args.output_root / rel, 256, rng)
        rows[split].append({
            "id": sample_id,
            "task": "tictactoe",
            "image_path": str(rel),
            "symbolic_state": symbolic,
            "caption": build_caption(symbolic),
            "canonical_label": build_canonical_label(symbolic),
            "split": split,
            "notation": to_notation(symbolic_state_to_board(symbolic)),
        })

    for split, split_rows in rows.items():
        write_jsonl(args.output_root / "metadata" / f"{split}.jsonl", split_rows)

    # Overlap check, recomputed from the files just written.
    written = {s: [state_key(json.loads(l)["canonical_label"])
                   for l in open(args.output_root / "metadata" / f"{s}.jsonl", encoding="utf-8")]
               for s in rows}
    overlap = {
        "heldout_test_in_original_train": sum(k in original and "train" in original[k] for k in written["test"]),
        "heldout_test_in_original_any_split": sum(k in original for k in written["test"]),
        "heldout_val_in_original_any_split": sum(k in original for k in written["val"]),
        "heldout_val_test_shared_states": len(set(written["val"]) & set(written["test"])),
        "duplicate_states_within_test": len(written["test"]) - len(set(written["test"])),
    }
    assert all(v == 0 for v in overlap.values()), overlap

    def occupied(split_rows):
        c = Counter(len(r["symbolic_state"]["X"]) + len(r["symbolic_state"]["O"]) for r in split_rows)
        return {str(k): c[k] for k in sorted(c)}

    summary = {
        "seed": args.seed,
        "generator": "src/data/generate_tictactoe.py (random_board, render_board), 256x256, 2-6 marks",
        "n_val": len(rows["val"]),
        "n_test": len(rows["test"]),
        "draws": draws,
        "rejected": dict(rejected),
        "overlap_check": overlap,
        "original_test_boards_whose_state_is_in_original_train": None,
        "occupied_cell_count_distribution": {s: occupied(r) for s, r in rows.items()},
    }
    orig_train = {k for k, s in original.items() if "train" in s}
    orig_test = [state_key(json.loads(l)["canonical_label"]) for l in open(ORIGINAL / "test.jsonl", encoding="utf-8")]
    summary["original_test_boards_whose_state_is_in_original_train"] = sum(k in orig_train for k in orig_test)
    (args.output_root / "metadata" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
