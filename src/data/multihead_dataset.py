from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


POSITION_ORDER = [
    "top left", "top middle", "top right",
    "middle left", "center", "middle right",
    "bottom left", "bottom middle", "bottom right",
]


class StructuredDataset(Dataset):
    def __init__(
        self,
        metadata_file: str | Path,
        task: str,
        label_maps: Dict[str, Dict[str, int]],
        split: Optional[str] = None,
        image_size: int = 64,
    ):
        self.metadata_file = Path(metadata_file).resolve()
        self.dataset_root = self.metadata_file.parent.parent
        self.task = task
        self.label_maps = label_maps

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if split is not None:
            rows = [r for r in rows if r.get("split") == split]

        self.rows = rows

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = self.dataset_root / row["image_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        targets = self.build_targets(row)

        meta = {
            "id": row["id"],
            "true_caption": row["caption"],
        }

        return image, targets, meta

    def build_targets(self, row):
        if self.task == "shapes":
            ss = row["symbolic_state"]
            o1 = ss["object_1"]
            o2 = ss["object_2"]

            values = {
                "object_1_size": o1["size"],
                "object_1_color": o1["color"],
                "object_1_shape": o1["shape"],
                "relation": ss["relation"],
                "object_2_size": o2["size"],
                "object_2_color": o2["color"],
                "object_2_shape": o2["shape"],
            }

        elif self.task == "numbers":
            ss = row["symbolic_state"]
            digits = str(ss["digits"])

            values = {
                "size": ss["size"],
                "color": ss["color"],
                "length": str(len(digits)),
            }

            max_digits = self.get_max_digit_heads()
            for i in range(max_digits):
                key = f"digit_{i}"
                values[key] = digits[i] if i < len(digits) else "<pad>"

        elif self.task == "tictactoe":
            label = row["canonical_label"]
            values = {}

            for pos in POSITION_ORDER:
                values[pos] = label[pos]

        else:
            raise ValueError(f"Unknown task: {self.task}")

        return {
            key: torch.tensor(self.label_maps[key][value], dtype=torch.long)
            for key, value in values.items()
        }

    def get_max_digit_heads(self):
        return len([k for k in self.label_maps if k.startswith("digit_")])


def load_rows(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_label_maps(task: str, rows: List[dict]) -> Dict[str, Dict[str, int]]:
    label_maps: Dict[str, Dict[str, int]] = {}

    def add_map(key: str, values: List[str]):
        values = sorted(set(values))
        label_maps[key] = {v: i for i, v in enumerate(values)}

    if task == "shapes":
        add_map("object_1_size", [r["symbolic_state"]["object_1"]["size"] for r in rows])
        add_map("object_1_color", [r["symbolic_state"]["object_1"]["color"] for r in rows])
        add_map("object_1_shape", [r["symbolic_state"]["object_1"]["shape"] for r in rows])

        add_map("relation", [r["symbolic_state"]["relation"] for r in rows])

        add_map("object_2_size", [r["symbolic_state"]["object_2"]["size"] for r in rows])
        add_map("object_2_color", [r["symbolic_state"]["object_2"]["color"] for r in rows])
        add_map("object_2_shape", [r["symbolic_state"]["object_2"]["shape"] for r in rows])

    elif task == "numbers":
        digits_list = [str(r["symbolic_state"]["digits"]) for r in rows]

        add_map("size", [r["symbolic_state"]["size"] for r in rows])
        add_map("color", [r["symbolic_state"]["color"] for r in rows])
        add_map("length", [str(len(d)) for d in digits_list])

        max_len = max(len(d) for d in digits_list)
        digit_vocab = ["<pad>"] + [str(i) for i in range(10)]

        for i in range(max_len):
            label_maps[f"digit_{i}"] = {v: idx for idx, v in enumerate(digit_vocab)}

    elif task == "tictactoe":
        for pos in POSITION_ORDER:
            label_maps[pos] = {
                "empty": 0,
                "X": 1,
                "O": 2,
            }

    else:
        raise ValueError(f"Unknown task: {task}")

    return label_maps


def invert_label_maps(label_maps: Dict[str, Dict[str, int]]) -> Dict[str, Dict[int, str]]:
    return {
        key: {idx: value for value, idx in mapping.items()}
        for key, mapping in label_maps.items()
    }


def reconstruct_caption(task: str, pred_values: Dict[str, str]) -> str:
    if task == "shapes":
        return (
            f"a {pred_values['object_1_size']} "
            f"{pred_values['object_1_color']} "
            f"{pred_values['object_1_shape']} is "
            f"{pred_values['relation']} a "
            f"{pred_values['object_2_size']} "
            f"{pred_values['object_2_color']} "
            f"{pred_values['object_2_shape']}"
        )

    if task == "numbers":
        digits = []
        i = 0
        while f"digit_{i}" in pred_values:
            d = pred_values[f"digit_{i}"]
            if d != "<pad>":
                digits.append(d)
            i += 1

        number = "".join(digits)

        return f"a {pred_values['size']} {pred_values['color']} {number}"

    if task == "tictactoe":
        x_positions = []
        o_positions = []

        for pos in POSITION_ORDER:
            value = pred_values[pos]
            if value == "X":
                x_positions.append(pos)
            elif value == "O":
                o_positions.append(pos)

        def format_positions(values):
            if not values:
                return "none"
            if len(values) == 1:
                return values[0]
            return " and ".join(values)

        return f"X is in {format_positions(x_positions)}; O is in {format_positions(o_positions)}"

    raise ValueError(f"Unknown task: {task}")
