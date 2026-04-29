from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.multihead_dataset import (
    StructuredDataset,
    build_label_maps,
    invert_label_maps,
    load_rows,
    reconstruct_caption,
)
from src.models.multihead_cnn import StructuredCNN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_metadata_files(project_root: Path, dataset: str):
    metadata_dir = project_root / "data" / "processed" / dataset / "metadata"

    single_file = metadata_dir / f"{dataset}_metadata.jsonl"
    train_file = metadata_dir / "train.jsonl"
    val_file = metadata_dir / "val.jsonl"
    test_file = metadata_dir / "test.jsonl"

    if single_file.exists():
        rows = load_rows(single_file)

        return {
            "format": "single",
            "all_rows": rows,
            "train_file": single_file,
            "val_file": single_file,
            "test_file": single_file,
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "source": str(single_file),
        }

    if train_file.exists() and val_file.exists() and test_file.exists():
        rows = load_rows(train_file) + load_rows(val_file) + load_rows(test_file)

        return {
            "format": "split",
            "all_rows": rows,
            "train_file": train_file,
            "val_file": val_file,
            "test_file": test_file,
            "train_split": None,
            "val_split": None,
            "test_split": None,
            "source": f"{train_file}, {val_file}, {test_file}",
        }

    raise FileNotFoundError(f"No metadata found in {metadata_dir}")


def evaluate(model, loader, loss_fn, device, head_names):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    correct_by_head = {h: 0 for h in head_names}
    total_by_head = {h: 0 for h in head_names}

    exact_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets, _meta in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)

            loss = 0.0
            all_correct = torch.ones(images.size(0), dtype=torch.bool, device=device)

            for head in head_names:
                logits = outputs[head]
                labels = targets[head]

                loss = loss + loss_fn(logits, labels)

                preds = logits.argmax(dim=1)
                correct = preds == labels

                correct_by_head[head] += correct.sum().item()
                total_by_head[head] += labels.size(0)

                all_correct &= correct

            total_loss += loss.item()
            total_batches += 1

            exact_correct += all_correct.sum().item()
            total_samples += images.size(0)

    head_accuracy = {
        h: correct_by_head[h] / total_by_head[h]
        for h in head_names
    }

    return {
        "loss": total_loss / total_batches,
        "exact_structured_accuracy": exact_correct / total_samples,
        "head_accuracy": head_accuracy,
    }


def save_predictions(
    model,
    loader,
    device,
    task,
    head_names,
    inverse_maps,
    output_file: Path,
):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, targets, meta in loader:
            images = images.to(device)
            outputs = model(images)

            batch_size = images.size(0)

            for i in range(batch_size):
                pred_values = {}

                for head in head_names:
                    pred_idx = outputs[head][i].argmax().item()
                    pred_values[head] = inverse_maps[head][pred_idx]

                pred_caption = reconstruct_caption(task, pred_values)

                predictions.append({
                    "id": meta["id"][i],
                    "true_caption": meta["true_caption"][i],
                    "pred_caption": pred_caption,
                    "exact_match": pred_caption == meta["true_caption"][i],
                    "predicted_attributes": pred_values,
                })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def train(args):
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[2]
    info = get_metadata_files(project_root, args.dataset)

    label_maps = build_label_maps(args.dataset, info["all_rows"])
    inverse_maps = invert_label_maps(label_maps)

    head_dims = {
        head: len(mapping)
        for head, mapping in label_maps.items()
    }
    head_names = list(head_dims.keys())

    train_dataset = StructuredDataset(
        metadata_file=info["train_file"],
        task=args.dataset,
        label_maps=label_maps,
        split=info["train_split"],
        image_size=args.image_size,
    )

    val_dataset = StructuredDataset(
        metadata_file=info["val_file"],
        task=args.dataset,
        label_maps=label_maps,
        split=info["val_split"],
        image_size=args.image_size,
    )

    test_dataset = StructuredDataset(
        metadata_file=info["test_file"],
        task=args.dataset,
        label_maps=label_maps,
        split=info["test_split"],
        image_size=args.image_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StructuredCNN(head_dims).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    artifacts_dir = project_root / "artifacts" / args.dataset / "structured_cnn"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    history = []

    print(f"Dataset: {args.dataset}")
    print(f"Metadata source: {info['source']}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Heads: {head_dims}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_batches = 0

        for images, targets, _meta in train_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)

            loss = 0.0
            for head in head_names:
                loss = loss + loss_fn(outputs[head], targets[head])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        train_loss = total_loss / total_batches
        val_metrics = evaluate(model, val_loader, loss_fn, device, head_names)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_exact_structured_accuracy": val_metrics["exact_structured_accuracy"],
            "val_head_accuracy": val_metrics["head_accuracy"],
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_exact_structured_acc={val_metrics['exact_structured_accuracy']:.4f}"
        )

        if val_metrics["exact_structured_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["exact_structured_accuracy"]
            torch.save(model.state_dict(), artifacts_dir / "best_model.pth")

    model.load_state_dict(torch.load(artifacts_dir / "best_model.pth", map_location=device))

    test_metrics = evaluate(model, test_loader, loss_fn, device, head_names)

    save_predictions(
        model=model,
        loader=test_loader,
        device=device,
        task=args.dataset,
        head_names=head_names,
        inverse_maps=inverse_maps,
        output_file=artifacts_dir / "test_predictions.json",
    )

    metrics = {
        "dataset": args.dataset,
        "model": "structured_cnn",
        "metadata_source": info["source"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "heads": head_dims,
        "best_val_exact_structured_accuracy": best_val_acc,
        "test_exact_structured_accuracy": test_metrics["exact_structured_accuracy"],
        "test_head_accuracy": test_metrics["head_accuracy"],
        "history": history,
    }

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(artifacts_dir / "best_model.pth")
    print(artifacts_dir / "metrics.json")
    print(artifacts_dir / "test_predictions.json")

    print(f"\nFinal Test Exact Structured Accuracy: {test_metrics['exact_structured_accuracy']:.4f}")
    print("Final Test Head Accuracies:")
    for head, acc in test_metrics["head_accuracy"].items():
        print(f"  {head}: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="shapes", choices=["shapes", "numbers", "tictactoe"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
