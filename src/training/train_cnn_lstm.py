from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.tokenised_data import SeqDataset, Vocabulary
from src.models.cnn_lstm import CNN_LSTM


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]

    max_len = max(len(c) for c in captions)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)

    for i, c in enumerate(captions):
        padded[i, :len(c)] = c

    return images, padded


def build_vocab_from_files(files):
    vocab = Vocabulary()

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    vocab.add_sentence(row["caption"])

    return vocab


def decode_tokens(tokens, vocab):
    words = []

    for token in tokens:
        word = vocab.idx2word.get(int(token), "<unk>")

        if word == "<end>":
            break

        if word in {"<pad>", "<start>"}:
            continue

        words.append(word)

    return " ".join(words)


def evaluate(model, loader, vocab_size, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    correct_tokens = 0
    total_tokens = 0

    correct_sentences = 0
    total_sentences = 0

    with torch.no_grad():
        for images, padded in loader:
            images = images.to(device)
            padded = padded.to(device)

            outputs = model(images, padded[:, :-1])
            targets = padded[:, 1:]

            # Remove first output step because decoder prepends image feature.
            outputs = outputs[:, 1:, :]

            loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()
            total_batches += 1

            preds = outputs.argmax(dim=2)

            for i in range(preds.size(0)):
                pred_seq = preds[i]
                target_seq = targets[i]

                mask = target_seq != 0

                correct_tokens += ((pred_seq == target_seq) & mask).sum().item()
                total_tokens += mask.sum().item()

                if torch.all(pred_seq[mask] == target_seq[mask]):
                    correct_sentences += 1

                total_sentences += 1

    return {
        "loss": total_loss / total_batches,
        "token_accuracy": correct_tokens / total_tokens,
        "sentence_accuracy": correct_sentences / total_sentences,
    }


def save_test_predictions(model, loader, vocab, device, output_file: Path):
    model.eval()
    predictions = []

    with torch.no_grad():
        sample_counter = 1

        for images, padded in loader:
            images = images.to(device)
            padded = padded.to(device)

            outputs = model(images, padded[:, :-1])
            outputs = outputs[:, 1:, :]

            preds = outputs.argmax(dim=2)
            targets = padded[:, 1:]

            for i in range(preds.size(0)):
                pred_caption = decode_tokens(preds[i].tolist(), vocab)
                true_caption = decode_tokens(targets[i].tolist(), vocab)

                predictions.append({
                    "id": f"sample_{sample_counter:06d}",
                    "true_caption": true_caption,
                    "pred_caption": pred_caption,
                    "exact_match": pred_caption == true_caption,
                })

                sample_counter += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="shapes", choices=["shapes", "numbers", "tictactoe"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    metadata_dir = project_root / "data" / "processed" / args.dataset / "metadata"

    single_file = metadata_dir / f"{args.dataset}_metadata.jsonl"
    train_file = metadata_dir / "train.jsonl"
    val_file = metadata_dir / "val.jsonl"
    test_file = metadata_dir / "test.jsonl"

    if single_file.exists():
        metadata_source = str(single_file)

        full_dataset = SeqDataset(single_file, split=None)
        vocab = full_dataset.vocab

        train_dataset = SeqDataset(single_file, split="train", vocab=vocab)
        val_dataset = SeqDataset(single_file, split="val", vocab=vocab)
        test_dataset = SeqDataset(single_file, split="test", vocab=vocab)

    elif train_file.exists() and val_file.exists() and test_file.exists():
        metadata_source = f"{train_file}, {val_file}, {test_file}"

        vocab = build_vocab_from_files([train_file, val_file, test_file])

        train_dataset = SeqDataset(train_file, split=None, vocab=vocab)
        val_dataset = SeqDataset(val_file, split=None, vocab=vocab)
        test_dataset = SeqDataset(test_file, split=None, vocab=vocab)

    else:
        raise FileNotFoundError(f"No valid metadata found in {metadata_dir}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    vocab_size = len(vocab.word2idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_LSTM(vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    artifacts_dir = project_root / "artifacts" / args.dataset / "cnn_lstm"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    history = []

    print(f"Dataset: {args.dataset}")
    print(f"Metadata source: {metadata_source}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocab size: {vocab_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for images, padded in train_loader:
            images = images.to(device)
            padded = padded.to(device)

            outputs = model(images, padded[:, :-1])
            targets = padded[:, 1:]

            outputs = outputs[:, 1:, :]

            loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        train_loss = total_loss / total_batches
        val_metrics = evaluate(model, val_loader, vocab_size, loss_fn, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_token_accuracy": val_metrics["token_accuracy"],
            "val_sentence_accuracy": val_metrics["sentence_accuracy"],
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_token_acc={val_metrics['token_accuracy']:.4f} | "
            f"val_sentence_acc={val_metrics['sentence_accuracy']:.4f}"
        )

        if val_metrics["token_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["token_accuracy"]
            torch.save(model.state_dict(), artifacts_dir / "best_model.pth")

    model.load_state_dict(torch.load(artifacts_dir / "best_model.pth", map_location=device))

    test_metrics = evaluate(model, test_loader, vocab_size, loss_fn, device)

    save_test_predictions(
        model=model,
        loader=test_loader,
        vocab=vocab,
        device=device,
        output_file=artifacts_dir / "test_predictions.json",
    )

    summary = {
        "dataset": args.dataset,
        "model": "cnn_lstm",
        "metadata_source": metadata_source,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "vocab_size": vocab_size,
        "best_val_token_accuracy": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_token_accuracy": test_metrics["token_accuracy"],
        "test_sentence_accuracy": test_metrics["sentence_accuracy"],
        "history": history,
    }

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(artifacts_dir / "best_model.pth")
    print(artifacts_dir / "metrics.json")
    print(artifacts_dir / "test_predictions.json")

    print(f"\nFinal Test Token Accuracy: {test_metrics['token_accuracy']:.4f}")
    print(f"Final Test Sentence Accuracy: {test_metrics['sentence_accuracy']:.4f}")


if __name__ == "__main__":
    train()
