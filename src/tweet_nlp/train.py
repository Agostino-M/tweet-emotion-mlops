"""Fine-tune a TweetClassifier on a TweetEval task.

Usage:
    tweet-nlp-train --config configs/emotion.yaml
    tweet-nlp-train --config configs/emotion.yaml --max-samples 64 --epochs 1   # smoke test

Outputs (in `--output-dir`, default `artifacts/<task>/`):
    model.pt        best state_dict (selected on validation macro-F1)
    tokenizer/      tokenizer files
    meta.json       config + labels + metrics, everything needed to rebuild the model
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tweet_nlp.augmentation import augment
from tweet_nlp.config import Config
from tweet_nlp.data import DEFAULT_DATA_DIR, load_split
from tweet_nlp.model import TweetClassifier
from tweet_nlp.preprocessing import clean_text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    """Tokenise per batch with dynamic padding (pads to the longest tweet in the batch)."""
    records = list(zip(df["text"].tolist(), df["label"].tolist(), strict=True))

    def collate(batch):
        texts, labels = zip(*batch, strict=True)
        enc = tokenizer(
            list(texts), truncation=True, max_length=max_length, padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=generator)


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    f1_macro: float
    labels: list[int]
    preds: list[int]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> EvalResult:
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, all_labels, all_preds = 0.0, [], []

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return EvalResult(
        loss=total_loss / max(len(loader), 1),
        accuracy=accuracy_score(all_labels, all_preds),
        f1_macro=f1_score(all_labels, all_preds, average="macro"),
        labels=all_labels,
        preds=all_preds,
    )


def load_and_prepare(cfg: Config, data_dir: Path, max_samples: int | None):
    splits = {}
    for split in ("train", "validation", "test"):
        df = load_split(cfg.task, split, data_dir)
        if max_samples:
            df = df.sample(n=min(max_samples, len(df)), random_state=cfg.training.seed)
        df = df.assign(text=df["text"].map(clean_text))
        splits[split] = df.reset_index(drop=True)
    splits["train"] = augment(splits["train"], cfg.augmentation, cfg.training.seed)
    return splits


def train(
    cfg: Config,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path | None = None,
    max_samples: int | None = None,
    device: str | None = None,
) -> dict:
    output_dir = Path(output_dir or Path("artifacts") / cfg.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.training.seed)
    dev = get_device(device)
    print(f"[{cfg.task}] device={dev} model={cfg.model.name}")

    splits = load_and_prepare(cfg, data_dir, max_samples)
    print(f"[{cfg.task}] sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    t = cfg.training
    train_loader = make_loader(splits["train"], tokenizer, cfg.model.max_length, t.batch_size,
                               shuffle=True, seed=t.seed)
    val_loader = make_loader(splits["validation"], tokenizer, cfg.model.max_length,
                             t.batch_size, shuffle=False)
    test_loader = make_loader(splits["test"], tokenizer, cfg.model.max_length, t.batch_size,
                              shuffle=False)

    model = TweetClassifier(cfg.model.name, cfg.num_labels, cfg.model.hidden_dim, cfg.model.dropout).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.learning_rate,
                                  weight_decay=t.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    history = []
    start = time.time()
    for epoch in range(1, t.epochs + 1):
        tr = run_epoch(model, train_loader, loss_fn, dev, optimizer, desc=f"train {epoch}")
        va = run_epoch(model, val_loader, loss_fn, dev, desc=f"val {epoch}")
        history.append({"epoch": epoch, "train_loss": tr.loss, "train_f1": tr.f1_macro,
                        "val_loss": va.loss, "val_f1": va.f1_macro, "val_acc": va.accuracy})
        print(f"[{cfg.task}] ep {epoch:02d} | train loss {tr.loss:.4f} f1 {tr.f1_macro:.4f} | "
              f"val loss {va.loss:.4f} f1 {va.f1_macro:.4f} acc {va.accuracy:.4f}")

        if va.f1_macro > best_f1 + t.min_delta:
            best_f1, best_epoch, bad_epochs = va.f1_macro, epoch, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= t.patience:
                print(f"[{cfg.task}] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    te = run_epoch(model, test_loader, loss_fn, dev, desc="test")
    print(classification_report(te.labels, te.preds, labels=list(range(cfg.num_labels)),
                                target_names=cfg.labels, digits=4, zero_division=0))

    metrics = {
        "best_epoch": best_epoch,
        "val_f1_macro": best_f1,
        "test_f1_macro": te.f1_macro,
        "test_accuracy": te.accuracy,
        "train_time_sec": round(time.time() - start, 1),
    }
    torch.save(best_state, output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    meta = {"config": cfg.to_dict(), "metrics": metrics, "history": history}
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{cfg.task}] val F1 {best_f1:.4f} | test F1 {te.f1_macro:.4f} -> {output_dir}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tweet classifier.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=None, help="subsample every split (for quick smoke tests)")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--model-name", default=None, help="override encoder (e.g. a tiny model)")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (auto if omitted)")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.model_name:
        cfg.model.name = args.model_name
    train(cfg, args.data_dir, args.output_dir, args.max_samples, args.device)


if __name__ == "__main__":
    main()
