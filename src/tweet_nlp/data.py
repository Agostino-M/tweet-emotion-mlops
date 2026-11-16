"""Dataset download and loading.

Data comes from TweetEval (Barbieri et al., 2020) on the Hugging Face Hub:
https://huggingface.co/datasets/cardiffnlp/tweet_eval

Raw splits are stored as parquet files in `data/raw/<task>/<split>.parquet`
with two columns: `text` (str) and `label` (int).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tweet_nlp.config import TASKS, Task

HF_DATASET = "cardiffnlp/tweet_eval"
SPLITS = ("train", "validation", "test")
DEFAULT_DATA_DIR = Path("data/raw")


def split_path(task: Task, split: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return Path(data_dir) / task / f"{split}.parquet"


def download(task: Task, data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> None:
    """Download all splits of a TweetEval task and save them as parquet."""
    from datasets import load_dataset

    if not force and all(split_path(task, s, data_dir).exists() for s in SPLITS):
        print(f"[{task}] already downloaded in {Path(data_dir) / task}")
        return

    ds = load_dataset(HF_DATASET, task)
    for split in SPLITS:
        out = split_path(task, split, data_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = ds[split].to_pandas()[["text", "label"]]
        df.to_parquet(out, index=False)
        print(f"[{task}] {split:<10} {len(df):>5} rows -> {out}")


def load_split(task: Task, split: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = split_path(task, split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `tweet-nlp-data --task {task}` first.")
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TweetEval splits.")
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    for task in TASKS if args.task == "all" else (args.task,):
        download(task, args.data_dir, args.force)


if __name__ == "__main__":
    main()
