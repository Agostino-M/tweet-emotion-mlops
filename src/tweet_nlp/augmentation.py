"""Training-time data augmentation via WordNet synonym replacement.

Only applied to the training split, never to validation/test or at inference time.
"""

from __future__ import annotations

import random
from functools import lru_cache

import pandas as pd

from tweet_nlp.config import AugmentationConfig


@lru_cache(maxsize=1)
def _wordnet():
    import nltk
    from nltk.corpus import wordnet

    try:
        wordnet.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    return wordnet


def synonym_replacement(text: str, n: int = 1, rng: random.Random | None = None) -> str:
    """Replace up to `n` random words that have a WordNet synset with their first synonym."""
    rng = rng or random.Random()
    wordnet = _wordnet()
    words = text.split()
    new_words = words.copy()

    candidates = [w for w in words if wordnet.synsets(w)]
    for _ in range(n):
        if not candidates:
            break
        word = rng.choice(candidates)
        synonym = wordnet.synsets(word)[0].lemmas()[0].name().replace("_", " ")
        new_words = [synonym if w == word else w for w in new_words]
    return " ".join(new_words)


def augment(df: pd.DataFrame, cfg: AugmentationConfig, seed: int = 42) -> pd.DataFrame:
    """Return `df` plus augmented rows according to the configured strategy."""
    if cfg.strategy == "none":
        return df

    rng = random.Random(seed)
    new_rows: list[tuple[str, int]] = []

    if cfg.strategy == "balance":
        max_count = df["label"].value_counts().max()
        for label, group in df.groupby("label"):
            texts = group["text"].tolist()
            for _ in range(max_count - len(group)):
                new_rows.append((synonym_replacement(rng.choice(texts), cfg.n_replacements, rng),
                                 int(label)))
    elif cfg.strategy == "multiply":
        for text, label in zip(df["text"], df["label"], strict=True):
            for _ in range(cfg.factor):
                new_rows.append((synonym_replacement(text, cfg.n_replacements, rng), int(label)))
    else:
        raise ValueError(f"Unknown augmentation strategy {cfg.strategy!r}")

    aug = pd.DataFrame(new_rows, columns=["text", "label"])
    return pd.concat([df, aug], ignore_index=True)
