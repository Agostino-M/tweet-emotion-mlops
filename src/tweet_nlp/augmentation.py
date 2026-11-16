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


@lru_cache(maxsize=1)
def _stopwords():
    import nltk
    from nltk.corpus import stopwords

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def _pos_tag():
    import nltk

    try:
        nltk.pos_tag(["test"])
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    return nltk.pos_tag


# First letter of a Penn Treebank tag -> WordNet part of speech.
_PENN_TO_WORDNET = {"J": "a", "N": "n", "V": "v", "R": "r"}


@lru_cache(maxsize=50_000)
def _synonyms(word: str, pos: str) -> tuple[str, ...]:
    """Single-word WordNet synonyms of `word` with part of speech `pos`, excluding the word itself.

    Sorted so that `rng.choice` is reproducible across processes (set order is not).
    """
    names = {
        lemma.name()
        for synset in _wordnet().synsets(word, pos=pos)
        for lemma in synset.lemmas()
    }
    return tuple(sorted(
        name for name in names if "_" not in name and name.lower() != word.lower()
    ))


def synonym_replacement(text: str, n: int = 1, rng: random.Random | None = None) -> str:
    """Replace up to `n` distinct words with a random WordNet synonym of the same part of speech.

    Stopwords, words with an unsupported part of speech and words without synonyms are never
    replaced. The number of words is preserved (multi-word synonyms are skipped).
    """
    rng = rng or random.Random()
    words = text.split()
    stopwords = _stopwords()

    # position -> possible synonyms, only for replaceable words
    options: dict[int, tuple[str, ...]] = {}
    for i, (word, tag) in enumerate(_pos_tag()(words)):
        pos = _PENN_TO_WORDNET.get(tag[:1])
        if pos is None or word.lower() in stopwords:
            continue
        if synonyms := _synonyms(word, pos):
            options[i] = synonyms

    new_words = words.copy()
    for i in rng.sample(sorted(options), k=min(n, len(options))):
        new_words[i] = rng.choice(options[i])
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
