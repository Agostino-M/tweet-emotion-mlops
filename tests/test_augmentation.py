import random

import pandas as pd
import pytest

from tweet_nlp.augmentation import augment, synonym_replacement
from tweet_nlp.config import AugmentationConfig

# Small explicit list: the tests should not depend on the implementation's stopword source.
STOPWORDS = {"i", "am", "was", "in", "the", "who", "me"}
SEEDS = [0, 1, 2]


def _df():
    # Every sentence has exactly one replaceable word ("afraid" would not work: WordNet has
    # no single-word synonym for it).
    return pd.DataFrame({
        "text": ["I am happy", "I am sad", "I am angry", "I am scared"],
        "label": [0, 0, 0, 1],
    })


def test_none_strategy_is_noop():
    df = _df()
    assert augment(df, AugmentationConfig(strategy="none")).equals(df)


def test_balance_equalises_classes():
    out = augment(_df(), AugmentationConfig(strategy="balance", n_replacements=1))
    counts = out["label"].value_counts()
    assert counts[0] == counts[1] == 3


def test_multiply_adds_factor_copies():
    out = augment(_df(), AugmentationConfig(strategy="multiply", factor=2))
    rows = _df().shape[0]
    assert len(out) == rows + rows * 2


def test_augmented_rows_differ_from_source():
    # With factor=1 the output is [originals..., augmented...] in the same order.
    out = augment(_df(), AugmentationConfig(strategy="multiply", factor=1))
    rows = _df().shape[0]
    originals, augmented = out["text"][:rows].tolist(), out["text"][rows:].tolist()

    assert all(o != a for o, a in zip(originals, augmented, strict=True))


def test_augmentation_is_deterministic():
    cfg = AugmentationConfig(strategy="multiply", factor=1)
    assert augment(_df(), cfg, seed=1).equals(augment(_df(), cfg, seed=1))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("text", ["I am happy", "I was in the house who loves me"])
def test_stopwords_are_kept(text, seed):
    out = synonym_replacement(text, n=5, rng=random.Random(seed))

    # strict=True also checks that the number of words is preserved.
    for orig, new in zip(text.split(), out.split(), strict=True):
        if orig.lower() in STOPWORDS:
            assert new == orig


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("text", ["I am happy", "I am sad"])
def test_synonym_differs_from_original(text, seed):
    out = synonym_replacement(text, n=1, rng=random.Random(seed))

    assert out.split() != text.split()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", [1, 2])
def test_at_most_n_words_change(n, seed):
    text = "the movie was great and the actors were brilliant"
    out = synonym_replacement(text, n=n, rng=random.Random(seed))

    changed = sum(o != w for o, w in zip(text.split(), out.split(), strict=True))
    assert changed <= n
