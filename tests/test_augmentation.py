import pandas as pd

from tweet_nlp.augmentation import augment
from tweet_nlp.config import AugmentationConfig


def _df():
    return pd.DataFrame({
        "text": ["I am happy", "I am sad", "I am angry", "I am afraid"],
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
    assert len(out) == 4 * 3


def test_augmentation_is_deterministic():
    cfg = AugmentationConfig(strategy="multiply", factor=1)
    assert augment(_df(), cfg, seed=1).equals(augment(_df(), cfg, seed=1))
