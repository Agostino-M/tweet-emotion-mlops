"""Tweet normalisation shared by training and serving.

The same `clean_text` must run at training time and at inference time, otherwise the model
sees a different input distribution in production (training/serving skew).

Pipeline: remove URLs & mentions -> emoticons to text -> emojis to text -> split hashtags -> normalise whitespace.
"""

from __future__ import annotations

import re
from functools import lru_cache

_URL_MENTION_RE = re.compile(r"http\S+|www\S+|@\w+")
_HASH_NO_SPACE_RE = re.compile(r"(?<!\s)(#)")
_WS_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _emot():
    import emot

    return emot.core.emot()


@lru_cache(maxsize=1)
def _wordsegment():
    import wordsegment

    wordsegment.load()
    return wordsegment


def sanitize_text(text: str) -> str:
    """Replace literal and real newlines/tabs with spaces and collapse whitespace."""
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    return _WS_RE.sub(" ", text).strip()


def convert_emoticons_to_text(text: str) -> str:
    """`:)` -> ` Happy face or smiley `."""
    found = _emot().emoticons(text)
    for emoticon, meaning in zip(found["value"], found["mean"], strict=False):
        text = text.replace(emoticon, f" {meaning} ")
    return text


def convert_emojis_to_text(text: str) -> str:
    """`😊` -> ` smiling face with smiling eyes `."""
    found = _emot().emoji(text)
    for emoji, meaning in zip(found["value"], found["mean"], strict=False):
        if meaning:
            meaning = meaning.replace(":", "").replace("_", " ")
            text = text.replace(emoji, f" {meaning} ")
    return text


def split_hashtags(text: str) -> str:
    """`#easterEgg` -> `easter egg` using word segmentation."""
    ws = _wordsegment()
    text = _HASH_NO_SPACE_RE.sub(r" \1", text)
    words = []
    for word in text.split():
        if word.startswith("#") and len(word) > 1:
            words.append(" ".join(ws.segment(word.lstrip("#"))) or word)
        else:
            words.append(word)
    return " ".join(words)


def clean_text(text: str) -> str:
    """Full preprocessing pipeline applied to every tweet."""
    text = _URL_MENTION_RE.sub("", text)
    text = convert_emoticons_to_text(text)
    text = convert_emojis_to_text(text)
    text = split_hashtags(text)
    return sanitize_text(text)
