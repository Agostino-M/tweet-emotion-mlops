from tweet_nlp.preprocessing import (
    clean_text,
    convert_emojis_to_text,
    convert_emoticons_to_text,
    sanitize_text,
    split_hashtags,
)


def test_sanitize_collapses_whitespace_and_literal_escapes():
    assert sanitize_text("  a\tb\\nc \n\n d  ") == "a b c d"


def test_removes_urls_and_mentions():
    out = clean_text("hello @alberto look https://example.com/x?y=1 www.foo.it now")
    assert "@" not in out
    assert "http" not in out and "www" not in out
    assert out == "hello look now"


def test_emoji_converted_to_words():
    out = convert_emojis_to_text("great 😊")
    assert "😊" not in out
    assert "smiling" in out
    assert ":" not in out and "_" not in out


def test_emoticon_converted_to_words():
    out = convert_emoticons_to_text("so good :)")
    assert ":)" not in out
    assert out.strip() != "so good"


def test_hashtags_are_segmented():
    assert split_hashtags("#easterEgg") == "easter egg"
    assert split_hashtags("love it#happyday") == "love it happy day"


def test_lone_hash_is_kept():
    assert split_hashtags("# 1") == "# 1"


def test_clean_text_full_pipeline():
    out = clean_text("This is a\ttest!\n😊 #happy @alberto https://w3schools.com")
    assert out.startswith("This is a test!")
    assert "happy" in out
    assert "@" not in out and "http" not in out
    assert "  " not in out


def test_clean_text_is_idempotent_on_plain_text():
    text = "just a normal sentence"
    assert clean_text(text) == text
