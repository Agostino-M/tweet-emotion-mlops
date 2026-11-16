import pytest

from tweet_nlp.model import TweetClassifier

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

TINY = "hf-internal-testing/tiny-random-RobertaModel"


@pytest.mark.network
@pytest.mark.parametrize("num_labels", [2, 4])
def test_forward_returns_logits_per_class(num_labels):
    tok = transformers.AutoTokenizer.from_pretrained(TINY)
    model = TweetClassifier(TINY, num_labels=num_labels, hidden_dim=8).eval()
    enc = tok(["hello world", "a much longer second tweet"], padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
    assert logits.shape == (2, num_labels)
