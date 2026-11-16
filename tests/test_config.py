from pathlib import Path

import pytest

from tweet_nlp.config import Config

CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("task,num_labels", [("emotion", 4)])
def test_shipped_configs_load(task, num_labels):
    cfg = Config.from_yaml(CONFIG_DIR / f"{task}.yaml")
    assert cfg.task == task
    assert cfg.num_labels == num_labels
    assert cfg.training.learning_rate > 0


def test_invalid_task_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"task": "sarcasm", "labels": ["a", "b"]})


def test_roundtrip_dict():
    cfg = Config.from_yaml(CONFIG_DIR / "emotion.yaml")
    assert Config.from_dict(cfg.to_dict()) == cfg
