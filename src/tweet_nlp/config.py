"""Typed configuration loaded from YAML files in `configs/`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml

Task = Literal["emotion"]
TASKS: tuple[Task, ...] = ("emotion",)


@dataclass
class ModelConfig:
    name: str = "roberta-base"
    dropout: float = 0.3
    hidden_dim: int = 64
    max_length: int = 128


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    epochs: int = 10
    patience: int = 3
    min_delta: float = 0.001
    seed: int = 42


@dataclass
class AugmentationConfig:
    strategy: Literal["none", "balance", "multiply"] = "none"
    factor: int = 1
    n_replacements: int = 1


@dataclass
class Config:
    task: Task
    labels: list[str]
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Config:
        if raw.get("task") not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {raw.get('task')!r}")
        return cls(
            task=raw["task"],
            labels=list(raw["labels"]),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            augmentation=AugmentationConfig(**raw.get("augmentation", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))
