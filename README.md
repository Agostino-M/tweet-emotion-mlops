# tweet-nlp-mlops

End-to-end MLOps project: **emotion classification** (anger / joy / optimism / sadness) on tweets,
from raw data to a monitored API running in the cloud.

The focus is the whole pipeline, not just the model:

| Stage | Tool | Status |
|---|---|---|
| Data download & preprocessing | Hugging Face `datasets`, `emot`, `wordsegment` | done |
| Training (RoBERTa fine-tuning) | PyTorch, Transformers | done |
| Data quality checks | Great Expectations | planned |
| Experiment tracking & model registry | MLflow | planned |
| Optimised inference | ONNX Runtime (int8) | planned |
| Serving | FastAPI | planned |
| Containerisation | Docker | planned |
| CI/CD | GitHub Actions | planned |
| Free always-on demo | Hugging Face Spaces | planned |
| Production deploy (on demand) | AWS ECS Fargate + ALB, Terraform | planned |
| Monitoring | Prometheus metrics, Grafana | planned |

## Data

[TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (Barbieri et al., 2020),
`emotion` subset. The data is downloaded on demand and is not stored in this repository.

| Classes | Train | Validation | Test | Metric |
|---|---|---|---|---|
| 4 | 3,257 | 374 | 1,421 | macro-F1 |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra train                 # create .venv with training deps
uv run tweet-nlp-data                 # download TweetEval emotion to data/raw/
uv run pytest                         # run tests
uv run ruff check .                   # lint

# smoke test on CPU with a tiny model
uv run tweet-nlp-train --config configs/emotion.yaml \
    --model-name hf-internal-testing/tiny-random-RobertaModel --max-samples 64 --epochs 1

# full training (GPU recommended)
uv run tweet-nlp-train --config configs/emotion.yaml
```

Artifacts are written to `artifacts/<task>/` (`model.pt`, `tokenizer/`, `meta.json`).

## Project layout

```
configs/            hyper-parameters (YAML)
src/tweet_nlp/
  config.py         typed config loading
  data.py           TweetEval download / loading
  preprocessing.py  tweet cleaning shared by training and serving
  augmentation.py   WordNet synonym replacement (train split only)
  model.py          RoBERTa encoder + MLP head
  train.py          training loop, early stopping on val macro-F1, test evaluation
tests/
```

## Origin

This project is a rebuild of a university NLP team assignment
([Agostino-M/NLPAssignments](https://github.com/Agostino-M/NLPAssignments/)), originally a set of
notebooks. Here the task is reimplemented from scratch as a production-style pipeline.

## Citation

```bibtex
@inproceedings{barbieri2020tweeteval,
  title     = {{TweetEval}: Unified Benchmark and Comparative Evaluation for Tweet Classification},
  author    = {Barbieri, Francesco and Camacho-Collados, Jose and Espinosa Anke, Luis and Neves, Leonardo},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  year      = {2020}
}
```

## License

[MIT](LICENSE)
