"""Run subject-independent classical baselines on a verified feature table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.models.baseline import (  # noqa: E402
    build_classifier_pipeline,
    evaluate_grouped_model,
)


def parse_args() -> argparse.Namespace:
    """Parse baseline experiment arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate one classical model with participant-disjoint folds."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument(
        "--model",
        choices=("logistic_regression", "svm", "random_forest"),
        default="logistic_regression",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return config


def _resolve_repository_path(path_value: str) -> Path:
    """Resolve a configured repository-relative path safely."""

    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> None:
    """Evaluate configured feature data and write non-fabricated artifacts."""

    args = parse_args()
    config = _load_config(args.config)
    data_config = config.get("data", {})
    evaluation_config = config.get("evaluation", {})
    models_config = config.get("models", {})
    feature_path = _resolve_repository_path(
        str(data_config.get("feature_table", "data/processed/features.csv"))
    )
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Verified feature table not found: {feature_path}. Integrate the real "
            "dataset and construct participant-labelled features before running a baseline."
        )

    table = pd.read_csv(feature_path)
    subject_column = str(data_config.get("subject_column", "subject_id"))
    target_column = str(data_config.get("target_column", "gait_phase"))
    required_columns = {subject_column, target_column}
    missing = required_columns - set(table.columns)
    if missing:
        raise ValueError(f"feature table is missing required columns: {sorted(missing)}")

    feature_columns = [
        column for column in table.columns if column not in {subject_column, target_column}
    ]
    if not feature_columns:
        raise ValueError("feature table contains no feature columns")
    features = table[feature_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(features)):
        raise ValueError("feature columns contain missing or non-finite values")

    model_options = models_config.get(args.model, {})
    if not isinstance(model_options, dict):
        raise ValueError(f"models.{args.model} must be a YAML mapping")
    pipeline = build_classifier_pipeline(
        args.model,
        random_seed=int(config.get("random_seed", 42)),
        model_options=model_options,
    )
    predictions, folds = evaluate_grouped_model(
        pipeline,
        features,
        table[target_column].to_numpy(),
        table[subject_column].to_numpy(),
        strategy=str(evaluation_config.get("strategy", "loso")),
        n_splits=int(evaluation_config.get("group_k_folds", 5)),
    )
    metrics = classification_metrics(table[target_column].to_numpy(), predictions)

    table_directory = REPOSITORY_ROOT / "results" / "tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    stem = args.model
    prediction_table = pd.DataFrame(
        {
            subject_column: table[subject_column],
            "y_true": table[target_column],
            "y_pred": predictions,
        }
    )
    prediction_table.to_csv(table_directory / f"{stem}_predictions.csv", index=False)
    with (table_directory / f"{stem}_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"metrics": metrics, "folds": folds}, stream, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
