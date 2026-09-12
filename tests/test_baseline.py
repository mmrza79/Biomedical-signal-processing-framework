"""Checks for participant-disjoint baseline evaluation infrastructure."""

import numpy as np
import pytest

from src.models.baseline import build_classifier_pipeline, evaluate_grouped_model


def test_loso_holds_out_whole_subjects() -> None:
    features = np.array(
        [
            [-2.0],
            [2.0],
            [-1.8],
            [1.8],
            [-2.2],
            [2.2],
        ]
    )
    targets = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array(["s1", "s1", "s2", "s2", "s3", "s3"])
    pipeline = build_classifier_pipeline("logistic_regression", random_seed=7)

    predictions, folds = evaluate_grouped_model(
        pipeline, features, targets, groups, strategy="loso"
    )

    assert predictions.shape == targets.shape
    assert len(folds) == 3
    assert all(len(record["test_groups"]) == 1 for record in folds)
    assert all(
        set(record["train_groups"]).isdisjoint(record["test_groups"])
        for record in folds
    )


def test_grouped_evaluation_requires_multiple_subjects() -> None:
    pipeline = build_classifier_pipeline("svm")

    with pytest.raises(ValueError, match="at least two groups"):
        evaluate_grouped_model(
            pipeline,
            [[0.0], [1.0]],
            [0, 1],
            ["same", "same"],
        )

