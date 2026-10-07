"""Check stored outputs against the data split and the underlying predictions."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from digits_workflow import load_split, make_models


def verify(output):
    output = Path(output)
    info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    with (output / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    with (output / "errors.csv").open(encoding="utf-8", newline="") as handle:
        errors = list(csv.DictReader(handle))
    with (output / "knn_cv.csv").open(encoding="utf-8", newline="") as handle:
        cv = list(csv.DictReader(handle))
    _, X, y, train, test = load_split()
    assert set(train).isdisjoint(test), "Training and test sets overlap"
    assert len(set(train) | set(test)) == len(y), "Split does not cover dataset"
    assert train.tolist() == info["train_indices"] and test.tolist() == info["test_indices"]
    assert set(y[train]) == set(y[test]) == set(range(10))
    assert info["workflow_sha256"] == hashlib.sha256(
        Path(__file__).with_name("digits_workflow.py").read_bytes()).hexdigest(), "Rerun after code changes"
    for row in cv:
        fold_scores = [float(row[f"fold_{i}_accuracy"]) for i in range(1, 6)]
        assert np.isclose(np.mean(fold_scores), float(row["mean_cv_accuracy"]))
        assert np.isclose(np.std(fold_scores), float(row["std_cv_accuracy"]))
    selected = int(max(cv, key=lambda row: float(row["mean_cv_accuracy"]))["k"])
    assert selected == info["selected_k"]
    assert [int(row["k"]) for row in cv] == info["k_candidates"]
    assert {row["model"] for row in predictions} == {row["model"] for row in metrics}
    assert errors == [row for row in predictions if int(row["is_error"])], "Error subset differs"
    for metric in metrics:
        rows = [row for row in predictions if row["model"] == metric["model"]]
        assert len(rows) == len(test) == metric["test_samples"]
        assert [int(row["test_position"]) for row in rows] == list(range(len(test)))
        assert [int(row["dataset_index"]) for row in rows] == test.tolist()
        assert [int(row["true_label"]) for row in rows] == y[test].tolist()
        wrong = sum(int(row["true_label"]) != int(row["predicted_label"]) for row in rows)
        assert all(int(row["is_error"]) == (row["true_label"] != row["predicted_label"]) for row in rows)
        assert wrong == metric["errors"]
        assert metric["correct"] + wrong == len(test)
        assert np.isclose(metric["accuracy"], 1 - wrong / len(test))
    # Check the fitted scaler against training statistics, not full-data statistics.
    model = make_models(selected)["svm"].fit(X[train], y[train])
    assert np.allclose(model.named_steps["standardscaler"].mean_, X[train].mean(axis=0))
    actual = model.predict(X[test])
    saved = [int(row["predicted_label"]) for row in predictions if row["model"] == "svm"]
    assert np.array_equal(actual, saved), "Saved SVM predictions cannot be reproduced"
    print("PASS: split, training-only scaling, CV selection, predictions and metrics are consistent.")
    print("SVM error pairs:", dict(Counter(
        (row["true_label"], row["predicted_label"]) for row in errors if row["model"] == "svm")))


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent / "results")
