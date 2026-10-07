"""Shared experiment functions for the script and the learning notebook."""
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
K_VALUES = [1, 3, 5, 7, 9]


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_split():
    digits = load_digits()
    # One image becomes one row with 64 pixel features.
    X = digits.images.reshape(len(digits.images), -1)
    y = digits.target
    train, test = train_test_split(np.arange(len(y)), test_size=0.25,
                                   random_state=SEED, stratify=y)
    return digits, X, y, train, test


def make_models(best_k):
    # Each fit learns its scaler from the supplied training data only.
    return {
        "svm": make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "knn_tuned": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)),
    }


def tune_knn(X_train, y_train, output):
    """Select k using training data only. Test data is never passed in."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    search = GridSearchCV(
        make_pipeline(StandardScaler(), KNeighborsClassifier()),
        {"kneighborsclassifier__n_neighbors": K_VALUES},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=1, refit=False,
    )
    search.fit(X_train, y_train)
    rows = []
    for i, params in enumerate(search.cv_results_["params"]):
        row = {"k": params["kneighborsclassifier__n_neighbors"],
               "mean_cv_accuracy": float(search.cv_results_["mean_test_score"][i]),
               "std_cv_accuracy": float(search.cv_results_["std_test_score"][i])}
        for fold in range(5):
            row[f"fold_{fold + 1}_accuracy"] = float(search.cv_results_[f"split{fold}_test_score"][i])
        rows.append(row)
    # The first candidate wins if mean CV scores tie.
    best_k = int(search.best_params_["kneighborsclassifier__n_neighbors"])
    write_csv(output / "knn_cv.csv", rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar([r["k"] for r in rows], [r["mean_cv_accuracy"] for r in rows],
                yerr=[r["std_cv_accuracy"] for r in rows], fmt="o-", capsize=4)
    ax.set(xlabel="Number of neighbours (k)", ylabel="Training CV accuracy",
           title="5-fold CV: mean and fold standard deviation", xticks=K_VALUES)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "knn_cv.png", dpi=140)
    plt.close(fig)
    return best_k, rows


def evaluate_models(models, digits, X, y, train, test, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, predictions = [], []
    for name, model in models.items():
        model.fit(X[train], y[train])
        prediction = model.predict(X[test])
        report = classification_report(y[test], prediction, zero_division=0, digits=4)
        (output / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(y[test], prediction, labels=range(10), ax=ax)
        ax.set_title(f"{name.upper()} confusion matrix")
        fig.tight_layout()
        fig.savefig(output / f"{name}_confusion_matrix.png", dpi=140)
        plt.close(fig)
        wrong = np.flatnonzero(prediction != y[test])
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for ax in axes.flat:
            ax.axis("off")
        # First 8 errors in test-array order, not a curated or random sample.
        for ax, i in zip(axes.flat, wrong[:8]):
            ax.imshow(digits.images[test[i]], cmap="gray_r")
            ax.set_title(f"id={test[i]}\ntrue={y[test[i]]}, pred={prediction[i]}")
        if not len(wrong):
            fig.suptitle("No errors on this test split")
        fig.tight_layout()
        fig.savefig(output / f"{name}_errors.png", dpi=140)
        plt.close(fig)
        knn = model.named_steps.get("kneighborsclassifier")
        results.append({"model": name, "accuracy": float(accuracy_score(y[test], prediction)),
                        "correct": int(np.sum(prediction == y[test])), "errors": int(len(wrong)),
                        "train_samples": len(train), "test_samples": len(test), "split_seed": SEED,
                        "k": knn.n_neighbors if knn is not None else None})
        for position, (idx, actual, predicted) in enumerate(zip(test, y[test], prediction)):
            predictions.append({"model": name, "test_position": position, "dataset_index": int(idx),
                                "true_label": int(actual), "predicted_label": int(predicted),
                                "is_error": int(actual != predicted)})
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "errors.csv", [r for r in predictions if r["is_error"]], list(predictions[0]))
    write_csv(output / "metrics.csv", results)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def save_run_info(output, train, test, best_k, cv_rows):
    output = Path(output)
    versions = {package: version(package) for package in ["numpy", "scipy", "scikit-learn", "matplotlib"]}
    (output / "environment.txt").write_text(
        "\n".join(f"{package}=={number}" for package, number in versions.items()) + "\n", encoding="utf-8")
    info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "packages": versions,
        "dataset": "sklearn.datasets.load_digits", "seed": SEED, "test_size": 0.25,
        "train_indices": train.tolist(), "test_indices": test.tolist(),
        "selection": "5-fold StratifiedKFold on training data; accuracy; refit=False",
        "k_candidates": K_VALUES, "selected_k": best_k,
        "best_cv_accuracy": max(row["mean_cv_accuracy"] for row in cv_rows),
        "workflow_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def run(output):
    digits, X, y, train, test = load_split()
    best_k, cv_rows = tune_knn(X[train], y[train], output)
    results = evaluate_models(make_models(best_k), digits, X, y, train, test, output)
    save_run_info(output, train, test, best_k, cv_rows)
    print(f"Selected k={best_k} using training CV only")
    for row in results:
        print(f"{row['model']}: accuracy={row['accuracy']:.4f}, errors={row['errors']}")
    print(f"Results saved in {Path(output).resolve()}")
    return results
