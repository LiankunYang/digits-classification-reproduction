"""Compare SVM and KNN on the built-in digits dataset."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_split():
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1)
    y = digits.target
    train, test = train_test_split(np.arange(len(y)), test_size=0.25,
                                   random_state=42, stratify=y)
    return digits, X, y, train, test


def choose_k(X_train, y_train):
    # Scaling is fitted separately inside each training fold.
    search = GridSearchCV(
        make_pipeline(StandardScaler(), KNeighborsClassifier()),
        {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9]},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy", refit=False, n_jobs=1,
    )
    search.fit(X_train, y_train)
    for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
        print(f"k={params['kneighborsclassifier__n_neighbors']}: CV accuracy={score:.4f}")
    return int(search.best_params_["kneighborsclassifier__n_neighbors"])


def make_models(k):
    return {
        "svm": make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)),
    }


def evaluate(models, digits, X, y, train, test, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for name, model in models.items():
        model.fit(X[train], y[train])
        prediction = model.predict(X[test])
        wrong = np.flatnonzero(prediction != y[test])
        row = {"model": name, "accuracy": float(accuracy_score(y[test], prediction)),
               "errors": int(len(wrong)), "train_samples": len(train),
               "test_samples": len(test), "split_seed": 42}
        if name == "knn":
            row["k"] = model.named_steps["kneighborsclassifier"].n_neighbors
        metrics.append(row)
        print(f"{name}: accuracy={row['accuracy']:.4f}, errors={row['errors']}")
        if name != "svm":
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(y[test], prediction, labels=range(10), ax=ax)
        ax.set_title("SVM confusion matrix")
        fig.tight_layout()
        fig.savefig(output / "svm_confusion_matrix.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for ax in axes.flat:
            ax.axis("off")
        # First eight errors in test-array order.
        for ax, i in zip(axes.flat, wrong[:8]):
            ax.imshow(digits.images[test[i]], cmap="gray_r")
            ax.set_title(f"id={test[i]}\ntrue={y[test[i]]}, pred={prediction[i]}")
        if not len(wrong):
            fig.suptitle("No errors on this test split")
        fig.tight_layout()
        fig.savefig(output / "svm_errors.png", dpi=140)
        plt.close(fig)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "results"
    digits, X, y, train, test = load_split()
    k = choose_k(X[train], y[train])
    print(f"Selected k={k}")
    evaluate(make_models(k), digits, X, y, train, test, output)
