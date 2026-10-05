"""Assistant-authored learning scaffold; inspired by scikit-learn's digits example."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    digits = load_digits()
    # Each 8 x 8 image becomes 64 numeric features.
    X = digits.images.reshape(len(digits.images), -1)
    y = digits.target
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=42, stratify=y
    )
    # Scaling is fitted only on training data. Hyperparameters are fixed in advance.
    models = {
        "svm": make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    results = []
    for name, model in models.items():
        model.fit(X[train], y[train])
        prediction = model.predict(X[test])
        accuracy = float(accuracy_score(y[test], prediction))
        report = classification_report(y[test], prediction, zero_division=0)
        (output / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")
        ConfusionMatrixDisplay.from_predictions(y[test], prediction, labels=range(10))
        plt.title(f"{name.upper()} confusion matrix")
        plt.savefig(output / f"{name}_confusion_matrix.png", bbox_inches="tight")
        plt.close()
        wrong = np.flatnonzero(prediction != y[test])[:8]
        if len(wrong):
            fig, axes = plt.subplots(2, 4, figsize=(10, 5))
            for ax in axes.flat:
                ax.axis("off")
            for ax, i in zip(axes.flat, wrong):
                ax.imshow(digits.images[test[i]], cmap="gray_r")
                ax.set_title(f"true={y[test[i]]}, pred={prediction[i]}")
            fig.tight_layout()
            fig.savefig(output / f"{name}_errors.png")
            plt.close(fig)
        results.append({"model": name, "accuracy": accuracy, "train_samples": len(train),
                        "test_samples": len(test), "split_seed": 42})
        print(f"{name}: accuracy={accuracy:.4f}")
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved in {output}")


if __name__ == "__main__":
    main()
