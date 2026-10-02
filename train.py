"""Train two classifiers on a small built-in dataset and save the scores.

The wine dataset ships with scikit-learn, so this script does not need a
network connection when it runs on a compute node.
"""

import json
from pathlib import Path

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=0,
        stratify=data.target,
    )

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500),
        ),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=0),
    }

    summary = {
        "dataset": "digits",
        "n_samples": int(data.data.shape[0]),
        "n_features": int(data.data.shape[1]),
        "n_classes": int(len(data.target_names)),
        "models": {},
    }
    report_sections = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(
            y_test,
            predictions,
            target_names=[str(name) for name in data.target_names],
        )
        summary["models"][name] = {"test_accuracy": round(float(accuracy), 4)}
        report_sections.append(f"{name}\ntest accuracy: {accuracy:.4f}\n\n{report}")
        print(f"{name}: test accuracy {accuracy:.4f}")

    metrics_path = results_dir / "metrics.json"
    report_path = results_dir / "classification_report.txt"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path.write_text("\n".join(report_sections))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
