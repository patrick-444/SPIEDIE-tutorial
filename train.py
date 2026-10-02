"""Train a small feedforward network on the digits dataset.

Uses Keras with the PyTorch backend. The dataset ships with scikit-learn, so
the job does not need a network connection on the compute node.

Example:
    python train.py --lr 0.001 --epochs 50 --seed 42
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import argparse
import csv
from pathlib import Path

import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small ANN on the digits dataset (Keras, PyTorch backend)."
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


class EpochLogger(keras.callbacks.Callback):
    """Print one line per epoch so SLURM logs update while the job runs."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"epoch {epoch + 1}/{self.params['epochs']} "
            f"loss={logs['loss']:.4f} accuracy={logs['accuracy']:.4f} "
            f"val_loss={logs['val_loss']:.4f} val_accuracy={logs['val_accuracy']:.4f}",
            flush=True,
        )


def build_model(n_features: int, n_classes: int, learning_rate: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


RESULT_COLUMNS = [
    "model",
    "dataset",
    "backend",
    "architecture",
    "optimizer",
    "learning_rate",
    "epochs",
    "batch_size",
    "seed",
    "n_samples",
    "n_features",
    "n_classes",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "test_loss",
    "test_accuracy",
]


def write_results(path: Path, row: dict) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=args.seed,
        stratify=data.target,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=args.seed,
        stratify=y_train,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_val = scaler.transform(x_val).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")

    print(
        f"backend={keras.backend.backend()} lr={args.lr} "
        f"epochs={args.epochs} seed={args.seed}",
        flush=True,
    )

    batch_size = 32
    model = build_model(x_train.shape[1], len(data.target_names), args.lr)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[EpochLogger()],
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"test loss: {test_loss:.4f}", flush=True)
    print(f"test accuracy: {test_accuracy:.4f}", flush=True)

    final = history.history
    results_path = results_dir / "results.csv"
    write_results(
        results_path,
        {
            "model": "feedforward_ann",
            "dataset": "digits",
            "backend": keras.backend.backend(),
            "architecture": "Dense(64, relu) -> Dense(32, relu) -> Dense(10, softmax)",
            "optimizer": "adam",
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": batch_size,
            "seed": args.seed,
            "n_samples": int(data.data.shape[0]),
            "n_features": int(data.data.shape[1]),
            "n_classes": int(len(data.target_names)),
            "train_loss": round(float(final["loss"][-1]), 4),
            "train_accuracy": round(float(final["accuracy"][-1]), 4),
            "val_loss": round(float(final["val_loss"][-1]), 4),
            "val_accuracy": round(float(final["val_accuracy"][-1]), 4),
            "test_loss": round(float(test_loss), 4),
            "test_accuracy": round(float(test_accuracy), 4),
        },
    )
    print(f"Wrote {results_path}", flush=True)


if __name__ == "__main__":
    main()
