import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from .config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from .data import load_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()

    model_path, names_path = MODEL_DIR / "crop_disease_model.keras", MODEL_DIR / "class_names.json"
    if not model_path.exists() or not names_path.exists():
        raise SystemExit("Model files not found. Train the model first.")

    _, _, test_ds, class_names = load_datasets(Path(args.data), args.batch_size)
    with open(names_path, encoding="utf-8") as f:
        saved_names = json.load(f)
    if class_names != saved_names:
        raise SystemExit("Dataset class order differs from saved model class order.")

    model = tf.keras.models.load_model(model_path)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    y_pred = model.predict(test_ds, verbose=1).argmax(axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names,
                                digits=4, zero_division=0))

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * .55),
                                    max(8, len(class_names) * .45)))
    display.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    fig.tight_layout()
    output = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(output, dpi=180)
    print(f"Confusion matrix saved to {output}")


if __name__ == "__main__":
    main()
