from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/custom_cnn_best.keras"
SPLIT_PATH = ROOT / "data/processed/splits.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

print("Loading best CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)

frame = pd.read_csv(SPLIT_PATH)
test_frame = frame[frame["split"] == "test"].copy()

paths = [
    str(ROOT / path)
    for path in test_frame["image_path"]
]
y_true = test_frame["label"].to_numpy(dtype=int)

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, (128, 128))
    return tf.cast(image, tf.float32)

test_ds = tf.data.Dataset.from_tensor_slices(paths)
test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE,
).batch(32).prefetch(tf.data.AUTOTUNE)

print(f"Evaluating {len(y_true)} test images...")
probabilities = model.predict(test_ds, verbose=1).ravel()
y_pred = (probabilities >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_true, probabilities)),
    "test_images": int(len(y_true)),
}

print("\nCUSTOM CNN — TEST RESULTS")
print("=" * 45)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name:15}: {value:.4f} ({value * 100:.2f}%)")
    else:
        print(f"{name:15}: {value}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Uninfected", "Parasitized"],
    digits=4,
    zero_division=0,
))

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

report = classification_report(
    y_true,
    y_pred,
    target_names=["Uninfected", "Parasitized"],
    output_dict=True,
    zero_division=0,
)
metrics["classification_report"] = report
metrics["confusion_matrix"] = cm.tolist()

(RESULTS / "custom_cnn_test_metrics.json").write_text(
    json.dumps(metrics, indent=2)
)

predictions = test_frame[[
    "image_path", "patient_id", "label"
]].copy()
predictions["probability_parasitized"] = probabilities
predictions["predicted_label"] = y_pred

predictions.to_csv(
    RESULTS / "custom_cnn_test_predictions.csv",
    index=False,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Uninfected", "Parasitized"],
)
display.plot(cmap="Blues", values_format="d")
plt.title("Custom CNN — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS / "custom_cnn_confusion_matrix.png", dpi=300)
plt.close()

print("\nSaved test metrics, predictions and confusion matrix.")
