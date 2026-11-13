from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

print("Loading MobileNetV2...")
model = tf.keras.models.load_model(
    ROOT / "models/mobilenetv2_best.keras"
)

frame = pd.read_csv(ROOT / "data/processed/splits.csv")
test = frame[frame["split"] == "test"].copy()

paths = [str(ROOT / p) for p in test["image_path"]]
y_true = test["label"].to_numpy(dtype=int)

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, (128, 128))
    return tf.cast(image, tf.float32)

dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

print(f"Evaluating {len(y_true)} test images...")
probabilities = model.predict(dataset, verbose=1).ravel()
y_pred = (probabilities >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred)),
    "recall": float(recall_score(y_true, y_pred)),
    "f1_score": float(f1_score(y_true, y_pred)),
    "roc_auc": float(roc_auc_score(y_true, probabilities)),
    "test_images": len(y_true),
}

print("\nMOBILENETV2 — TEST RESULTS")
print("=" * 45)

for key, value in metrics.items():
    print(f"{key:15}: {value:.4f}" if isinstance(value, float)
          else f"{key:15}: {value}")

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=["Uninfected", "Parasitized"],
    digits=4,
))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

metrics["confusion_matrix"] = cm.tolist()

(RESULTS / "mobilenetv2_test_metrics.json").write_text(
    json.dumps(metrics, indent=2)
)

predictions = test[
    ["image_path", "patient_id", "label"]
].copy()
predictions["probability_parasitized"] = probabilities
predictions["predicted_label"] = y_pred
predictions.to_csv(
    RESULTS / "mobilenetv2_test_predictions.csv",
    index=False,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Uninfected", "Parasitized"],
)
display.plot(cmap="Blues", values_format="d")
plt.title("MobileNetV2 — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS / "mobilenetv2_confusion_matrix.png",
    dpi=300,
)
plt.close()

print("\nSaved MobileNetV2 evaluation artifacts.")
