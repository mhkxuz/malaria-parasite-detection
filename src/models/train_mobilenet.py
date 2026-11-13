from pathlib import Path
import json

import tensorflow as tf

from src.data.dataset import load_datasets
from src.models.mobilenet import build_mobilenet

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS = 4

print("TensorFlow:", tf.__version__)
print("Loading datasets...")

train_ds, val_ds, _ = load_datasets()

print("Building MobileNetV2...")
model = build_mobilenet()
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "mobilenetv2_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1,
    ),
]

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(MODEL_DIR / "mobilenetv2_final.keras")

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

(RESULTS_DIR / "mobilenetv2_history.json").write_text(
    json.dumps(history_data, indent=2)
)

print("\nMobileNetV2 training complete.")
print("Best model:", MODEL_DIR / "mobilenetv2_best.keras")
print("History:", RESULTS_DIR / "mobilenetv2_history.json")
