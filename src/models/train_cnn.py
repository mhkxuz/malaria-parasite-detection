from pathlib import Path
import json

import tensorflow as tf

from src.data.dataset import load_datasets
from src.models.cnn import build_custom_cnn

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS = 4

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("Loading datasets...")

train_ds, val_ds, _ = load_datasets()

model = build_custom_cnn()
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "custom_cnn_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

print("\nStarting custom CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(MODEL_DIR / "custom_cnn_final.keras")

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

(RESULTS_DIR / "custom_cnn_history.json").write_text(
    json.dumps(history_data, indent=2)
)

print("\nTraining complete.")
print("Best model:", MODEL_DIR / "custom_cnn_best.keras")
print("Final model:", MODEL_DIR / "custom_cnn_final.keras")
print("History:", RESULTS_DIR / "custom_cnn_history.json")
