import argparse
import json
from pathlib import Path
import tensorflow as tf
from .config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from .data import load_datasets
from .model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    train_ds, val_ds, _, class_names = load_datasets(Path(args.data), args.batch_size)
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(len(class_names))
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "crop_disease_model.keras"),
            monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(RESULTS_DIR / "training_log.csv")),
    ]
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks)
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Training finished; epochs completed: {len(history.history['loss'])}")


if __name__ == "__main__":
    main()
