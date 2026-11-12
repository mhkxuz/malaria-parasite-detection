from pathlib import Path
import tensorflow as tf
from .config import IMAGE_SIZE, SEED


def load_datasets(data_dir: Path, batch_size: int = 32):
    train_dir, val_dir, test_dir = (data_dir / name for name in ("train", "val", "test"))
    for directory in (train_dir, val_dir, test_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {directory}. Run src.split_dataset first.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=batch_size,
        label_mode="int", shuffle=True, seed=SEED)
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, class_names=class_names, image_size=IMAGE_SIZE,
        batch_size=batch_size, label_mode="int", shuffle=False)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, class_names=class_names, image_size=IMAGE_SIZE,
        batch_size=batch_size, label_mode="int", shuffle=False)

    autotune = tf.data.AUTOTUNE
    return (train_ds.prefetch(autotune), val_ds.prefetch(autotune),
            test_ds.prefetch(autotune), class_names)
