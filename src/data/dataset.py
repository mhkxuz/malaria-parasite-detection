from pathlib import Path

import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = PROJECT_ROOT / "data/processed/splits.csv"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.float32)


def make_dataset(frame, training=False, batch_size=BATCH_SIZE):
    paths = [
        str(PROJECT_ROOT / path)
        for path in frame["image_path"].tolist()
    ]
    labels = frame["label"].to_numpy(dtype="float32")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=42,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def load_datasets(batch_size=BATCH_SIZE):
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}\n"
            "Run src/data/prepare_dataset.py first."
        )

    frame = pd.read_csv(SPLIT_FILE)

    train = make_dataset(
        frame[frame["split"] == "train"],
        training=True,
        batch_size=batch_size,
    )

    validation = make_dataset(
        frame[frame["split"] == "validation"],
        batch_size=batch_size,
    )

    test = make_dataset(
        frame[frame["split"] == "test"],
        batch_size=batch_size,
    )

    return train, validation, test
