from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MODEL_PATH = ROOT / "models/custom_cnn_best.keras"
PREDICTIONS_PATH = RESULTS / "custom_cnn_test_predictions.csv"

IMAGE_SIZE = (128, 128)


def load_image(image_path):
    image = tf.io.read_file(str(ROOT / image_path))
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32)


def generate_gradcam(model, image, layer_name):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            tf.expand_dims(image, axis=0),
            training=False,
        )
        score = predictions[:, 0]

    gradients = tape.gradient(score, conv_outputs)

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum > 0,
        heatmap / (maximum + tf.keras.backend.epsilon()),
        heatmap,
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE,
    )

    return heatmap.numpy().squeeze()


def main():
    print("Loading custom CNN...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Automatically identify the last convolutional layer.
    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    if not conv_layers:
        raise RuntimeError("No Conv2D layer found in the model.")

    last_conv_layer = conv_layers[-1].name
    print("Grad-CAM layer:", last_conv_layer)

    frame = pd.read_csv(PREDICTIONS_PATH)

    # Select one example from each available confusion-matrix category.
    categories = {
        "True Positive": (1, 1),
        "True Negative": (0, 0),
        "False Positive": (0, 1),
        "False Negative": (1, 0),
    }

    selected = []

    for name, (actual, predicted) in categories.items():
        matches = frame[
            (frame["label"] == actual)
            & (frame["predicted_label"] == predicted)
        ]

        if matches.empty:
            print("No example available:", name)
            continue

        # Pick the most confident example in that category.
        if name == "True Positive" or name == "False Positive":
            sample = matches.loc[
                matches["probability_parasitized"].idxmax()
            ]
        else:
            sample = matches.loc[
                matches["probability_parasitized"].idxmin()
            ]

        selected.append((name, sample))

    if not selected:
        raise RuntimeError("No test examples available.")

    fig, axes = plt.subplots(
        len(selected), 2,
        figsize=(9, 4 * len(selected)),
        squeeze=False,
    )

    for row, (category, sample) in enumerate(selected):
        image = load_image(sample["image_path"])
        heatmap = generate_gradcam(
            model,
            image,
            last_conv_layer,
        )

        image_np = image.numpy().astype(np.uint8)

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(
            f"{category}\n"
            f"Actual: {int(sample['label'])}, "
            f"Predicted: {int(sample['predicted_label'])}\n"
            f"P(parasitized): "
            f"{sample['probability_parasitized']:.3f}"
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(image_np)
        axes[row, 1].imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )
        axes[row, 1].set_title(f"{category} - Grad-CAM")
        axes[row, 1].axis("off")

        print(f"Generated Grad-CAM: {category}")

    fig.suptitle(
        "Grad-CAM Explainability - Custom CNN",
        fontsize=14,
    )
    fig.tight_layout()

    output = RESULTS / "custom_cnn_gradcam.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
