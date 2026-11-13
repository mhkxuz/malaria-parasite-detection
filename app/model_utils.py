"""
Model management, preprocessing, and prediction utilities for Malaria Detection.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import streamlit as st

# Silence unnecessary TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models/custom_cnn_best.keras"
TARGET_SIZE = (128, 128)


@st.cache_resource(show_spinner="Loading trained Custom CNN model...")
def load_trained_model(model_path: Optional[str] = None) -> tf.keras.Model:
    """
    Load and cache the trained Keras model.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights file not found at: {path}. "
            "Please ensure 'models/custom_cnn_best.keras' exists."
        )
    model = tf.keras.models.load_model(str(path))
    return model


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Dynamically identify the name of the final Conv2D layer in the model.
    Avoids hardcoding layer names.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # In case of nested models or containers
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("No Conv2D layer found in the model architecture.")


def get_all_conv_layers(model: tf.keras.Model) -> list[str]:
    """
    Return names of all Conv2D layers in the model in feedforward order.
    """
    names = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            names.append(layer.name)
        elif hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Conv2D):
                    names.append(sub.name)
    return names


def preprocess_image(
    image_input: Any,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Validate, convert to RGB, and resize the input image.

    Note on normalization:
    The trained Custom CNN model contains its own internal Rescaling layer
    (tf.keras.layers.Rescaling(1.0 / 255)).
    Therefore, the input array passed to the model must be in raw [0, 255] float32 values.
    Do NOT normalize by dividing by 255 here.

    Returns:
        (preprocessed_array_with_batch_dim, pil_image_resized)
    """
    try:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input))
        else:
            # File-like object or UploadedFile
            pil_img = Image.open(image_input)
    except Exception as e:
        raise ValueError(f"Unable to read or decode image file: {e}")

    # Verify image integrity
    try:
        pil_img.verify()
        # Reopen after verify() because verify() can close or corrupt the file pointer
        if hasattr(image_input, "seek"):
            image_input.seek(0)
            pil_img = Image.open(image_input)
        elif isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input))
    except Exception:
        # Fallback to direct conversion if verify isn't re-readable
        pass

    # Ensure 3-channel RGB (removes alpha or grayscale variance)
    pil_img = pil_img.convert("RGB")

    # Resize to model input shape
    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float32 numpy array with range [0, 255]
    img_array = np.array(pil_resized, dtype=np.float32)

    # Add batch dimension: (1, 128, 128, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch, pil_resized


def predict_cell(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Execute forward inference on a preprocessed cell image batch.

    Output format:
    - Raw sigmoid output p represents Probability(Parasitized)
    - 1 - p represents Probability(Uninfected)
    - Threshold 0.5 separates Class 0 (Uninfected) and Class 1 (Parasitized)
    """
    predictions = model.predict(img_batch, verbose=0)
    prob_parasitized = float(predictions[0, 0])
    # Clamp to [0.0, 1.0] for safety
    prob_parasitized = max(0.0, min(1.0, prob_parasitized))
    prob_uninfected = 1.0 - prob_parasitized

    is_parasitized = prob_parasitized >= threshold
    predicted_class = "Parasitized" if is_parasitized else "Uninfected"
    predicted_label = 1 if is_parasitized else 0
    confidence = prob_parasitized if is_parasitized else prob_uninfected

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probability_parasitized": prob_parasitized,
        "probability_uninfected": prob_uninfected,
        "confidence": confidence,
        "confidence_percent": confidence * 100.0,
        "threshold": threshold,
    }
