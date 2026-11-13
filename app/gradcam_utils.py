"""
Grad-CAM Explainability Utilities for Malaria Cell Images.
"""

import os
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from app.model_utils import find_last_conv_layer, TARGET_SIZE


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    layer_name: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    Generate normalized Grad-CAM heatmap array for a single preprocessed image.

    Args:
        model: Trained tf.keras.Model
        img_batch: Preprocessed image tensor of shape (1, 128, 128, 3) with values in [0, 255]
        layer_name: Optional target Conv2D layer name. If None, dynamically resolved.

    Returns:
        (heatmap_array, resolved_layer_name, error_message)
    """
    try:
        # Dynamically discover target layer if not explicitly supplied
        target_layer = layer_name or find_last_conv_layer(model)

        # Build submodel routing the target layer activations and final prediction
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(target_layer).output,
                model.output,
            ],
        )

        # Compute gradients with respect to convolutional feature maps
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch, training=False)
            # Binary classification: output is single sigmoid score for Parasitized
            score = predictions[:, 0]

        gradients = tape.gradient(score, conv_outputs)
        if gradients is None:
            return None, target_layer, "Failed to compute gradients for target layer."

        # Global average pooling of gradients: spatial weights
        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))

        # Weighted combination of feature maps
        conv_outputs_single = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs_single * weights, axis=-1)

        # Apply ReLU to retain only features with a positive influence on score
        heatmap = tf.maximum(heatmap, 0)

        # Normalize heatmap to [0, 1]
        maximum = tf.reduce_max(heatmap)
        heatmap = tf.where(
            maximum > 0,
            heatmap / (maximum + tf.keras.backend.epsilon()),
            heatmap,
        )

        # Resize heatmap back to match original image dimensions
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis],
            TARGET_SIZE,
        )

        heatmap_np = heatmap_resized.numpy().squeeze()
        return heatmap_np, target_layer, None

    except Exception as e:
        return None, None, f"Grad-CAM computation encountered an error: {str(e)}"


def overlay_gradcam_on_image(
    original_pil: Image.Image,
    heatmap: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.45,
) -> Tuple[Image.Image, Image.Image]:
    """
    Create a colored heatmap visualization and blend it with the original cell image.

    Args:
        original_pil: Original PIL Image (RGB, resized to 128x128)
        heatmap: 2D numpy array [128, 128] with values in [0.0, 1.0]
        colormap: Matplotlib colormap name (e.g. 'jet', 'viridis', 'plasma', 'inferno')
        alpha: Overlay opacity between 0.0 (only original) and 1.0 (only heatmap)

    Returns:
        (colored_heatmap_pil, blended_overlay_pil)
    """
    # Ensure correct sizing
    if original_pil.size != TARGET_SIZE:
        original_pil = original_pil.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

    img_np = np.array(original_pil, dtype=np.float32)

    # Fetch colormap from matplotlib
    try:
        cmap = matplotlib.colormaps.get_cmap(colormap)
    except Exception:
        cmap = matplotlib.colormaps.get_cmap("jet")

    # Map heatmap [0, 1] to RGB [0, 255]
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Strip alpha if present
    colored_heatmap_uint8 = np.uint8(colored_heatmap * 255.0)

    # Alpha blending: (1 - alpha) * image + alpha * heatmap
    blended = np.clip(
        (1.0 - alpha) * img_np + alpha * (colored_heatmap * 255.0),
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(colored_heatmap_uint8), Image.fromarray(blended)


def explain_gradcam_findings(
    predicted_class: str,
    probability_parasitized: float,
    layer_name: str,
) -> Dict[str, str]:
    """
    Generate an interpretable explanation tailored to the prediction and Grad-CAM output.
    Reflects scientific accuracy regarding model activations versus biological claims.
    """
    if predicted_class == "Parasitized":
        summary = (
            f"The Custom CNN classified this cell as Parasitized "
            f"(model probability: {probability_parasitized * 100:.2f}%). The warm regions "
            f"(red/yellow) indicate spatial locations in the final convolutional layer (`{layer_name}`) "
            f"where activations produced the strongest positive gradient toward the parasite class output."
        )
        interpretability_note = (
            "Interpretability Caution: Grad-CAM visualizes internal mathematical activation patterns "
            "of the convolutional filters and gradient flow; it does not prove that the network has learned "
            "medically meaningful parasite features. Visual alignment with cellular spots cannot be claimed as "
            "verified identification of ring forms, chromatin dots, or trophozoites without independent biological validation."
        )
    else:
        summary = (
            f"The Custom CNN classified this cell as Uninfected "
            f"(model probability: {(1.0 - probability_parasitized) * 100:.2f}%). The Grad-CAM heatmap "
            f"displays gradient sensitivity relative to the parasite output score. Activations observed around "
            f"cell boundaries or image corners may reflect sensitivity to image edges, background illumination, "
            f"staining gradients, or dataset-specific cropping artifacts rather than intracellular features."
        )
        interpretability_note = (
            "Interpretability Caution: Low, diffuse, or peripheral activation does not confirm the absence of parasites. "
            "Grad-CAM reflects model feature sensitivity under its trained weights, not an exhaustive clinical or "
            "biological proof of non-infection."
        )

    technical_context = (
        f"Resolution & Upsampling: Gradients were computed with respect to the terminal "
        f"convolutional layer (`{layer_name}`), which has a native feature map grid of 16 × 16 "
        f"(with 128 channels). This coarse 16 × 16 activation map was upsampled via bilinear interpolation "
        f"to match the 128 × 128 image for display; it does not represent native 128 × 128 pixel-level localization."
    )

    return {
        "summary": summary,
        "biological_context": interpretability_note,
        "technical_context": technical_context,
    }
