import argparse
import json
import numpy as np
import tensorflow as tf
from .config import IMAGE_SIZE, MODEL_DIR


def predict_image(image_path: str):
    model_path, names_path = MODEL_DIR / "crop_disease_model.keras", MODEL_DIR / "class_names.json"
    if not model_path.exists() or not names_path.exists():
        raise FileNotFoundError("Train the model first; model files are missing.")
    with open(names_path, encoding="utf-8") as f:
        class_names = json.load(f)
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    probabilities = tf.keras.models.load_model(model_path).predict(
        np.expand_dims(array, axis=0), verbose=0)[0]
    best = int(np.argmax(probabilities))
    return class_names[best], float(probabilities[best])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a leaf image")
    args = parser.parse_args()
    label, score = predict_image(args.image)
    print(f"Prediction: {label}")
    print(f"Model score: {score:.2%} (not a calibrated probability)")


if __name__ == "__main__":
    main()
