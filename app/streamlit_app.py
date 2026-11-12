import json
import sys
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import IMAGE_SIZE, MODEL_DIR

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿")
st.title("🌿 Crop Disease Detection")
st.write("Upload a crop-leaf image to see the model's predicted class.")

model_path, names_path = MODEL_DIR / "crop_disease_model.keras", MODEL_DIR / "class_names.json"
if not model_path.exists() or not names_path.exists():
    st.warning("The trained model is not available yet. Follow README.md to train it.")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)


with open(names_path, encoding="utf-8") as f:
    class_names = json.load(f)

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    resized = image.resize(IMAGE_SIZE)
    batch = np.expand_dims(tf.keras.utils.img_to_array(resized), axis=0)
    scores = load_model().predict(batch, verbose=0)[0]
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    st.subheader("Prediction")
    st.write(f"**{class_names[best]}**")
    st.metric("Model score", f"{scores[best]:.1%}")
    st.caption("This score is not a calibrated probability or a diagnosis.")
    st.subheader("Top predictions")
    for index in order[:5]:
        st.write(f"{class_names[int(index)]}: {scores[index]:.1%}")
        st.progress(float(scores[index]))

st.divider()
st.caption("Educational prototype; controlled-dataset performance may not generalize to field conditions.")
