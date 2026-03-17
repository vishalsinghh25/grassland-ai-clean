import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("🌿 Grassland Health Monitoring AI")

WEIGHTS_PATH = "model_weights.h5"

# Build same model architecture
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224,224,3)),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
         file_id = "1uTKYwpe6Ruwx7pw5Nqbv6BgLvnkrrNxL"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)

    model = build_model()
    model.load_weights(WEIGHTS_PATH)

    return model

model = load_model()

class_names = ["bare", "degraded", "healthy"]

info = {
    "healthy": "🌿 Healthy grassland: Good vegetation and biodiversity.",
    "degraded": "⚠️ Degraded grassland: Overgrazing or soil damage.",
    "bare": "🟤 Bare land: No vegetation, risk of desertification."
}

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    image = image.resize((224,224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
    st.info(info[predicted_class])
