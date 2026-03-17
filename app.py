import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("🌿 Grassland Health Monitoring AI")

# Download model from Google Drive
MODEL_PATH = "grassland_health_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "PASTE_YOUR_DRIVE_LINK_HERE"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["bare", "degraded", "healthy"]

info = {
    "healthy": "🌿 Healthy grassland: Good vegetation and biodiversity.",
    "degraded": "⚠️ Degraded grassland: Overgrazing or soil damage.",
    "bare": "🟤 Bare land: No vegetation, risk of desertification."
}

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    image = image.resize((224,224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
    st.info(info[predicted_class])
