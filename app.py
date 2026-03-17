import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("🌿 Grassland Health Monitoring AI")

MODEL_PATH = "grassland_health_model.h5"

# Download model only once
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1Y9c33KwHsFs5EMGOlxL0maLbfOEM3Qmi"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
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

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
    st.info(info[predicted_class])
