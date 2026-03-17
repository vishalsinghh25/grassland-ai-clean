import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("🌿 Grassland Health Monitoring AI")

MODEL_PATH = "grassland_health_model.h5"

# Load model (download only once)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "1Y9c33KwHsFs5EMGOlxL0maLbfOEM3Qmi"
        gdown.download(
            f"https://drive.google.com/uc?export=download&id={file_id}",
            MODEL_PATH,
            quiet=False
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Class labels (same as training)
class_names = ["bare", "degraded", "healthy"]

# Info messages
info = {
    "healthy": "🌿 Healthy grassland: Good vegetation and biodiversity.",
    "degraded": "⚠️ Degraded grassland: Overgrazing or soil damage.",
    "bare": "🟤 Bare land: No vegetation, risk of desertification."
}

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Output
    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
    st.info(info[predicted_class])
