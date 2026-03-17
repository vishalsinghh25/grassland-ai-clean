import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Grassland AI", layout="centered")

st.title("🌿 Grassland Health Monitoring AI")

MODEL_PATH = "final_model.h5"

# Load model safely
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "1uTKYwpe6Ruwx7pw5Nqbv6BgLvnkrrNxL"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )
    return model

model = load_model()

# Classes
class_names = ["bare", "degraded", "healthy"]

# Info messages
info = {
    "healthy": "🌿 Healthy grassland: Good vegetation and biodiversity.",
    "degraded": "⚠️ Degraded grassland: Overgrazing or soil damage.",
    "bare": "🟤 Bare land: No vegetation, risk of desertification."
}

# Upload
uploaded_file = st.file_uploader("📤 Upload Grassland Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Output
    st.success(f"✅ Prediction: {predicted_class.upper()}")
    st.write(f"📊 Confidence: {confidence:.2f}")
    st.info(info[predicted_class])
