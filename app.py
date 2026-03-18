import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Grassland Health AI", page_icon="🌿", layout="wide")

# --- 2. CONSTANTS ---
MODEL_PATH = "grassland_model.h5"
GDRIVE_FILE_ID = "1iC_p6UJNGMoLixlHVY7L0KyQzNRWemll"

# --- 3. THE KNOWLEDGE BASE ---
info_details = {
    "healthy": {
        "status": "🟢 Healthy",
        "desc": "Lush vegetation with high biodiversity. Excellent soil protection.",
        "actions": "Maintain current grazing levels. Monitor for invasive species."
    },
    "degraded": {
        "status": "🟡 Degraded",
        "desc": "Thinning grass cover. Soil starting to show through patches.",
        "actions": "Reduce livestock density. Allow the land to rest and re-seed."
    },
    "bare": {
        "status": "🔴 Bare Land",
        "desc": "Little to no vegetation. High risk of erosion and desertification.",
        "actions": "Immediate exclusion of livestock. Implement soil restoration techniques."
    }
}

# --- 4. MANUAL ARCHITECTURE BUILD ---
# This ignores the corrupted metadata in the .h5 file and creates a fresh model
def build_model_skeleton():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

@st.cache_resource
def load_grassland_ai():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        with st.spinner("Downloading AI Model..."):
            gdown.download(url, MODEL_PATH, quiet=False)
    
    # Step 1: Build the empty skeleton
    model = build_model_skeleton()
    
    try:
        # Step 2: Only load the weights into the skeleton
        # This bypasses all 'InputLayer' and 'batch_shape' errors
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        # Step 3: Last-ditch effort (Standard load)
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except:
            st.error(f"Critical Loading Error: {e}")
            return None

# --- 5. APP UI ---
st.title("🌿 Grassland Health Monitoring AI")
model = load_grassland_ai()
class_names = ["bare", "degraded", "healthy"]

col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_file = st.file_uploader("Upload Grassland Photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

if uploaded_file is not None and model is not None:
    with col_right:
        with st.spinner("Analyzing..."):
            # Process image
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            idx = np.argmax(predictions[0])
            conf = np.max(predictions[0])
            label = class_names[idx]
            
            # Results
            st.subheader(f"Assessment: {info_details[label]['status']}")
            st.write(f"**Confidence:** {conf:.2%}")
            st.progress(float(conf))
            
            st.markdown(f"**Description:** {info_details[label]['desc']}")
            st.success(f"**Suggested Action:** {info_details[label]['actions']}")

elif model is None:
    st.error("Model failed to load. Please check your GitHub requirements and Reboot the app.")
