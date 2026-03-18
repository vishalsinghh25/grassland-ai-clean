import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# 1. Page Configuration
st.set_page_config(page_title="Grassland Health AI", page_icon="🌿")

# 2. Constants
WEIGHTS_PATH = "grassland_model.h5"
GDRIVE_FILE_ID = "1iC_p6UJNGMoLixlHVY7L0KyQzNRWemll"

# 3. Manually Define Architecture 
# This must match EXACTLY how you trained the model
def build_model_architecture():
    model = tf.keras.Sequential([
        # Using a standard Input layer that works across versions
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
        
        # 3 classes: bare, degraded, healthy
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

# 4. Load Function
@st.cache_resource
def load_grassland_ai():
    if not os.path.exists(WEIGHTS_PATH):
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        with st.spinner("Downloading model weights..."):
            gdown.download(url, WEIGHTS_PATH, quiet=False)
    
    # Create the skeleton
    model = build_model_architecture()
    
    try:
        # Load weights only (skips the problematic metadata/config)
        model.load_weights(WEIGHTS_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        # Fallback: Try loading as full model if weights-only fails
        try:
            return tf.keras.models.load_model(WEIGHTS_PATH)
        except:
            return None

# 5. UI Logic
st.title("🌿 Grassland Health Monitoring AI")

model = load_grassland_ai()
class_names = ["bare", "degraded", "healthy"]

uploaded_file = st.file_uploader("Upload a grassland photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        # Preprocess
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        result_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        label = class_names[result_index]

    # Output
    st.success(f"**Prediction:** {label.upper()}")
    st.write(f"**Confidence:** {confidence:.2%}")
    
    if label == "healthy":
        st.info("🌿 The grassland appears healthy and well-vegetated.")
    elif label == "degraded":
        st.warning("⚠️ Signs of degradation or overgrazing detected.")
    else:
        st.error("🟤 Bare land detected. Risk of soil erosion/desertification.")

elif model is None:
    st.error("Could not initialize AI. Please check your internet connection or Google Drive link.")
