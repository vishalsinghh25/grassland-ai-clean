import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Grassland Health AI",
    page_icon="🌿",
    layout="centered"
)

# 2. Constants
MODEL_PATH = "grassland_model.h5"
# File ID from your Google Drive link
GDRIVE_FILE_ID = "1iC_p6UJNGMoLixlHVY7L0KyQzNRWemll"

# 3. Model Loading Function
@st.cache_resource
def load_grassland_model():
    """Downloads model from Drive if not present and loads it."""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        with st.spinner("Downloading AI model from Google Drive... This may take a minute."):
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None

    try:
        # Load the full .h5 model
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# 4. App Interface
st.title("🌿 Grassland Health Monitoring AI")
st.markdown("""
Upload a photo of grassland to analyze its health status. 
This AI classifies images into **Healthy**, **Degraded**, or **Bare** land.
""")

# Load the model
model = load_grassland_model()

# Define classes and info
class_names = ["bare", "degraded", "healthy"]
info_cards = {
    "healthy": "🟢 **Healthy**: Good vegetation cover and biodiversity.",
    "degraded": "🟡 **Degraded**: Visible signs of overgrazing or soil erosion.",
    "bare": "🔴 **Bare**: Little to no vegetation. High risk of desertification."
}

# 5. Image Upload & Prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocessing
    with st.spinner("AI is analyzing the land..."):
        # Match the input size (224x224)
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0]) # Use softmax if model output is logits
        
        # If your model already has a softmax layer, use:
        # result_index = np.argmax(predictions[0])
        # confidence = np.max(predictions[0])
        
        result_index = np.argmax(score)
        confidence = np.max(score)
        label = class_names[result_index]

    # 6. Display Results
    st.divider()
    st.subheader(f"Result: {label.title()}")
    st.progress(float(confidence))
    st.write(f"**Confidence Level:** {confidence:.2%}")
    st.info(info_cards[label])

elif model is None:
    st.warning("Model could not be loaded. Please check the Google Drive link permissions.")

st.sidebar.info("Developed with Streamlit & TensorFlow")
