import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Grassland Health AI",
    page_icon="🌿",
    layout="wide"
)

# --- 2. CONSTANTS & DRIVE INFO ---
MODEL_PATH = "grassland_model.h5"
GDRIVE_FILE_ID = "1iC_p6UJNGMoLixlHVY7L0KyQzNRWemll"

# --- 3. EXPANDED KNOWLEDGE BASE ---
info_details = {
    "healthy": {
        "status": "🟢 Optimal Condition",
        "desc": "High biodiversity with dense perennial grass cover. Soil is well-protected from erosion.",
        "causes": "Proper grazing rotation, adequate rainfall, and sustainable land management.",
        "actions": "Maintain current stocking rates. Conduct seasonal biomass monitoring to prevent future overgrazing.",
        "impact": "High Carbon Sequestration & Water Filtration."
    },
    "degraded": {
        "status": "🟡 Warning: Sub-Optimal",
        "desc": "Vegetation is thinning. Invasive species or weeds may be starting to take over.",
        "causes": "Overgrazing, early signs of drought, or soil compaction from heavy machinery.",
        "actions": "Reduce livestock density immediately. Implement 'Rest-Rotation' grazing to allow plants to recover.",
        "impact": "Increased Runoff & Loss of Nutrients."
    },
    "bare": {
        "status": "🔴 Critical: Emergency",
        "desc": "Significant soil exposure. High risk of nutrient runoff and permanent desertification.",
        "causes": "Long-term overgrazing, severe drought, or fire damage.",
        "actions": "Immediate exclusion of livestock. Consider reseeding and mechanical soil pitting to catch rainwater.",
        "impact": "High Risk of Desertification & Dust Storms."
    }
}

# --- 4. MODEL LOADING (Version Stable) ---
@st.cache_resource
def load_grassland_ai():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        with st.spinner("Downloading AI model..."):
            gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        # Using tf.keras.models.load_model with the pinned versions in requirements.txt
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 5. MAIN INTERFACE ---
st.title("🌿 Grassland Health Monitoring AI")
st.markdown("### Advanced Ecological Assessment Tool")

model = load_grassland_ai()
class_names = ["bare", "degraded", "healthy"]

# Sidebar for extra info
with st.sidebar:
    st.header("About the Project")
    st.write("This AI uses Computer Vision to monitor the health of grassland ecosystems.")
    st.divider()
    st.write("📊 **Model Info:** CNN (Inception-based)")
    st.write("📍 **Target:** Rangeland Management")

# Layout: Upload on left, Results on right
col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_file = st.file_uploader("Upload a landscape or ground photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

# --- 6. PREDICTION & DETAILED INFO ---
if uploaded_file is not None and model is not None:
    with col_right:
        with st.spinner("Analyzing vegetation patterns..."):
            # Preprocessing
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            result_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            label = class_names[result_index]
            
            data = info_details[label]

        # UI Results
        st.subheader(f"Assessment: {data['status']}")
        st.progress(float(confidence))
        st.write(f"**AI Confidence:** {confidence:.2%}")
        
        # Tabs for more information
        tab1, tab2, tab3 = st.tabs(["📝 Description", "🔍 Causes", "🛠️ Actions"])
        
        with tab1:
            st.markdown(f"**Ecological Status:** {data['desc']}")
            st.write(f"**Environmental Impact:** {data['impact']}")
        
        with tab2:
            st.write(f"**Primary Drivers:** {data['causes']}")
            st.info("Note: Weather patterns and soil type can also influence these results.")
            
        with tab3:
            st.success(f"**Recommended Strategy:** \n\n {data['actions']}")
            
        # Download Result button
        report_text = f"Grassland Health Report\nResult: {label}\nConfidence: {confidence:.2%}\nAdvice: {data['actions']}"
        st.download_button("Download Report", report_text, file_name="grassland_report.txt")

elif model is None:
    st.error("AI Initialization failed. Check the logs.")

st.divider()
st.caption("Disclaimer: This AI is a decision-support tool. Always consult with local ecological experts for final land management decisions.")
