import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# -------------------------------
# Model download link (Google Drive)
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

# -------------------------------
# Download and load model
# -------------------------------
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model... Please wait (only once)..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return load_model(MODEL_PATH)

model = download_and_load_model()

# -------------------------------
# Class Labels
# -------------------------------
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Satellite Image Classifier", page_icon="🛰️", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌍 Satellite Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a satellite image to classify it as <strong>Cloudy</strong>, <strong>Desert</strong>, <strong>Green Area</strong>, or <strong>Water</strong>.</p>", unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("📤 Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((256, 256))
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess Image
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display Results
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🔍 Prediction Result")
    st.success(f"🌟 **Predicted Class:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")
    st.markdown("<hr>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Please upload a satellite image file (JPG, JPEG, PNG).")
