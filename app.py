#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import requests
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import joblib
from io import BytesIO
import base64

# ============================================================
# Helper function to encode image to base64
# ============================================================
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        data = image_file.read()
    return base64.b64encode(data).decode()

# Get base64 string for your background image
background_base64 = get_base64_image("/Users/anandhu/Downloads/image.jpg.avif")

# ============================================================
# 🚀 Page Configuration & Custom Styling
# ============================================================
st.set_page_config(page_title="Flood Risk Assessment Tool", layout="wide")

# Inject custom CSS including the background image from the local file
st.markdown(f"""
    <style>
    .big-title {{ font-size:40px !important; text-align: center; font-weight: bold; color: #FFFFFF; }}
    .desc {{ text-align: center; font-size: 18px; color: #F8F9FA; }}
    .home-bg {{
        background: linear-gradient(to bottom, rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.8)),
        url("data:image/jpg;base64,{background_base64}");
        background-size: cover;
        height: 400px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
    }}
    .sidebar-title {{ font-size: 22px; font-weight: bold; color: #FFFFFF; }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 🔄 Load Models
# ============================================================

# Flood risk prediction models (Random Forest & XGBoost)
rf_model = joblib.load("/Users/anandhu/Downloads/Final Project/combined_dataset/structured/random_forest_model.pkl")
xgb_model = joblib.load("/Users/anandhu/Downloads/Final Project/combined_dataset/structured/xgboost_model.pkl")

# UNet model for image segmentation (cached for efficiency)
@st.cache_resource
def load_unet_model():
    return load_model("unet_finetuned_best.h5", compile=False)

unet_model = load_unet_model()

# ============================================================
# 🔍 Helper Functions
# ============================================================

# Flood risk classification function
def classify_flood_risk(prediction):
    if prediction == 0:
        return "Low Risk"
    elif prediction == 1:
        return "Medium Risk"
    else:
        return "High Risk"

# Default feature cases for flood prediction
default_cases = {
    "Low Risk": {"MonsoonIntensity": 2, "ClimateChange": 2, "Landslides": 2, "DamsQuality": 8, "CoastalVulnerability": 2, "IneffectiveDisasterPreparedness": 2, "InadequatePlanning": 2, "Deforestation": 2, "Urbanization": 2, "Encroachments": 1, "WetlandLoss": 2, "AgriculturalPractices": 2, "DeterioratingInfrastructure": 2, "PoliticalFactors": 6, "Watersheds": 2},
    "Medium Risk": {"MonsoonIntensity": 5, "ClimateChange": 5, "Landslides": 5, "DamsQuality": 5, "CoastalVulnerability": 5, "IneffectiveDisasterPreparedness": 5, "InadequatePlanning": 5, "Deforestation": 5, "Urbanization": 5, "Encroachments": 5, "WetlandLoss": 5, "AgriculturalPractices": 5, "DeterioratingInfrastructure": 5, "PoliticalFactors": 5, "Watersheds": 5},
    "High Risk": {"MonsoonIntensity": 9, "ClimateChange": 9, "Landslides": 9, "DamsQuality": 2, "CoastalVulnerability": 8, "IneffectiveDisasterPreparedness": 9, "InadequatePlanning": 9, "Deforestation": 9, "Urbanization": 9, "Encroachments": 9, "WetlandLoss": 9, "AgriculturalPractices": 9, "DeterioratingInfrastructure": 9, "PoliticalFactors": 3, "Watersheds": 8}
}

# Image segmentation functions
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_segmentation(image):
    processed_image = preprocess_image(image)
    prediction = unet_model.predict(processed_image)[0][:, :, 0]
    return prediction

# ============================================================
# 📌 Sidebar Navigation
# ============================================================
st.sidebar.markdown('<p class="sidebar-title">🌊 Flood Risk Assessment Tool</p>', unsafe_allow_html=True)
menu = st.sidebar.radio("Select an Option", ["🏠 Home", "🖼️ Image Segmentation", "📊 Flood Prediction", "🌦️ Weather & News"])

# ============================================================
# 🏠 Home Page
# ============================================================
if menu == "🏠 Home":
    st.markdown("""
    <div class="home-bg">
        <div>
            <h1 class="big-title">Flood Risk Assessment Tool</h1>
            <p class="desc">Real-time flood risk predictions and image segmentation for disaster management.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 🖼️ Image Segmentation Page
# ============================================================
elif menu == "🖼️ Image Segmentation":
    st.write("## 🖼️ Upload Image for Segmentation")
    uploaded_image = st.file_uploader("Upload a Flood-Affected Area Image (jpg, png)", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        prediction = predict_segmentation(image)

        st.write("### 🎛️ Adjust Visualization")
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
        opacity = st.slider("Mask Opacity", 0.0, 1.0, 0.5, 0.05)

        # Create a binary mask based on threshold
        mask = (prediction > threshold).astype(np.uint8)

        st.write("### 🖼️ Original Image")
        st.image(image, use_column_width=True)

        st.write("### 🟢 Segmentation Mask")
        st.image(mask * 255, use_column_width=True)

        st.write("### 🖌️ Overlay Image + Mask")
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(mask, cmap="jet", alpha=opacity)
        ax.axis("off")
        st.pyplot(fig)

# ============================================================
# 📊 Flood Prediction Page
# ============================================================
elif menu == "📊 Flood Prediction":
    st.markdown("## 🌊 Flood Risk Assessment")
    st.write("Enter the environmental conditions below to predict the flood risk or select a default case.")

    selected_case = st.selectbox("Choose a default risk case", ["None", "Low Risk", "Medium Risk", "High Risk"])
    
    # Set feature values based on the default case or use a neutral default (Medium Risk = 5 for all)
    if selected_case != "None":
        feature_values = default_cases[selected_case]
    else:
        feature_values = {key: 5 for key in default_cases["Medium Risk"].keys()}

    st.write("### Default Feature Values")
    st.dataframe(pd.DataFrame(feature_values, index=[0]), use_container_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌧️ Weather & Climate Conditions")
        with st.expander("Expand to enter details"):
            monsoon_intensity = st.slider('Monsoon Intensity 🌧️', 0, 10, feature_values["MonsoonIntensity"], format="%d")
            climate_change = st.slider('Climate Change Impact 🌍', 0, 10, feature_values["ClimateChange"], format="%d")
            coastal_vulnerability = st.slider('Coastal Vulnerability 🏝️', 0, 10, feature_values["CoastalVulnerability"], format="%d")
            deforestation = st.slider('Deforestation 🌲', 0, 10, feature_values["Deforestation"], format="%d")
            landslides = st.slider('Landslide Risk ⛰️', 0, 10, feature_values["Landslides"], format="%d")
            watersheds = st.slider('Watershed Condition 🌊', 0, 10, feature_values["Watersheds"], format="%d")

    with col2:
        st.markdown("### 🏗️ Infrastructure & Preparedness")
        with st.expander("Expand to enter details"):
            dams_quality = st.slider('Dams & Reservoirs 🏞️', 0, 10, feature_values["DamsQuality"], format="%d")
            ineffective_disaster_prep = st.slider('Disaster Preparedness 🚨', 0, 10, feature_values["IneffectiveDisasterPreparedness"], format="%d")
            inadequate_planning = st.slider('Inadequate Planning 🏗️', 0, 10, feature_values["InadequatePlanning"], format="%d")
            urbanization = st.slider('Urbanization 🏢', 0, 10, feature_values["Urbanization"], format="%d")
            encroachments = st.slider('Encroachments 🚧', 0, 10, feature_values["Encroachments"], format="%d")
            wetland_loss = st.slider('Wetland Loss 🌾', 0, 10, feature_values["WetlandLoss"], format="%d")
            agricultural_practices = st.slider('Agricultural Practices 🚜', 0, 10, feature_values["AgriculturalPractices"], format="%d")
            deteriorating_infra = st.slider('Deteriorating Infrastructure 🏚️', 0, 10, feature_values["DeterioratingInfrastructure"], format="%d")
            political_factors = st.slider('Political Factors 🗳️', 0, 10, feature_values["PoliticalFactors"], format="%d")
    
    # Build the input dictionary using the slider values
    input_features = {
        "MonsoonIntensity": monsoon_intensity,
        "ClimateChange": climate_change,
        "Landslides": landslides,
        "DamsQuality": dams_quality,
        "CoastalVulnerability": coastal_vulnerability,
        "IneffectiveDisasterPreparedness": ineffective_disaster_prep,
        "InadequatePlanning": inadequate_planning,
        "Deforestation": deforestation,
        "Urbanization": urbanization,
        "Encroachments": encroachments,
        "WetlandLoss": wetland_loss,
        "AgriculturalPractices": agricultural_practices,
        "DeterioratingInfrastructure": deteriorating_infra,
        "PoliticalFactors": political_factors,
        "Watersheds": watersheds,
    }
    input_data = pd.DataFrame([input_features])

    st.markdown("---")
    if st.button("🚀 Predict Flood Risk"):
        rf_prediction = rf_model.predict(input_data)[0]
        xgb_prediction = xgb_model.predict(input_data)[0]
        
        st.subheader("🔍 Predictions:")
        st.write(f"🌳 **Random Forest Prediction:** {classify_flood_risk(rf_prediction)}")
        st.write(f"⚡ **XGBoost Prediction:** {classify_flood_risk(xgb_prediction)}")

# ============================================================
# 🌦️ Weather & News Page
# ============================================================
elif menu == "🌦️ Weather & News":
    st.markdown("## 🌦️ Weather & News")
    st.info("Weather and news updates coming soon!")

