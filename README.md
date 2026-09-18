# AI-Driven Flood Prediction and Mapping System

An AI-driven solution that predicts flood risk using structured hydrological data and unstructured satellite imagery. Combines Random Forest, XGBoost, and U-Net models to assess risk levels and map flood-affected areas from imagery.

## Features

- Home - overview of the tool
- Image Segmentation - upload a flood-affected area image and get a U-Net segmentation mask with adjustable threshold and overlay opacity
- Flood Prediction - enter environmental and infrastructure conditions (or pick a default Low/Medium/High risk case) to get flood risk predictions from Random Forest and XGBoost models
- Weather & News - placeholder section for future weather/news integration

## Tech stack

- Python
- Streamlit (UI)
- pandas / numpy
- scikit-learn (Random Forest), XGBoost
- TensorFlow / Keras (U-Net image segmentation)
- Pillow, matplotlib (image handling and visualization)
- joblib (model loading)

## Project structure

- `app.py` - Streamlit app (Home, Image Segmentation, Flood Prediction, Weather & News)
- `Data_preprocess.ipynb` - data cleaning and preprocessing
- `structured_prediction.ipynb` - Random Forest / XGBoost model training for flood risk
- `image_preprocess_segmentation.ipynb` - image preprocessing for segmentation
- `Analyses.ipynb`, `Visuals.ipynb`, `Flood_test.ipynb` - exploratory analysis, visualisation and testing notebooks
- `unet_custom_best.h5`, `unet_finetuned_best.h5` - trained U-Net model weights
- `combined_dataset/`, `data/` - datasets used for training and testing
- `COM726_Final_Dissertation_2025.pdf`, `Final_Dissertation_2025.docx`, `AI-Driven Flood Prediction and Mapping System.pptx` - dissertation write-up and presentation for this project

## Getting started

```bash
git clone https://github.com/Anandhu336/AI-Driven-Flood-Prediction-and-Mapping-System.git
cd AI-Driven-Flood-Prediction-and-Mapping-System
pip install streamlit pandas numpy scikit-learn xgboost tensorflow pillow matplotlib joblib requests
streamlit run app.py
```

Note: `app.py` currently loads the background image and the Random Forest/XGBoost models from absolute local paths (under `/Users/anandhu/...`). Update those paths (or move the referenced files into the repo) before running the app on another machine.

## Usage

1. Launch the app and use the sidebar to pick a section.
2. In Image Segmentation, upload a flood-affected area photo to see the predicted flood mask.
3. In Flood Prediction, choose a default risk case or set your own environmental/infrastructure values, then click "Predict Flood Risk" to see the Random Forest and XGBoost predictions.
