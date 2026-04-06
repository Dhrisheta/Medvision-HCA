# MedVision AI

This is a lightweight, mobile-responsive web application that analyzes medical images (such as skin lesions) using a pre-trained deep learning model (MobileNetV2). The app applies **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight the regions the AI found most significant in making its prediction, helping visualize the \"areas of concern\".

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

## Features
- **Live Camera & File Upload**: Snap a photo using your phone or upload an existing image.
- **AI Classification**: Processes the image and provides prediction percentages.
- **Grad-CAM Visualization**: Generates a heatmap overlaid on the image to show AI focus areas.
- **Mobile-Responsive UI**: Clean interface built with Streamlit.

## Disclaimer
> **⚠️ MEDICAL DISCLAIMER**: This application is for **educational and demonstrational purposes ONLY**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Setup Instructions

### 1. Prerequisites
Make sure you have Python 3.8 or higher installed on your system. 

### 2. Installation
Navigate to the project folder and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Running the App
Once the dependencies are installed, you can start the Streamlit server:

```bash
streamlit run app.py
```

This will automatically open the application in your default web browser (typically at `http://localhost:8501`).

## Project Structure
- `app.py`: The main Streamlit web application with UI components.
- `model_utils.py`: Utilities for loading the MobileNetV2 model and preprocessing input images.
- `gradcam.py`: Implementation of the Grad-CAM algorithm for generating heatmaps.
- `requirements.txt`: Python package dependencies.

## Usage Notes for Medical Application
Currently, the application uses **MobileNetV2** with standard `ImageNet` weights to serve as a demonstrational template. ImageNet is a generic object dataset and lacks the specific knowledge to diagnose medical pathologies (like identifying Melanoma). 

To adapt this into a true medical tool:
1. Train a model on a medical dataset (e.g., the ISIC skin lesion dataset).
2. Save your fine-tuned model as an `.h5` or `SavedModel` file.
3. Update `load_model()` in `model_utils.py` to point to your new model weights rather than ImageNet.
