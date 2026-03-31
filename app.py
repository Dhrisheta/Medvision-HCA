import streamlit as st
from PIL import Image
import numpy as np
import model_utils
import gradcam

st.set_page_config(
    page_title="AI Lesion Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.warning(
    "MEDICAL DISCLAIMER: This application is for educational and demonstrational purposes ONLY.\n"
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment.\n"
    "Always seek the advice of your physician or other qualified health provider with any questions "
    "you may have regarding a medical condition."
)

st.title("AI Health Image Analyzer")
st.markdown("Upload an image of a skin lesion or body part. The AI will analyze it and generate a heatmap.")

with st.sidebar:
    st.header("Settings")
    st.info("Currently using MobileNetV2 (ImageNet pretrained).")
    heatmap_intensity = st.slider("Heatmap Overlay Intensity", 0.1, 1.0, 0.4, 0.1)

@st.cache_resource
def load_app_model():
    with st.spinner("Loading AI Model (MobileNetV2)..."):
        return model_utils.load_model()

model = load_app_model()

input_mode = st.radio("Choose image source:", ["Upload File", "Use Camera"])

uploaded_file = None
if input_mode == "Use Camera":
    uploaded_file = st.camera_input("Take a picture")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = model_utils.prepare_image(image)
    
    with st.spinner("Analyzing Image..."):
        predictions, raw_preds = model_utils.predict_class(model, img_array)
        FINAL_CONV_LAYER = "out_relu"
        
        # Grad-CAM visualization
        try:
            heatmap = gradcam.make_gradcam_heatmap(img_array, model, FINAL_CONV_LAYER)
            img_np = np.array(image.resize((224, 224)))
            superimposed_img = gradcam.overlay_gradcam(img_np, heatmap, alpha=heatmap_intensity)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("AI Attention Heatmap")
                st.image(superimposed_img, use_column_width=True, caption="Red areas indicate where the AI focused most.")
            
            st.divider()
            st.subheader("AI Analysis Results")
            for _, class_name, prob in predictions:
                col_name, col_prog = st.columns([1, 4])
                with col_name:
                    st.write(f"**{class_name.replace('_', ' ').capitalize()}**")
                with col_prog:
                    st.progress(float(prob))
                    st.write(f"{prob*100:.1f}%")
                    
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
