import streamlit as st
from PIL import Image
import numpy as np
import model_utils
import gradcam
import cv_analyzer
import advanced_visuals
import chatbot_engine
import progress_tracker
import json

# Ensure page configuration happens FIRST
st.set_page_config(
    page_title="Advanced Lesion AI V2",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
<style>
/* Dark Medical Theme Variables */
:root {
    --primary-glow: #00f0ff;
    --medical-red: #ff3366;
    --dark-glass: rgba(20, 20, 30, 0.75);
    --border-glass: rgba(255, 255, 255, 0.1);
}

/* Base Body Style */
.stApp {
    background: radial-gradient(circle at top right, #111122 0%, #000000 100%);
    color: #ffffff;
}

/* Glassmorphism Containers */
div[data-testid="stVerticalBlock"] > div.element-container {
    background: var(--dark-glass);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
}

/* Removing glass from individual elements to prevent nesting issues */
div[data-testid="stVerticalBlock"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Header Text formatting */
h1, h2, h3 {
    background: -webkit-linear-gradient(45deg, #00f0ff, #0088ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-weight: 700;
}

/* Metrics Emphasis */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    color: var(--primary-glow) !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: rgba(10, 10, 15, 0.95);
    border-right: 1px solid var(--primary-glow);
}

hr {
    border-color: var(--primary-glow);
    opacity: 0.3;
}

/* Premium Chatbot Styling */
/* User Message Bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background-color: rgba(0, 136, 255, 0.15) !important;
    border: 1px solid rgba(0, 136, 255, 0.3);
    border-radius: 15px 15px 0px 15px;
    padding: 10px 20px;
    margin-bottom: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

/* Assistant Message Bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background: var(--dark-glass);
    border: 1px solid var(--border-glass);
    border-radius: 15px 15px 15px 0px;
    padding: 10px 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(10px);
}

/* Chat Input Field container cleanup */
div[data-testid="stChatInput"] {
    border: 1px solid var(--primary-glow) !important;
    border-radius: 20px !important;
    background: var(--dark-glass) !important;
}

</style>
""", unsafe_allow_html=True)

# Important Warning Notice
st.error(
    "**CLINICAL DEMO V2**: This is an advanced demonstration wrapper over a generic MobileNet architecture. "
    "Diagnoses are DETERMINISTICALLY SIMULATED for demonstration and are NOT medical diagnoses. "
    "DO NOT USE ON REAL PATIENTS.",
    icon="⚠️"
)

st.title("⚕️ Advanced Digital Dermatologist V2")
st.markdown("**Powered by Deep Learning & Computer Vision Edge Analytics**")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Control")
    st.info("MobileNetV2 Layer Engine: **Active**\n\nOpenCV Analytics: **Active**")
    heatmap_intensity = st.slider("Heatmap Overlay Intensity", 0.1, 1.0, 0.5, 0.05)
    st.divider()
    
    st.header("🧠 AI Assistant Context")
    grok_api_key = st.text_input("xAI API Key (Grok)", type="password", help="Enter an xAI API key for real interactive visual analysis via Grok. Otherwise, bot uses simulated mock mode.")
    
    st.divider()
    
    input_mode = st.radio("📡 Data Source:", ["Local File System", "Live Visual Stream (Camera)"])
    st.caption("The live camera will auto-analyze when a frame is captured.")

# Model Loading
@st.cache_resource
def load_app_model():
    with st.spinner("Initializing Deep Neural Network..."):
        return model_utils.load_model()

model = load_app_model()

# Image Input Routing
uploaded_file = None
st.markdown("### 📥 Input Feed")
col_input, _ = st.columns([1, 1])

with col_input:
    if input_mode == "Live Visual Stream (Camera)":
        st.info("💡 Smart Camera Guidance: Ensure the lesion is centered and well-lit before capturing.")
        uploaded_file = st.camera_input("Acquire Image")
    else:
        uploaded_file = st.file_uploader("Drop image data (.jpg, .png)", type=["jpg", "jpeg", "png"])

# Process Analysis Engine
if uploaded_file is not None:
    st.divider()
    
    # ---------------- UI REWRITE: 3-1 SPLIT ---------------- #
    col_main, col_chat = st.columns([3.5, 1.5], gap="large")
    
    # Pre-processing
    image = Image.open(uploaded_file).convert('RGB')
    img_array = model_utils.prepare_image(image)
    image_np = np.array(image)
    
    with st.spinner("Executing Pathology Scans..."):
        # 1. Advanced Disease Classification (Simulated via Hashing)
        predictions, raw_preds = model_utils.predict_class(model, img_array)
        top_disease = predictions[0][1]
        top_prob = float(predictions[0][2]) * 100.0
        
        # 2. Damage Area Calculation (OpenCV)
        damage_pct, boundary_map, lesion_mask = cv_analyzer.calculate_damage_percentage(image_np)

    ######## LEFT COLUMN (MAIN APP) ########
    with col_main:
        st.header("🧠 Real-time Analytics Dashboard")
        
        # Main Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Primary Diagnosis", value=f"{top_disease}")
        m2.metric(label="Confidence Rating", value=f"{top_prob:.1f}%", delta="High")
        m3.metric(label="Damage Area Coverage", value=f"{damage_pct:.1f}%", delta="Measured Surface" if damage_pct > 0 else "Clear")

        st.divider()
        
        # Risk Meter & Connect
        st.subheader("⚠️ Clinical Risk Assessment & Connect")
        risk_col1, risk_col2 = st.columns([2, 1])
        with risk_col1:
            risk_fig, risk_val = progress_tracker.generate_risk_meter_chart(top_disease, top_prob, damage_pct)
            st.plotly_chart(risk_fig, use_container_width=True)
            
        with risk_col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.success("Analysis Complete.")
            st.download_button(
                label="📥 Download Clinical Report",
                data=f"DIAGNOSIS REPORT\n\nCondition: {top_disease}\nConfidence: {top_prob:.1f}%\nSeverity Coverage: {damage_pct:.1f}%\nRisk Score: {risk_val:.1f}/100\n\n*Simulated demonstration report generated by AI.*",
                file_name="clinical_report.txt",
                mime="text/plain",
                use_container_width=True
            )
            st.button("📅 Book Consultation", type="primary", use_container_width=True)

        st.divider()

        # AR Smart Camera & Visual Diagnostics Core
        st.subheader("👁️ AR Detection & Model Reasoning")
        
        # 3. Grad-CAM processing 
        FINAL_CONV_LAYER = "out_relu"
        try:
            heatmap = gradcam.make_gradcam_heatmap(img_array, model, FINAL_CONV_LAYER)
            img_resize_np = np.array(image.resize((224, 224)))
            superimposed_img = gradcam.overlay_gradcam(img_resize_np, heatmap, alpha=heatmap_intensity)
            
            # Display Tri-View Images
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(image, use_column_width=True, caption="Source Truth")
            with c2:
                # Mock AR overlay instruction
                if input_mode == "Live Visual Stream (Camera)":
                    st.caption("✅ **AR Smart Guide**: Distance & lighting optimal.")
                st.image(boundary_map, use_column_width=True, caption="AR Boundary Extraction (Red)")
            with c3:
                st.image(superimposed_img, use_column_width=True, caption="Model Reasoning (Grad-CAM)")
                
        except Exception as e:
            st.error(f"Grad-CAM Extraction Failed: {e}")

        st.divider()

        # Progression Tracking
        st.subheader("📈 AI Progress Tracker")
        st.caption("Compare visual timeline and severity score improvements or worsening over the last 6 months.")
        prog_col1, prog_col2 = st.columns([1, 1])
        with prog_col1:
            past_img = progress_tracker.simulate_past_image(image_np)
            st.image([past_img, image_np], caption=["Simulated Past (T-6 Mths)", "Today's Assessment"])
            
        with prog_col2:
            sev_fig = progress_tracker.generate_severity_chart(damage_pct)
            st.plotly_chart(sev_fig, use_container_width=True)

        st.divider()
        st.subheader("🔬 Advanced Lesion Topology Metrics")
        
        # Color Palette, Radar Chart, 3D Elevation
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            st.markdown("##### Clinical Risk Radar")
            radar_fig = advanced_visuals.generate_radar_chart(top_disease, top_prob)
            st.plotly_chart(radar_fig, use_container_width=True)
            
        with adv_col2:
            st.markdown("##### Variegation Extract")
            hex_colors = advanced_visuals.extract_lesion_colors(image_np, lesion_mask)
            if hex_colors:
                html_str = "<div style='display: flex; gap: 5px; justify-content: space-between;'>"
                for c in hex_colors:
                    html_str += f'''
                    <div style="flex: 1;">
                        <div style="background-color: {c}; height: 50px; border-radius: 5px; border: 1px solid rgba(255,255,255,0.4);"></div>
                        <p style="text-align: center; font-size: 10px; margin-top: 5px;">{c}</p>
                    </div>
                    '''
                html_str += "</div>"
                st.markdown(html_str, unsafe_allow_html=True)
            else:
                st.info("Insufficient lesion contrast for color extraction.")
                
        with adv_col3:
            st.markdown("##### 3D Topographical Density")
            elevation_fig = advanced_visuals.generate_3d_elevation(image_np, lesion_mask)
            st.plotly_chart(elevation_fig, use_container_width=True)

    ######## RIGHT COLUMN (CHAT BOT) ########
    with col_chat:
        st.markdown("<div class='chatbot-container'>", unsafe_allow_html=True)
        st.header("🤖 Clinical Chat")
        st.caption("Contextually aware assistant.")
        st.divider()
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": f"The preliminary simulation suggests **{top_disease}**. Do you have any specific questions regarding the structural features, risk timeline, or next steps?"})
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("Ask a clinical query..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Processing neural query..."):
                    response = chatbot_engine.generate_chat_response(
                        prompt, image, grok_api_key, st.session_state.messages, top_disease
                    )
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown("</div>", unsafe_allow_html=True)
