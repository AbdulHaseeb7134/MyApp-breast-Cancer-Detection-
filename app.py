import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# 1. Initialize global variables to prevent NameErrors
model = None

# 2. Page Configuration
st.set_page_config(
    page_title="Abdul Haseeb | AI Portfolio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Advanced Premium CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 42px !important;
        font-weight: 800;
        background: linear-gradient(45deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: left;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    .sub-title {
        font-size: 19px !important;
        color: #A0AEC0;
        margin-bottom: 25px;
        font-weight: 400;
    }
    .group-header {
        font-size: 22px !important;
        font-weight: 600;
        color: #E2E8F0;
        border-bottom: 2px solid #3182CE;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .info-card {
        background-color: #1A202C;
        padding: 22px;
        border-radius: 12px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .profile-card {
        background-color: #1A202C;
        padding: 15px;
        border-radius: 8px;
        border-top: 3px solid #3182CE;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Model Loading Pipeline (Smart Auto-Detect File Name)
@st.cache_resource
def load_model_file():
    import os
    # We check all possible names you might have saved it as
    possible_names = ['svc_model.pkl', 'Breast-Cancer-Detection', 'Breast-Cancer-Detection.pkl']
    
    for name in possible_names:
        if os.path.exists(name):
            with open(name, 'rb') as file:
                return pickle.load(file)
    
    # If none match, raise the error
    raise FileNotFoundError("Could not find the model file.")

try:
    model = load_model_file()
except FileNotFoundError:
    st.error("🚨 Core model file not found in your folder! Make sure your saved model file (like 'svc_model.pkl' or 'Breast-Cancer-Detection') is sitting right next to this app.py file inside your Downloads folder.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Model loading error: {e}")
    st.stop()

# Load Dataset Specifications
cancer_data = load_breast_cancer()
feature_names = cancer_data.feature_names

# 5. Premium Sidebar: Abdul Haseeb's Academic Identity
with st.sidebar:
    st.markdown("## 🎗️ Executive Terminal")
    st.markdown("---")
    
    st.markdown("### 👨‍💻 Developer Profile")
    st.markdown("""
    <div class="profile-card">
        <strong style="color:#FFF; font-size:16px;">Abdul Haseeb</strong><br>
        <span style="color:#A0AEC0; font-size:14px;">BS Artificial Intelligence</span><br>
        <span style="color:#3182CE; font-size:13px; font-weight:600;">Semester 4</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Course Project:**")
    st.info("🧠 Programming for AI")
    
    st.markdown("---")
    st.markdown("### 📊 Project Architecture")
    st.markdown("""
    * **Model:** Support Vector Classifier (SVC)
    * **Optimization:** GridSearchCV Hyperparameter Tuning
    * **Accuracy Achieved:** **97.3%**
    * **Dataset:** Wisconsin Diagnostic Breast Cancer
    """)
    st.markdown("---")
    st.caption("© 2026 Abdul Haseeb | Academic Portfolio Project")

# 6. Main Hero Section / Presentation Canvas
st.markdown('<div class="main-title">Haseeb-AI | Intelligent Tumor Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Machine Learning Predictive System for Cellular Nuclei Evaluation</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <strong>🔬 Student Project Note:</strong> Welcome to my Programming for AI final semester submission. 
    This interactive gateway maps 30 distinct geometrical measurements taken from fine needle aspirate (FNA) 
    biopsy images into a highly optimized hyper-dimensional boundary to isolate malignant abnormalities.
</div>
""", unsafe_allow_html=True)

# 7. Interactive Metric Matrix Configuration (Clean Tabs Layout)
st.markdown('<div class="group-header">📊 Cellular Geometry Matrix (Adjust Values)</div>', unsafe_allow_html=True)

input_dict = {}
tab1, tab2, tab3 = st.tabs(["🔬 Mean Attributes", "📐 Standard Error (SE)", "⚠️ Worst Metrics"])

with tab1:
    col1, col2 = st.columns(2, gap="medium")
with tab2:
    col3, col4 = st.columns(2, gap="medium")
with tab3:
    col5, col6 = st.columns(2, gap="medium")

for i, feature in enumerate(feature_names):
    default_val = float(np.mean(cancer_data.data[:, i]))
    display_label = feature.replace(" error", " SE").title()
    
    if "mean" in feature:
        if i % 2 == 0:
            with col1: input_dict[feature] = st.number_input(display_label, value=default_val, format="%.4f")
        else:
            with col2: input_dict[feature] = st.number_input(display_label, value=default_val, format="%.4f")
    elif "error" in feature:
        if i % 2 == 0:
            with col3: input_dict[feature] = st.number_input(display_label, value=default_val, format="%.4f")
        else:
            with col4: input_dict[feature] = st.number_input(display_label, value=default_val, format="%.4f")
    else:
        if i % 2 == 0:
            with col5: input_dict[feature] = st.number_input(display_label, value=default_val, format="%.4f")
        else:
            with col6: input_dict[feature] = st.number_input(display_label, value=default_val, format="%.4f")

# 8. Model Execution & Diagnosis Display
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="group-header">⚡ Core Inference Engine</div>', unsafe_allow_html=True)

if st.button("RUN DIAGNOSTIC CLASSIFICATION PIPELINE", type="primary", use_container_width=True):
    if model is not None:
        features_df = pd.DataFrame([input_dict])[feature_names]
        
        with st.spinner("Processing vectors through optimized SVC boundary..."):
            prediction = model.predict(features_df)
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Pipeline Status", value="Execution Complete")
            st.metric(label="Calculated Model Accuracy", value="97.3 %")
            
        with res_col2:
            if prediction[0] == 0:
                st.error("### 🚨 Diagnostic Output: MALIGNANT")
                st.markdown("The computed mathematical parameters correlate intensely with clinical records of **malignant** samples. Immediate professional verification is suggested.")
            else:
                st.success("### ✅ Diagnostic Output: BENIGN")
                st.markdown("The computed mathematical parameters fall safely within standard **benign** classification clusters. Regular medical follow-ups remain recommended.")
    else:
        st.error("Model core is uninitialized. Please restart the application server.")