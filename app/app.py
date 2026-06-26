import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(os.path.abspath(__file__)) in sys.path:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pandas as pd
import streamlit as st
import config
from src.evaluation.predict import predict_diagnostic_risk
from app.utils import execute_pipeline, check_plots_exist

# Page setup for premium brand experience
st.set_page_config(
    page_title="AuraMed Diagnostic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom premium CSS for glassmorphic styles and dark-mode feel
st.markdown("""
<style>
    .main {
        background-color: #080c16;
        color: #f8fafc;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00f2fe, #a855f7);
        color: black !important;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 242, 254, 0.4);
    }
    div[data-testid="stMetricValue"] {
        color: #00f2fe;
        font-size: 28px;
        font-weight: 800;
    }
    .metric-card {
        background: rgba(17, 25, 46, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(16px);
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }
    .highlight-card-red {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .highlight-card-green {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown("<h1 style='text-align: center; color: #00f2fe; margin-bottom: 5px;'>✦ AURA<span style='color: white;'>MED</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 14px;'>Disease Diagnostic & Predictive Intelligence Engine</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)

# Sidebar Configuration Layout
st.sidebar.markdown("<h2 style='color:#00f2fe;'>⚙️ Configuration</h2>", unsafe_allow_html=True)

# Select Disease Dataset
dataset_name = st.sidebar.selectbox(
    "Target Disease Condition",
    options=["heart_disease", "diabetes", "breast_cancer"],
    format_func=lambda x: config.DATASET_METADATA[x]["name"]
)

# Select ML Model
algorithm_name = st.sidebar.selectbox(
    "Classifier Model",
    options=["random_forest", "xgboost", "svm", "logistic_regression"],
    format_func=lambda x: x.upper().replace("_", " ")
)

st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
st.sidebar.markdown("### 🧬 Pipeline Operations")

# Global training execution button
if st.sidebar.button("Run Global Model Training"):
    with st.spinner("Executing model pipeline..."):
        success, message = execute_pipeline()
        if success:
            st.sidebar.success("Training complete!")
            st.sidebar.text_area("Pipeline Logs", value=message, height=150)
            # Clear Streamlit cache to reload assets
            st.rerun()
        else:
            st.sidebar.error("Pipeline failure!")
            st.sidebar.text_area("Errors", value=message, height=150)

# Sidebar status
plots_present = check_plots_exist()
if plots_present:
    st.sidebar.info("🟢 Trained models & plots are loaded.")
else:
    st.sidebar.warning("🟡 Models not trained yet. Run Global Training.")

# Tabs configuration
tab1, tab2, tab3 = st.tabs([
    "🎯 Interactive Diagnostic Tool", 
    "📈 Model Analytics & Validation", 
    "📊 Exploratory Data Analysis"
])

# Metadata helper variables
metadata = config.DATASET_METADATA[dataset_name]
target_col = metadata["target_col"]

# Tab 1: Interactive Predictor
with tab1:
    st.header("Patient Bio-metrics Input Form")
    st.write("Provide parameters below to run real-time diagnostic risk estimation.")
    
    # Check if models are available before running prediction
    scaler_path = os.path.join(config.SAVED_MODELS_DIR, f"{dataset_name}_scaler.pkl")
    if not os.path.exists(scaler_path):
        st.warning("⚠️ Trained models not found for this dataset. Please run 'Run Global Model Training' in the sidebar first.")
    else:
        # Dynamic inputs generation
        inputs = {}
        
        # Grid layout for parameters
        cols = st.columns(3)
        for idx, feat in enumerate(metadata["features"]):
            col = cols[idx % 3]
            with col:
                if feat["type"] == "select":
                    # Dropdown select boxes
                    options_map = {opt["value"]: opt["label"] for opt in feat["options"]}
                    selected_label = st.selectbox(
                        label=feat["label"],
                        options=list(options_map.values()),
                        help=feat["help"],
                        key=f"input_{dataset_name}_{feat['name']}"
                    )
                    # Resolve label to numerical value
                    val = [k for k, v in options_map.items() if v == selected_label][0]
                    inputs[feat["name"]] = val
                else:
                    # Slider or input box for continuous numbers
                    inputs[feat["name"]] = st.number_input(
                        label=feat["label"],
                        min_value=float(feat["min"]),
                        max_value=float(feat["max"]),
                        value=float(feat["default"]),
                        step=float(feat["step"]),
                        help=feat["help"],
                        key=f"input_{dataset_name}_{feat['name']}"
                    )
                    
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Inference prediction trigger
        if st.button("Execute Diagnostic Inference"):
            with st.spinner("Analyzing patient vitals..."):
                try:
                    result = predict_diagnostic_risk(dataset_name, algorithm_name, inputs)
                    
                    st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
                    st.subheader("Diagnostic Assessment Results")
                    
                    # Columns layout for metrics and dial
                    left_col, right_col = st.columns([1, 2])
                    
                    with left_col:
                        prob_percent = int(result["probability"] * 100)
                        st.markdown(f"### Probability Score")
                        st.markdown(f"<h1 style='color:#00f2fe; font-size:64px; font-weight:800; margin:0;'>{prob_percent}%</h1>", unsafe_allow_html=True)
                        st.write("Estimated Pathological Probability")
                        
                    with right_col:
                        if result["disease_detected"]:
                            st.markdown(f"""
                            <div class="highlight-card-red">
                                <h3 style="color:#ef4444; margin-top:0;">⚠️ POSITIVE DIAGNOSIS RISK DETECTED</h3>
                                <p style="color:#f8fafc; font-size:14px; margin-bottom:0;">
                                    The {algorithm_name.upper()} model predicts a HIGH probability of disease outcome. 
                                    A clinical checkup and secondary screening are recommended.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="highlight-card-green">
                                <h3 style="color:#10b981; margin-top:0;">✅ NEGATIVE DIAGNOSIS (LOW RISK)</h3>
                                <p style="color:#f8fafc; font-size:14px; margin-bottom:0;">
                                    The model evaluates diagnostic metrics at low risk thresholds.
                                    Normal biometric patterns observed.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    st.markdown("<br>", unsafe_allow_html=True)
                    # Breakdown factors
                    with st.expander("Show Patient Factors Summary"):
                        st.write(pd.DataFrame([inputs]))
                        
                except Exception as ex:
                    st.error(f"Inference error: {str(ex)}")

# Tab 2: Model Analytics & Figures
with tab2:
    st.header("Classifier Performance Reports")
    st.write("Validation reports generated on the holdout test partition.")
    
    if not plots_present:
        st.warning("⚠️ Metrics plots are not compiled yet. Run 'Run Global Model Training' in the sidebar.")
    else:
        # Load figures
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm_img_path = os.path.join(config.FIGURES_DIR, f"confusion_matrix_heart.png" if dataset_name == "heart_disease" else "confusion_matrix_cancer.png" if dataset_name == "breast_cancer" else "confusion_matrix.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_container_width=True)
                
            st.subheader("Feature Importance")
            feat_img_path = os.path.join(config.FIGURES_DIR, "feature_importance_diabetes.png" if dataset_name == "diabetes" else "feature_importance.png" if dataset_name == "breast_cancer" else "feature_importance.png")
            # If heart disease (SVM), SVM doesn't have simple feature importance plot
            if dataset_name == "heart_disease":
                st.info("Feature importance plots are supported for Random Forest and XGBoost. SVM uses target-feature correlations.")
            elif os.path.exists(feat_img_path):
                st.image(feat_img_path, use_container_width=True)
                
        with col2:
            st.subheader("ROC Curve Comparison")
            roc_img_path = os.path.join(config.FIGURES_DIR, f"roc_curves_{dataset_name}.png")
            if os.path.exists(roc_img_path):
                st.image(roc_img_path, use_container_width=True)
                
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
        
        # Load and render final_report.md
        report_path = os.path.join(config.REPORTS_DIR, "final_report.md")
        if os.path.exists(report_path):
            st.subheader("Global Metrics Summary table")
            with open(report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())

# Tab 3: Exploratory Data Analysis
with tab3:
    st.header("Exploratory Data Analysis (EDA)")
    
    # Load raw dataset to display
    try:
        raw_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, metadata["raw_filename"]))
        
        st.subheader("Raw Dataset Preview")
        st.dataframe(raw_df.head(20), use_container_width=True)
        
        # Show key statistics
        st.subheader("Dataset Summary Statistics")
        st.write(raw_df.describe())
        
        # Load EDA plots
        st.subheader("Data Distributions & Correlations")
        eda_img_path = os.path.join(config.FIGURES_DIR, f"eda_{dataset_name}_plots.png")
        if os.path.exists(eda_img_path):
            st.image(eda_img_path, use_container_width=True)
            
    except Exception as ex:
        st.info("Dataset raw files not loaded. Run global model training in the sidebar to download and compile data splits.")
