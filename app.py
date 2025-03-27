import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report ,confusion_matrix  
import numpy as np


# Load models and scalers
models_and_scalers = {
    "logistics_binary": ("logistics_binary.joblib", "logistics_binary_scaler.joblib"),
    "logistics_ovr": ("logistics_ovr.joblib", "logistics_ovr_scaler.joblib"),
    "logistics_multinomial": ("logistics_multinomial.joblib", "logistics_multinomial_scaler.joblib"),
    "svm_binary": ("svm_binary.joblib", "svm_binary_scaler.joblib"),
    "svm_multi": ("svm_multi.joblib", "svm_multi_scaler.joblib")
}

# Set page config first
st.set_page_config(
    page_title="Iris Classifier Plus",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.9);
    }
    h1 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)

def load_models():
    # (Keep your existing model loading code here)
    return models_and_scalers  # Replace with your actual return

def create_input_pair(label, min_val, max_val, default_val, key_suffix):
    col1, col2 = st.columns(2)
    with col1:
        slider_val = st.slider(
            label, min_val, max_val, default_val,
            key=f"slider_{key_suffix}",
            help=f"Adjust {label.lower()}"
        )
    with col2:
        number_val = st.number_input(
            label, min_val, max_val, slider_val,
            step=0.1, format="%.1f",
            key=f"number_{key_suffix}",
            on_change=lambda: st.session_state.update({f"slider_{key_suffix}": st.session_state[f"number_{key_suffix}"]})
        )
    return number_val

def main():
    st.title("🌸 Iris Species Classifier Pro")
    st.markdown("Explore different ML models for iris flower classification with interactive features!")
    
    # Load models
    models = load_models()
    
    # Sidebar with model selection and info
    with st.sidebar:
        st.header("Model Configuration")
        model_options = [
            "logistics_binary",
            "logistics_ovr",
            "logistics_multinomial`",
            "svm_binary",
            "svm_multi"
        ]
        selected_model = st.selectbox("Choose Model", model_options)
        
        model_info = {
            "logistics_binary": "Distinguishes Setosa from other species",
            "svm_binary": "SVM classifier for Setosa vs others",
            # Add descriptions for other models
        }
        st.markdown(f"**Model Info:** {model_info.get(selected_model, '')}")
    
    # Input Section
    with st.expander("🔧 Input Features", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            sepal_length = create_input_pair("Sepal Length (cm)", 4.0, 8.0, 5.8, "sepal_len")
            sepal_width = create_input_pair("Sepal Width (cm)", 2.0, 4.5, 3.0, "sepal_wid")
        with cols[1]:
            petal_length = create_input_pair("Petal Length (cm)", 1.0, 7.0, 4.0, "petal_len")
            petal_width = create_input_pair("Petal Width (cm)", 0.1, 2.5, 1.3, "petal_wid")
        
        # Example buttons
        example_cols = st.columns(3)
        with example_cols[0]:
            if st.button("🌼 Setosa Example", help="Load typical Setosa measurements"):
                st.session_state.update({
                    "number_sepal_len": 5.1, "number_sepal_wid": 3.5,
                    "number_petal_len": 1.4, "number_petal_wid": 0.2
                })
        # Add similar buttons for other species
    
    # Real-time Prediction
    if st.checkbox("Enable Live Prediction", True):
        user_data = {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width
        }
        
        # Get model and scaler based on selection
        model, scaler = get_model_and_scaler(selected_model, models)
        
        # Make prediction
        prediction = predict_data(user_data, model, scaler)
        
        # Visualization columns
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.subheader("Prediction Result")
            species = get_species_name(prediction, selected_model)
            color = "#2ecc71" if "Setosa" in species else "#e74c3c"
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color};">
                <h2 style="color: white; text-align: center;">{species}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with viz_col2:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(scaler.transform(pd.DataFrame([user_data])))[0]
                fig = px.bar(
                    x=model.classes_, 
                    y=proba, 
                    color=model.classes_,
                    labels={'x': 'Species', 'y': 'Probability'},
                    title='Classification Confidence'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance Visualization
        if hasattr(model, 'coef_'):
            st.subheader("Feature Importance")
            coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_[0]
            fig = px.bar(
                x=user_data.keys(),
                y=coefs,
                labels={'x': 'Feature', 'y': 'Coefficient Value'},
                color=user_data.keys()
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Model Comparison Section
    with st.expander("📊 Model Performance Comparison"):
        # Add confusion matrices or performance metrics
        pass

def get_model_and_scaler(selected_model, models):
    # Map selected model to appropriate model and scaler
    # Implement based on your model structure
    return model, scaler

def get_species_name(prediction, model_type):
    # Add proper label mapping based on model type
    return "Setosa" if prediction == 0 else "Other"

if __name__ == "__main__":
    main()
