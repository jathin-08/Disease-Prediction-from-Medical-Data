import sys
import os
# Inject project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import config

def get_model_and_scaler(dataset_name, algorithm_name):
    """
    Retrieves the serialized classifier and corresponding standard scaler.
    Fits the file names requested in directory mapping.
    """
    model_filename = ""
    if dataset_name == "heart_disease" and algorithm_name == "svm":
        model_filename = "svm_heart.pkl"
    elif dataset_name == "diabetes" and algorithm_name == "random_forest":
        model_filename = "rf_diabetes.pkl"
    elif dataset_name == "breast_cancer" and algorithm_name == "xgboost":
        model_filename = "xgb_cancer.pkl"
    else:
        # Generic lookup fallback if they choose other models
        model_filename = "best_model.pkl"
        
    model_path = os.path.join(config.SAVED_MODELS_DIR, model_filename)
    scaler_path = os.path.join(config.SAVED_MODELS_DIR, f"{dataset_name}_scaler.pkl")
    
    # Check fallback if files don't exist
    if not os.path.exists(model_path):
        model_path = os.path.join(config.SAVED_MODELS_DIR, "best_model.pkl")
        
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Trained models or scalers not found. Run the training script first.")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    return model, scaler

def predict_diagnostic_risk(dataset_name, algorithm_name, inputs):
    """
    Executes inference for a single patient profile.
    inputs: dict mapping feature names to biometric values.
    """
    model, scaler = get_model_and_scaler(dataset_name, algorithm_name)
    metadata = config.DATASET_METADATA[dataset_name]
    
    # Align features in correct schema order
    ordered_features = []
    for feat in metadata["features"]:
        val = inputs.get(feat["name"])
        if val is None:
            val = feat["default"]
        ordered_features.append(float(val))
        
    # Scale inputs
    input_arr = np.array([ordered_features])
    input_scaled = scaler.transform(input_arr)
    
    # Make prediction
    pred_class = int(model.predict(input_scaled)[0])
    
    # Get probability if model supports it
    prob = 0.5
    try:
        prob = float(model.predict_proba(input_scaled)[0][1])
    except Exception:
        # Fallback if probability is not supported by backend wrapper
        pass
        
    return {
        "prediction_class": pred_class,
        "probability": round(prob, 4),
        "disease_detected": bool(pred_class)
    }
