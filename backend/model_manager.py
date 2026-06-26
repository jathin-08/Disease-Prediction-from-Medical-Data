import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .data_loader import (
    load_heart_disease_dataset, 
    load_diabetes_dataset, 
    load_breast_cancer_dataset,
    get_dataset_metadata
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")
os.makedirs(MODEL_DIR, exist_ok=True)

# In-memory store for trained models and scalers to prevent excessive file reads
_trained_models = {}
_scalers = {}

def get_dataset(dataset_name):
    """Retrieves pandas dataframe for specified dataset."""
    if dataset_name == "heart_disease":
        return load_heart_disease_dataset()
    elif dataset_name == "diabetes":
        return load_diabetes_dataset()
    elif dataset_name == "breast_cancer":
        return load_breast_cancer_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def init_model(algorithm):
    """Initializes the classification model."""
    if algorithm == "svm":
        # probability=True is required to get predict_proba for SVM
        return SVC(probability=True, kernel="rbf", random_state=42)
    elif algorithm == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == "xgboost":
        # eval_metric='logloss' avoids warnings
        return XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def train_and_evaluate(dataset_name, algorithm):
    """
    Trains a model on the specified dataset and algorithm.
    Calculates detailed metrics, confusion matrix, ROC-AUC data, and feature importances.
    """
    df = get_dataset(dataset_name)
    metadata = get_dataset_metadata(dataset_name)
    target_col = metadata["target_col"]
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Standard scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train model
    model = init_model(algorithm)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Map confusion matrix components
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    if cm.size != 4:
        # Fallback if binary split resulted in single class
        if len(np.unique(y_test)) == 1:
            if y_test.iloc[0] == 0:
                tn = len(y_test)
            else:
                tp = len(y_test)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Subsample ROC data to keep payload size light (~20 points)
    step = max(1, len(fpr) // 20)
    fpr_sub = fpr[::step].tolist()
    tpr_sub = tpr[::step].tolist()
    # Ensure end point is included
    if fpr_sub[-1] != fpr[-1]:
        fpr_sub.append(fpr[-1])
        tpr_sub.append(tpr[-1])

    # Feature Importance
    importances = []
    if algorithm == "random_forest":
        importances = model.feature_importances_.tolist()
    elif algorithm == "xgboost":
        importances = model.feature_importances_.tolist()
    elif algorithm == "logistic_regression":
        # Use absolute value of coefficients
        importances = np.abs(model.coef_[0]).tolist()
        # Normalize to sum up to 1 for visual parity
        sum_imp = sum(importances)
        if sum_imp > 0:
            importances = [i / sum_imp for i in importances]
    else:
        # SVM doesn't expose clean feature importances for non-linear kernels.
        # Fallback: Absolute correlation with target variable on training set
        correlations = [abs(np.corrcoef(X_train.iloc[:, i], y_train)[0, 1]) for i in range(X_train.shape[1])]
        # Replace NaNs with 0
        correlations = [0.0 if np.isnan(c) else c for c in correlations]
        sum_corr = sum(correlations)
        if sum_corr > 0:
            importances = [c / sum_corr for c in correlations]
        else:
            importances = [1.0 / len(feature_names)] * len(feature_names)

    # Pair features with importance scores and sort
    feature_importance_list = sorted(
        [{"feature": f, "importance": round(imp, 4)} for f, imp in zip(feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )

    # Save to memory
    model_key = f"{dataset_name}_{algorithm}"
    _trained_models[model_key] = model
    _scalers[model_key] = scaler

    # Save to disk as cache
    try:
        with open(os.path.join(MODEL_DIR, f"{model_key}_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(MODEL_DIR, f"{model_key}_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
    except Exception as e:
        print(f"Error saving model cache to disk: {e}")

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "roc_curve": {
            "fpr": fpr_sub,
            "tpr": tpr_sub
        },
        "feature_importances": feature_importance_list
    }

def get_or_load_model(dataset_name, algorithm):
    """Helper to retrieve model and scaler, loading from disk cache if needed."""
    model_key = f"{dataset_name}_{algorithm}"
    
    # 1. Check in-memory store
    if model_key in _trained_models and model_key in _scalers:
        return _trained_models[model_key], _scalers[model_key]

    # 2. Check disk cache
    model_path = os.path.join(MODEL_DIR, f"{model_key}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{model_key}_scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            
            # Cache in memory
            _trained_models[model_key] = model
            _scalers[model_key] = scaler
            return model, scaler
        except Exception:
            pass

    # 3. Model not trained, perform training
    print(f"Model {model_key} not found in cache. Training fresh model...")
    train_and_evaluate(dataset_name, algorithm)
    return _trained_models[model_key], _scalers[model_key]

def make_prediction(dataset_name, algorithm, input_data):
    """
    Accepts patient features, scales them, and performs prediction.
    Returns prediction class and probability score.
    """
    model, scaler = get_or_load_model(dataset_name, algorithm)
    metadata = get_dataset_metadata(dataset_name)
    
    # Align input features in correct column order
    ordered_features = []
    for feat in metadata["features"]:
        val = input_data.get(feat["name"])
        if val is None:
            val = feat["default"]
        ordered_features.append(float(val))

    # Scale and predict
    input_arr = np.array([ordered_features])
    input_scaled = scaler.transform(input_arr)
    
    pred_class = int(model.predict(input_scaled)[0])
    prob_score = float(model.predict_proba(input_scaled)[0][1])

    return {
        "prediction_class": pred_class,
        "probability": round(prob_score, 4),
        "disease_detected": bool(pred_class)
    }

def get_eda_data(dataset_name):
    """
    Generates statistics and distributions for EDA charts.
    Returns:
      - Class distributions (target variable counts)
      - Summary statistics (means, ranges)
      - Key correlation coefficients
    """
    df = get_dataset(dataset_name)
    metadata = get_dataset_metadata(dataset_name)
    target_col = metadata["target_col"]

    # Target distribution
    counts = df[target_col].value_counts().to_dict()
    target_distribution = {
        "Normal / Negative": int(counts.get(0, 0)),
        "Disease / Positive": int(counts.get(1, 0))
    }

    # Summary statistics
    summary = {}
    for col in df.columns:
        if col == target_col:
            continue
        summary[col] = {
            "mean": round(float(df[col].mean()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2)
        }

    # Top feature correlations with target
    correlations = df.corr()[target_col].drop(target_col).fillna(0).to_dict()
    corr_list = sorted(
        [{"feature": k, "correlation": round(v, 4)} for k, v in correlations.items()],
        key=lambda x: abs(x["correlation"]),
        reverse=True
    )

    return {
        "target_distribution": target_distribution,
        "feature_summary": summary,
        "correlations": corr_list
    }
