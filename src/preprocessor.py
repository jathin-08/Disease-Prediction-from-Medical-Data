import sys
import os
# Inject project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config
from src.data_loader import get_raw_dataset

def preprocess_and_split_all():
    """
    Loads all datasets, splits into train/test, scales numerical features,
    saves the StandardScaler to saved_models/ and saves splits to data/processed/.
    """
    datasets_mapping = {
        "heart_disease": {
            "train_file": "heart_train.csv",
            "test_file": "heart_test.csv"
        },
        "diabetes": {
            "train_file": "diabetes_train.csv",
            "test_file": "diabetes_test.csv"
        },
        "breast_cancer": {
            "train_file": "cancer_train.csv",
            "test_file": "cancer_test.csv"
        }
    }

    for name, files in datasets_mapping.items():
        print(f"Preprocessing and splitting {name} dataset...")
        df = get_raw_dataset(name)
        target_col = config.DATASET_METADATA[name]["target_col"]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Train / test split (using stratify for class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Scale inputs
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler to saved_models/
        scaler_path = os.path.join(config.SAVED_MODELS_DIR, f"{name}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        # Reconstruct scaled DataFrames
        df_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        df_train_scaled[target_col] = y_train.values
        
        df_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        df_test_scaled[target_col] = y_test.values
        
        # Save to data/processed/
        train_path = os.path.join(config.PROCESSED_DATA_DIR, files["train_file"])
        test_path = os.path.join(config.PROCESSED_DATA_DIR, files["test_file"])
        
        df_train_scaled.to_csv(train_path, index=False)
        df_test_scaled.to_csv(test_path, index=False)
        
        print(f"Processed splits saved to {train_path} and {test_path}")

def load_processed_data(dataset_name):
    """Loads and returns train/test features and labels for a specific dataset."""
    datasets_mapping = {
        "heart_disease": {
            "train_file": "heart_train.csv",
            "test_file": "heart_test.csv"
        },
        "diabetes": {
            "train_file": "diabetes_train.csv",
            "test_file": "diabetes_test.csv"
        },
        "breast_cancer": {
            "train_file": "cancer_train.csv",
            "test_file": "cancer_test.csv"
        }
    }
    
    if dataset_name not in datasets_mapping:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    files = datasets_mapping[dataset_name]
    train_path = os.path.join(config.PROCESSED_DATA_DIR, files["train_file"])
    test_path = os.path.join(config.PROCESSED_DATA_DIR, files["test_file"])
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        # Trigger preprocessing if splits are missing
        preprocess_and_split_all()
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    target_col = config.DATASET_METADATA[dataset_name]["target_col"]
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    return X_train, X_test, y_train, y_test
