import sys
import os
# Inject project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import config

def load_raw_heart_disease():
    """
    Loads raw Cleveland Heart Disease dataset.
    Downloads from UCI to data/raw/ or uses cached file. Fallback: generates synthetic.
    """
    raw_path = os.path.join(config.RAW_DATA_DIR, config.DATASET_METADATA["heart_disease"]["raw_filename"])
    
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)

    print("Heart Disease dataset not found locally. Downloading from UCI repository...")
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    try:
        df = pd.read_csv(config.HEART_URL, names=columns, na_values="?")
        # Impute missing cells with median values
        df = df.fillna(df.median())
        # Convert target to binary classification (0: normal, 1: disease)
        df['target'] = (df['target'] > 0).astype(int)
        df.to_csv(raw_path, index=False)
        print(f"Heart Disease dataset saved to {raw_path}")
        return df
    except Exception as e:
        print(f"Error downloading Heart Disease dataset: {e}. Generating synthetic fallback.")
        return generate_synthetic_heart_disease(raw_path)

def generate_synthetic_heart_disease(save_path):
    np.random.seed(42)
    n_samples = 300
    
    age = np.random.normal(54, 9, n_samples).astype(int)
    sex = np.random.binomial(1, 0.68, n_samples)
    cp = np.random.choice([1, 2, 3, 4], n_samples, p=[0.08, 0.16, 0.28, 0.48])
    trestbps = np.random.normal(131, 17, n_samples).astype(int)
    chol = np.random.normal(246, 51, n_samples).astype(int)
    fbs = np.random.binomial(1, 0.15, n_samples)
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.49, 0.01, 0.50])
    thalach = np.random.normal(149, 23, n_samples).astype(int)
    exang = np.random.binomial(1, 0.32, n_samples)
    oldpeak = np.random.exponential(1.0, n_samples)
    oldpeak = np.clip(oldpeak, 0.0, 6.2)
    slope = np.random.choice([1, 2, 3], n_samples, p=[0.47, 0.46, 0.07])
    ca = np.random.choice([0.0, 1.0, 2.0, 3.0], n_samples, p=[0.58, 0.22, 0.13, 0.07])
    thal = np.random.choice([3.0, 6.0, 7.0], n_samples, p=[0.55, 0.06, 0.39])
    
    # Probability logic
    z = -3.0 + 0.03 * age + 0.5 * sex + 0.6 * (cp == 4) + 0.01 * trestbps + 0.005 * chol + 0.8 * exang - 0.02 * thalach + 1.0 * oldpeak + 0.8 * ca
    prob = 1 / (1 + np.exp(-z))
    target = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': np.round(oldpeak, 1), 'slope': slope, 'ca': ca, 'thal': thal,
        'target': target
    })
    
    df['age'] = df['age'].clip(29, 80)
    df['trestbps'] = df['trestbps'].clip(90, 200)
    df['chol'] = df['chol'].clip(120, 560)
    df['thalach'] = df['thalach'].clip(70, 210)
    
    df.to_csv(save_path, index=False)
    print(f"Synthetic Heart Disease dataset saved to {save_path}")
    return df

def load_raw_diabetes():
    """
    Loads raw Pima Indians Diabetes dataset.
    Downloads to data/raw/ or uses cached file. Fallback: generates synthetic.
    """
    raw_path = os.path.join(config.RAW_DATA_DIR, config.DATASET_METADATA["diabetes"]["raw_filename"])
    
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)

    print("Diabetes dataset not found locally. Downloading raw file...")
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    try:
        df = pd.read_csv(config.DIABETES_URL, names=columns)
        df.to_csv(raw_path, index=False)
        print(f"Diabetes dataset saved to {raw_path}")
        return df
    except Exception as e:
        print(f"Error downloading Diabetes dataset: {e}. Generating synthetic fallback.")
        return generate_synthetic_diabetes(raw_path)

def generate_synthetic_diabetes(save_path):
    np.random.seed(42)
    n_samples = 768
    
    pregnancies = np.random.poisson(3.8, n_samples)
    glucose = np.random.normal(120, 30, n_samples).astype(int)
    blood_pressure = np.random.normal(69, 12, n_samples).astype(int)
    skin_thickness = np.random.normal(20, 15, n_samples).astype(int)
    insulin = np.random.normal(80, 115, n_samples).astype(int)
    bmi = np.random.normal(32, 7, n_samples)
    diabetes_pedigree = np.random.exponential(0.47, n_samples)
    age = np.random.normal(33, 11, n_samples).astype(int)
    
    pregnancies = np.clip(pregnancies, 0, 17)
    glucose = np.clip(glucose, 0, 200)
    blood_pressure = np.clip(blood_pressure, 0, 122)
    skin_thickness = np.clip(skin_thickness, 0, 99)
    insulin = np.clip(insulin, 0, 846)
    bmi = np.clip(bmi, 0.0, 67.1)
    diabetes_pedigree = np.clip(diabetes_pedigree, 0.078, 2.42)
    age = np.clip(age, 21, 81)
    
    z = -6.0 + 0.1 * pregnancies + 0.03 * glucose + 0.01 * blood_pressure + 0.005 * insulin + 0.08 * bmi + 0.9 * diabetes_pedigree + 0.02 * age
    prob = 1 / (1 + np.exp(-z))
    outcome = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': np.round(bmi, 1),
        'DiabetesPedigreeFunction': np.round(diabetes_pedigree, 3), 'Age': age,
        'Outcome': outcome
    })
    df.to_csv(save_path, index=False)
    print(f"Synthetic Diabetes dataset saved to {save_path}")
    return df

def load_raw_breast_cancer():
    """
    Loads raw Wisconsin Breast Cancer dataset.
    Loads from sklearn, converts target to binary diagnosis, saves to data/raw/.
    """
    raw_path = os.path.join(config.RAW_DATA_DIR, config.DATASET_METADATA["breast_cancer"]["raw_filename"])
    
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)

    print("Loading Breast Cancer dataset via sklearn.datasets...")
    try:
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        df = df.rename(columns={'target': 'Outcome'})
        # Map 1: malignant (cancer), 0: benign
        df['Outcome'] = 1 - df['Outcome']
        
        key_features = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 
            'mean fractal dimension', 'Outcome'
        ]
        df = df[key_features]
        df.columns = [col.replace(' ', '_').title() for col in df.columns]
        df.to_csv(raw_path, index=False)
        print(f"Breast Cancer dataset saved to {raw_path}")
        return df
    except Exception as e:
        print(f"Error loading Breast Cancer dataset from sklearn: {e}. Generating synthetic fallback.")
        return generate_synthetic_breast_cancer(raw_path)

def generate_synthetic_breast_cancer(save_path):
    np.random.seed(42)
    n_samples = 569
    
    mean_radius = np.random.normal(14.1, 3.5, n_samples)
    mean_texture = np.random.normal(19.3, 4.3, n_samples)
    mean_perimeter = mean_radius * 6.28 + np.random.normal(0, 5, n_samples)
    mean_area = 3.14 * (mean_radius ** 2) + np.random.normal(0, 50, n_samples)
    mean_smoothness = np.random.normal(0.096, 0.014, n_samples)
    mean_compactness = np.random.normal(0.104, 0.052, n_samples)
    mean_concavity = np.random.normal(0.088, 0.079, n_samples)
    mean_concave_points = np.random.normal(0.048, 0.038, n_samples)
    mean_symmetry = np.random.normal(0.181, 0.027, n_samples)
    mean_fractal_dimension = np.random.normal(0.062, 0.007, n_samples)
    
    mean_radius = np.clip(mean_radius, 6.9, 28.1)
    mean_texture = np.clip(mean_texture, 9.7, 39.2)
    mean_perimeter = np.clip(mean_perimeter, 43.7, 188.5)
    mean_area = np.clip(mean_area, 143.5, 2501.0)
    mean_smoothness = np.clip(mean_smoothness, 0.05, 0.16)
    mean_compactness = np.clip(mean_compactness, 0.01, 0.34)
    mean_concavity = np.clip(mean_concavity, 0.0, 0.42)
    mean_concave_points = np.clip(mean_concave_points, 0.0, 0.20)
    mean_symmetry = np.clip(mean_symmetry, 0.1, 0.3)
    mean_fractal_dimension = np.clip(mean_fractal_dimension, 0.04, 0.097)
    
    z = -10.0 + 0.5 * mean_radius + 0.1 * mean_texture + 30 * mean_concave_points + 10 * mean_concavity
    prob = 1 / (1 + np.exp(-z))
    outcome = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'Mean_Radius': np.round(mean_radius, 2),
        'Mean_Texture': np.round(mean_texture, 2),
        'Mean_Perimeter': np.round(mean_perimeter, 2),
        'Mean_Area': np.round(mean_area, 1),
        'Mean_Smoothness': np.round(mean_smoothness, 4),
        'Mean_Compactness': np.round(mean_compactness, 4),
        'Mean_Concavity': np.round(mean_concavity, 4),
        'Mean_Concave_Points': np.round(mean_concave_points, 4),
        'Mean_Symmetry': np.round(mean_symmetry, 4),
        'Mean_Fractal_Dimension': np.round(mean_fractal_dimension, 4),
        'Outcome': outcome
    })
    df.to_csv(save_path, index=False)
    print(f"Synthetic Breast Cancer dataset saved to {save_path}")
    return df

def get_raw_dataset(dataset_name):
    """Retrieves specified dataset as a pandas dataframe."""
    if dataset_name == "heart_disease":
        return load_raw_heart_disease()
    elif dataset_name == "diabetes":
        return load_raw_diabetes()
    elif dataset_name == "breast_cancer":
        return load_raw_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
