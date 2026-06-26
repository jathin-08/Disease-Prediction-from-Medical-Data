import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# URLs for datasets
DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Create data cache directory if it doesn't exist
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def load_heart_disease_dataset():
    """
    Loads the Cleveland Heart Disease dataset.
    Attempts to download from UCI, otherwise generates synthetic data.
    """
    local_path = os.path.join(CACHE_DIR, "heart_disease.csv")
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception:
            pass

    print("Attempting to download Heart Disease dataset from UCI...")
    try:
        # Cleveland heart disease data
        df = pd.read_csv(HEART_URL, names=columns, na_values="?")
        # Fill missing values with median
        df = df.fillna(df.median())
        # The target column has values 0 (normal) and 1, 2, 3, 4 (heart disease)
        # Convert to binary classification: 0 = normal, 1 = disease
        df['target'] = (df['target'] > 0).astype(int)
        df.to_csv(local_path, index=False)
        return df
    except Exception as e:
        print(f"Failed to download Heart Disease dataset: {e}. Generating synthetic fallback.")
        return generate_synthetic_heart_disease(local_path)

def generate_synthetic_heart_disease(save_path=None):
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
    # limit oldpeak range
    oldpeak = np.clip(oldpeak, 0.0, 6.2)
    slope = np.random.choice([1, 2, 3], n_samples, p=[0.47, 0.46, 0.07])
    ca = np.random.choice([0.0, 1.0, 2.0, 3.0], n_samples, p=[0.58, 0.22, 0.13, 0.07])
    thal = np.random.choice([3.0, 6.0, 7.0], n_samples, p=[0.55, 0.06, 0.39])
    
    # Target correlation logic to make machine learning models work reasonably
    # High age, male sex, type 4 CP, high cholesterol, low thalach, high oldpeak increase disease probability
    z = -3.0 + 0.03 * age + 0.5 * sex + 0.6 * (cp == 4) + 0.01 * trestbps + 0.005 * chol + 0.8 * exang - 0.02 * thalach + 1.0 * oldpeak + 0.8 * ca
    prob = 1 / (1 + np.exp(-z))
    target = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': np.round(oldpeak, 1), 'slope': slope, 'ca': ca, 'thal': thal,
        'target': target
    })
    
    # Clip parameters to realistic medical ranges
    df['age'] = df['age'].clip(29, 80)
    df['trestbps'] = df['trestbps'].clip(90, 200)
    df['chol'] = df['chol'].clip(120, 560)
    df['thalach'] = df['thalach'].clip(70, 210)
    
    if save_path:
        df.to_csv(save_path, index=False)
    return df

def load_diabetes_dataset():
    """
    Loads the Pima Indians Diabetes dataset.
    Attempts to download from a public mirror, otherwise generates synthetic data.
    """
    local_path = os.path.join(CACHE_DIR, "diabetes.csv")
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]

    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception:
            pass

    print("Attempting to download Diabetes dataset...")
    try:
        df = pd.read_csv(DIABETES_URL, names=columns)
        df.to_csv(local_path, index=False)
        return df
    except Exception as e:
        print(f"Failed to download Diabetes dataset: {e}. Generating synthetic fallback.")
        return generate_synthetic_diabetes(local_path)

def generate_synthetic_diabetes(save_path=None):
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
    
    # Ensure realistic non-negative values
    pregnancies = np.clip(pregnancies, 0, 17)
    glucose = np.clip(glucose, 0, 200)
    blood_pressure = np.clip(blood_pressure, 0, 122)
    skin_thickness = np.clip(skin_thickness, 0, 99)
    insulin = np.clip(insulin, 0, 846)
    bmi = np.clip(bmi, 0.0, 67.1)
    diabetes_pedigree = np.clip(diabetes_pedigree, 0.078, 2.42)
    age = np.clip(age, 21, 81)
    
    # Target correlation logic
    z = -6.0 + 0.1 * pregnancies + 0.03 * glucose + 0.01 * blood_pressure + 0.005 * insulin + 0.08 * bmi + 0.9 * diabetes_pedigree + 0.02 * age
    prob = 1 / (1 + np.exp(-z))
    outcome = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': np.round(bmi, 1),
        'DiabetesPedigreeFunction': np.round(diabetes_pedigree, 3), 'Age': age,
        'Outcome': outcome
    })
    
    if save_path:
        df.to_csv(save_path, index=False)
    return df

def load_breast_cancer_dataset():
    """
    Loads the Wisconsin Breast Cancer (Diagnostic) dataset from scikit-learn.
    Since scikit-learn is installed, this is offline-capable and highly reliable.
    """
    local_path = os.path.join(CACHE_DIR, "breast_cancer.csv")
    
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception:
            pass

    try:
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        # Rename target column to make it standard
        df = df.rename(columns={'target': 'Outcome'})
        # In sklearn breast cancer, 0 = malignant, 1 = benign. 
        # Let's map it so 1 = malignant (cancer) and 0 = benign (normal)
        df['Outcome'] = 1 - df['Outcome']
        
        # Select key features instead of all 30 features to keep the form readable and practical
        key_features = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 
            'mean fractal dimension', 'Outcome'
        ]
        df = df[key_features]
        # Clean column names (replace spaces with underscores, capitalize)
        df.columns = [col.replace(' ', '_').title() for col in df.columns]
        df.to_csv(local_path, index=False)
        return df
    except Exception as e:
        print(f"Failed to load breast cancer dataset from sklearn: {e}. Generating synthetic fallback.")
        return generate_synthetic_breast_cancer(local_path)

def generate_synthetic_breast_cancer(save_path=None):
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
    
    # Clip values
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
    
    # Correlation
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
    
    if save_path:
        df.to_csv(save_path, index=False)
    return df

def get_dataset_metadata(dataset_name):
    """
    Returns feature explanations, descriptions, ranges and standard placeholders
    for dynamically building HTML forms.
    """
    if dataset_name == "heart_disease":
        return {
            "name": "Heart Disease (Cleveland)",
            "description": "Predict the presence of heart disease using patient health metrics.",
            "target_col": "target",
            "features": [
                {"name": "age", "label": "Age", "type": "number", "min": 1, "max": 120, "default": 54, "step": 1, "help": "Age in years"},
                {"name": "sex", "label": "Sex", "type": "select", "options": [{"value": 1, "label": "Male"}, {"value": 0, "label": "Female"}], "default": 1, "help": "Biological sex"},
                {"name": "cp", "label": "Chest Pain Type", "type": "select", "options": [
                    {"value": 1, "label": "Typical Angina"},
                    {"value": 2, "label": "Atypical Angina"},
                    {"value": 3, "label": "Non-anginal Pain"},
                    {"value": 4, "label": "Asymptomatic"}
                ], "default": 3, "help": "Chest pain classification"},
                {"name": "trestbps", "label": "Resting Blood Pressure (mmHg)", "type": "number", "min": 80, "max": 220, "default": 130, "step": 1, "help": "Resting blood pressure on admission"},
                {"name": "chol", "label": "Serum Cholesterol (mg/dl)", "type": "number", "min": 100, "max": 600, "default": 240, "step": 1, "help": "Serum cholesterol level"},
                {"name": "fbs", "label": "Fasting Blood Sugar > 120 mg/dl", "type": "select", "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}], "default": 0, "help": "Fasting blood sugar level"},
                {"name": "restecg", "label": "Resting ECG Results", "type": "select", "options": [
                    {"value": 0, "label": "Normal"},
                    {"value": 1, "label": "ST-T Wave Abnormality"},
                    {"value": 2, "label": "Left Ventricular Hypertrophy"}
                ], "default": 0, "help": "Resting electrocardiographic results"},
                {"name": "thalach", "label": "Max Heart Rate Achieved", "type": "number", "min": 60, "max": 220, "default": 150, "step": 1, "help": "Maximum heart rate achieved during exercise stress test"},
                {"name": "exang", "label": "Exercise Induced Angina", "type": "select", "options": [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}], "default": 0, "help": "Angina induced by exercise"},
                {"name": "oldpeak", "label": "ST Depression (oldpeak)", "type": "number", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1, "help": "ST depression induced by exercise relative to rest"},
                {"name": "slope", "label": "Slope of Peak Exercise ST", "type": "select", "options": [
                    {"value": 1, "label": "Upsloping"},
                    {"value": 2, "label": "Flat"},
                    {"value": 3, "label": "Downsloping"}
                ], "default": 1, "help": "The slope of the peak exercise ST segment"},
                {"name": "ca", "label": "Number of Major Vessels (0-3)", "type": "select", "options": [
                    {"value": 0, "label": "0"},
                    {"value": 1, "label": "1"},
                    {"value": 2, "label": "2"},
                    {"value": 3, "label": "3"}
                ], "default": 0, "help": "Number of major vessels colored by fluoroscopy"},
                {"name": "thal", "label": "Thalassemia Type", "type": "select", "options": [
                    {"value": 3, "label": "Normal"},
                    {"value": 6, "label": "Fixed Defect"},
                    {"value": 7, "label": "Reversable Defect"}
                ], "default": 3, "help": "Blood flow defect type"}
            ]
        }
    elif dataset_name == "diabetes":
        return {
            "name": "Diabetes (Pima Indians)",
            "description": "Predict the likelihood of diabetes in female patients of Pima Indian heritage.",
            "target_col": "Outcome",
            "features": [
                {"name": "Pregnancies", "label": "Pregnancies", "type": "number", "min": 0, "max": 20, "default": 3, "step": 1, "help": "Number of times pregnant"},
                {"name": "Glucose", "label": "Glucose Level (mg/dl)", "type": "number", "min": 0, "max": 200, "default": 117, "step": 1, "help": "Plasma glucose concentration a 2 hours in an oral glucose tolerance test"},
                {"name": "BloodPressure", "label": "Blood Pressure (mmHg)", "type": "number", "min": 0, "max": 150, "default": 72, "step": 1, "help": "Diastolic blood pressure"},
                {"name": "SkinThickness", "label": "Triceps Skin Fold Thickness (mm)", "type": "number", "min": 0, "max": 100, "default": 23, "step": 1, "help": "Triceps skin fold thickness"},
                {"name": "Insulin", "label": "2-Hour Serum Insulin (mu U/ml)", "type": "number", "min": 0, "max": 900, "default": 30, "step": 1, "help": "2-hour serum insulin"},
                {"name": "BMI", "label": "Body Mass Index (BMI)", "type": "number", "min": 0.0, "max": 70.0, "default": 32.0, "step": 0.1, "help": "Body mass index (weight in kg/(height in m)^2)"},
                {"name": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Value", "type": "number", "min": 0.01, "max": 3.0, "default": 0.37, "step": 0.001, "help": "Diabetes pedigree function (genetic score)"},
                {"name": "Age", "label": "Age", "type": "number", "min": 21, "max": 100, "default": 29, "step": 1, "help": "Age in years"}
            ]
        }
    elif dataset_name == "breast_cancer":
        return {
            "name": "Breast Cancer (Wisconsin)",
            "description": "Predict whether a breast mass is malignant or benign from tissue sample measurements.",
            "target_col": "Outcome",
            "features": [
                {"name": "Mean_Radius", "label": "Mean Radius", "type": "number", "min": 5.0, "max": 30.0, "default": 14.1, "step": 0.01, "help": "Mean of distances from center to points on the perimeter"},
                {"name": "Mean_Texture", "label": "Mean Texture", "type": "number", "min": 5.0, "max": 40.0, "default": 19.3, "step": 0.01, "help": "Standard deviation of gray-scale values"},
                {"name": "Mean_Perimeter", "label": "Mean Perimeter", "type": "number", "min": 40.0, "max": 200.0, "default": 92.0, "step": 0.01, "help": "Mean size of the core tumor perimeter"},
                {"name": "Mean_Area", "label": "Mean Area", "type": "number", "min": 100.0, "max": 2600.0, "default": 654.8, "step": 0.1, "help": "Mean size of the core tumor area"},
                {"name": "Mean_Smoothness", "label": "Mean Smoothness", "type": "number", "min": 0.01, "max": 0.20, "default": 0.096, "step": 0.0001, "help": "Mean of local variation in radius lengths"},
                {"name": "Mean_Compactness", "label": "Mean Compactness", "type": "number", "min": 0.01, "max": 0.40, "default": 0.104, "step": 0.0001, "help": "Mean of perimeter^2 / area - 1.0"},
                {"name": "Mean_Concavity", "label": "Mean Concavity", "type": "number", "min": 0.0, "max": 0.50, "default": 0.088, "step": 0.0001, "help": "Mean of severity of concave portions of the contour"},
                {"name": "Mean_Concave_Points", "label": "Mean Concave Points", "type": "number", "min": 0.0, "max": 0.25, "default": 0.048, "step": 0.0001, "help": "Mean for number of concave portions of the contour"},
                {"name": "Mean_Symmetry", "label": "Mean Symmetry", "type": "number", "min": 0.05, "max": 0.40, "default": 0.181, "step": 0.0001, "help": "Mean symmetry of tissue structure"},
                {"name": "Mean_Fractal_Dimension", "label": "Mean Fractal Dimension", "type": "number", "min": 0.01, "max": 0.10, "default": 0.062, "step": 0.0001, "help": "Mean for coastline approximation - 1.0"}
            ]
        }
    return None
