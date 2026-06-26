import os

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model serialization directory
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")

# Reports and figures directories
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Datasets URLs
DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Datasets Meta-information for Unified Modeling & UI Forms
DATASET_METADATA = {
    "heart_disease": {
        "name": "Heart Disease (Cleveland)",
        "description": "Predict the presence of heart disease using patient health metrics.",
        "target_col": "target",
        "raw_filename": "heart_disease.csv",
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
    },
    "diabetes": {
        "name": "Diabetes (Pima Indians)",
        "description": "Predict the likelihood of diabetes in female patients of Pima Indian heritage.",
        "target_col": "Outcome",
        "raw_filename": "diabetes.csv",
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
    },
    "breast_cancer": {
        "name": "Breast Cancer (Wisconsin)",
        "description": "Predict whether a breast mass is malignant or benign from tissue sample measurements.",
        "target_col": "Outcome",
        "raw_filename": "breast_cancer.csv",
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
}
