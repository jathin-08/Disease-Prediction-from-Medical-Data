# AuraMed: Disease Diagnostic & Predictive Intelligence

AuraMed is a high-fidelity web application and machine learning engine designed to predict the likelihood of three medical conditions (Heart Disease, Diabetes, and Breast Cancer) based on patient biometric data. It integrates standard python backend APIs with a custom glassmorphic user dashboard to train, validate, and compare four classification models in real-time.

## Key Features

- **Interactive Diagnostic Form**: Dynamically generates patient questionnaires depending on the active disease pathology.
- **Multiple Classifiers Support**: Implements and trains the following algorithms:
  - **Random Forest Classifier**: Ensemble of decision trees using bootstrap aggregation.
  - **XGBoost (Extreme Gradient Boosting)**: Gradient boosted trees with regularized objective functions.
  - **Support Vector Machine (SVM)**: Support vector classification with Radial Basis Function (RBF) non-linear kernel.
  - **Logistic Regression**: Linear classification regularized using L2 penalty.
- **Model Analytics Dashboard**: Computes validation metrics on a 25% holdout dataset:
  - Accuracy, Precision, Recall, F1-Score, and ROC-AUC score.
  - Confusion Matrix summarizing True/False Positive/Negative splits.
  - Live plotted Receiver Operating Characteristic (ROC) curve.
  - Feature Importance contributions.
- **Exploratory Data Analysis (EDA)**: Displays dataset parameters, class distributions, and correlation coefficients.
- **Robust Data Pipeline**: Automatically fetches datasets from UCI and scikit-learn public repositories with built-in synthetic backup generation for offline reliability.

---

## Directory Structure

```
code_alpha-2/
│
├── backend/
│   ├── __init__.py
│   ├── main.py            # FastAPI routing & serving static files
│   ├── data_loader.py     # Public dataset downloading and synthetic generator fallback
│   └── model_manager.py   # Model training, scaling, evaluation, and prediction pipeline
│
├── frontend/
│   ├── index.html         # User dashboard layout
│   ├── style.css          # Glassmorphic space-dark CSS theme
│   └── app.js             # Form generation, client state, and Chart.js integrations
│
├── requirements.txt       # Python dependencies list
├── .gitignore             # Git ignore definitions
└── README.md              # Project documentation
```

---

## Getting Started

### 1. Requirements
Ensure you have Python 3.10+ installed on your system.

### 2. Setup Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Application
Launch the FastAPI uvicorn development server:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Once running, navigate to `http://127.0.0.1:8000` in your web browser.

---

## Project Specifications

### Datasets:
1. **Heart Disease (Cleveland)**: Evaluates 13 features (chest pain type, cholesterol, resting blood pressure, age, sex, etc.) to predict binary presence of heart disease.
2. **Diabetes (Pima Indians)**: Evaluates 8 physiological measurements (Glucose, Blood Pressure, BMI, Insulin, etc.) to diagnose diabetes probability in female patients.
3. **Breast Cancer (Wisconsin Diagnostic)**: Analyzes 10 key physical characteristics of tissue cell nuclei (mean radius, perimeter, symmetry, concavity, etc.) to detect malignancy.

### Algorithms details:
- **Logistic Regression**: Serves as a baseline model; coefficients represent log-odds impact.
- **Support Vector Machine (SVM)**: Projects features into a higher-dimensional space using RBF kernel to draw non-linear boundary margins.
- **Random Forest**: Builds multiple decision trees and votes on predictions to reduce variance.
- **XGBoost**: Iteratively trains weak tree models to correct errors of predecessor models, optimizing a regularized loss function.
