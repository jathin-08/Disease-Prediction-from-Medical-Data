import os
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from .data_loader import get_dataset_metadata
from .model_manager import train_and_evaluate, make_prediction, get_eda_data

app = FastAPI(title="Disease Prediction API", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for frontend files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

class TrainRequest(BaseModel):
    dataset_name: str
    algorithm: str

class PredictRequest(BaseModel):
    dataset_name: str
    algorithm: str
    inputs: Dict[str, Any]

# API Endpoints
@app.get("/api/datasets")
def list_datasets():
    """Returns the names and summaries of all available datasets."""
    datasets = ["heart_disease", "diabetes", "breast_cancer"]
    results = {}
    for ds in datasets:
        meta = get_dataset_metadata(ds)
        if meta:
            results[ds] = {
                "name": meta["name"],
                "description": meta["description"],
                "features_count": len(meta["features"])
            }
    return results

@app.get("/api/datasets/{dataset_name}/metadata")
def get_metadata(dataset_name: str):
    """Retrieves field schemas, data types, and default values for a dataset."""
    meta = get_dataset_metadata(dataset_name)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return meta

@app.post("/api/train")
def train_model(payload: TrainRequest):
    """Trains a model using a specific dataset and algorithm, returning test evaluation metrics."""
    try:
        metrics = train_and_evaluate(payload.dataset_name, payload.algorithm)
        return metrics
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/api/predict")
def predict(payload: PredictRequest):
    """Predicts disease outcome based on user-supplied features using the specified model."""
    try:
        prediction = make_prediction(payload.dataset_name, payload.algorithm, payload.inputs)
        return prediction
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/datasets/{dataset_name}/eda")
def get_eda(dataset_name: str):
    """Retrieves dataset distribution and correlation summary stats for plotting."""
    try:
        eda_stats = get_eda_data(dataset_name)
        return eda_stats
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading EDA details: {str(e)}")

# Frontend Static Routing (without requiring aiofiles)
@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("Frontend files not created yet.", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/style.css")
def read_style():
    style_path = os.path.join(FRONTEND_DIR, "style.css")
    if not os.path.exists(style_path):
        return Response(status_code=404)
    with open(style_path, "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/css")

@app.get("/app.js")
def read_app_js():
    js_path = os.path.join(FRONTEND_DIR, "app.js")
    if not os.path.exists(js_path):
        return Response(status_code=404)
    with open(js_path, "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="application/javascript")
