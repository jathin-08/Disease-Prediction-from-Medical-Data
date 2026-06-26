import sys
import os
# Inject project root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import pandas as pd
import numpy as np
import config

from src.data_loader import get_raw_dataset
from src.preprocessor import preprocess_and_split_all, load_processed_data
from src.models.svm_model import SVMModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.evaluation.metrics import calculate_classification_metrics
from src.evaluation.visualize import (
    plot_confusion_matrix, 
    plot_roc_curves, 
    plot_feature_importance, 
    plot_eda_plots
)

def train_and_save_pipeline():
    """
    Main orchestration function to train models, evaluate performance, 
    save artifacts and figures, and generate reports.
    """
    # 1. Split and scale raw datasets
    preprocess_and_split_all()
    
    # Track overall best model details
    best_overall_accuracy = 0.0
    best_overall_model = None
    best_overall_name = ""
    
    datasets = ["heart_disease", "diabetes", "breast_cancer"]
    algorithms = {
        "svm": SVMModel,
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel
    }

    # Store performance history
    pipeline_results = {}

    for ds in datasets:
        print(f"\n==================================================")
        print(f"Training models for dataset: {ds}")
        print(f"==================================================")
        
        # Load preprocessed splits
        X_train, X_test, y_train, y_test = load_processed_data(ds)
        
        # Plot and save EDA charts
        raw_df = get_raw_dataset(ds)
        eda_plot_path = os.path.join(config.FIGURES_DIR, f"eda_{ds}_plots.png")
        plot_eda_plots(raw_df, ds, save_path=eda_plot_path)
        # Also copy heart_disease eda as default eda_plots.png
        if ds == "heart_disease":
            plot_eda_plots(raw_df, ds, save_path=os.path.join(config.FIGURES_DIR, "eda_plots.png"))
            
        roc_data = {}
        ds_results = {}
        
        for algo_name, model_class in algorithms.items():
            print(f"Training {algo_name}...")
            
            # Train model
            model = model_class()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Evaluate metrics
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
            print(f"-> Test Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {metrics['auc']:.4f}")
            
            ds_results[algo_name] = {
                "model": model,
                "metrics": metrics,
                "y_prob": y_prob
            }
            
            # ROC curve plotting data
            roc_data[algo_name] = (y_test, y_prob)
            
            # Save specific models mapping requested in layout
            if ds == "heart_disease" and algo_name == "svm":
                save_model(model, "svm_heart.pkl")
            elif ds == "diabetes" and algo_name == "random_forest":
                save_model(model, "rf_diabetes.pkl")
            elif ds == "breast_cancer" and algo_name == "xgboost":
                save_model(model, "xgb_cancer.pkl")
                
            # Track overall best model
            if metrics["accuracy"] > best_overall_accuracy:
                best_overall_accuracy = metrics["accuracy"]
                best_overall_model = model
                best_overall_name = f"{algo_name}_{ds}"

        # Save results for reporting
        pipeline_results[ds] = ds_results
        
        # Plot ROC comparison curves for the dataset
        roc_plot_path = os.path.join(config.FIGURES_DIR, f"roc_curves_{ds}.png")
        plot_roc_curves(roc_data, save_path=roc_plot_path)
        if ds == "breast_cancer":
            # Set default roc_curves.png as breast cancer ROC comparison
            plot_roc_curves(roc_data, save_path=os.path.join(config.FIGURES_DIR, "roc_curves.png"))

        # Save feature importances and confusion matrices for the key models
        if ds == "heart_disease":
            # Feature correlations or coefficients
            svm_metrics = ds_results["svm"]["metrics"]
            plot_confusion_matrix(svm_metrics["confusion_matrix"], os.path.join(config.FIGURES_DIR, "confusion_matrix_heart.png"))
            
        elif ds == "diabetes":
            rf_model = ds_results["random_forest"]["model"]
            rf_metrics = ds_results["random_forest"]["metrics"]
            # Save default confusion matrix
            plot_confusion_matrix(rf_metrics["confusion_matrix"], os.path.join(config.FIGURES_DIR, "confusion_matrix.png"))
            
            # Extract feature importance
            importances = dict(zip(X_train.columns, rf_model.feature_importances_))
            plot_feature_importance(importances, os.path.join(config.FIGURES_DIR, "feature_importance_diabetes.png"))
            
        elif ds == "breast_cancer":
            xgb_model = ds_results["xgboost"]["model"]
            xgb_metrics = ds_results["xgboost"]["metrics"]
            
            importances = dict(zip(X_train.columns, xgb_model.feature_importances_))
            # Save default feature importance and confusion matrix
            plot_feature_importance(importances, os.path.join(config.FIGURES_DIR, "feature_importance.png"))
            plot_confusion_matrix(xgb_metrics["confusion_matrix"], os.path.join(config.FIGURES_DIR, "confusion_matrix_cancer.png"))

    # Save overall best model to best_model.pkl
    if best_overall_model is not None:
        print(f"\nSaving best overall model ({best_overall_name}) with accuracy {best_overall_accuracy:.4f}")
        save_model(best_overall_model, "best_model.pkl")

    # Generate final report files
    generate_markdown_report(pipeline_results)
    generate_pdf_report(pipeline_results)

def save_model(model, filename):
    """Helper to pickle models to saved_models/."""
    path = os.path.join(config.SAVED_MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Serialized model saved to {path}")

def generate_markdown_report(results):
    """Generates a text report summarizing validation results."""
    report_path = os.path.join(config.REPORTS_DIR, "final_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AuraMed: Pipeline Training & Validation Report\n\n")
        f.write("This report summarizes evaluation results for classification algorithms on diagnostic datasets.\n\n")
        
        for ds, ds_res in results.items():
            f.write(f"## Dataset: {config.DATASET_METADATA[ds]['name']}\n\n")
            f.write("| Algorithm | Accuracy | Precision | Recall | F1-Score | AUC |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for algo, details in ds_res.items():
                m = details["metrics"]
                auc_val = f"{m['auc']:.4f}" if m['auc'] is not None else "N/A"
                f.write(f"| {algo.upper()} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {auc_val} |\n")
            f.write("\n")
            
    print(f"Generated text report saved to {report_path}")

def generate_pdf_report(results):
    """
    Attempts to generate final_report.pdf using reportlab.
    Falls back to a warning/text if reportlab is not present.
    """
    pdf_path = os.path.join(config.REPORTS_DIR, "final_report.pdf")
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Define styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#002b49'),
            spaceAfter=20
        )
        subtitle_style = ParagraphStyle(
            'SubTitleStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.gray,
            spaceAfter=30
        )
        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#005b96'),
            spaceBefore=15,
            spaceAfter=10
        )
        
        story.append(Paragraph("AuraMed: Pipeline Training & Validation Report", title_style))
        story.append(Paragraph("Automated report summarizing diagnostic metrics evaluated on holdout datasets.", subtitle_style))
        story.append(Spacer(1, 10))
        
        for ds, ds_res in results.items():
            story.append(Paragraph(f"Dataset: {config.DATASET_METADATA[ds]['name']}", heading_style))
            
            # Build metrics table
            data = [["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]]
            for algo, details in ds_res.items():
                m = details["metrics"]
                auc_val = f"{m['auc']:.4f}" if m['auc'] is not None else "N/A"
                data.append([
                    algo.upper().replace("_", " "),
                    f"{m['accuracy']:.4f}",
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1_score']:.4f}",
                    auc_val
                ])
                
            t = Table(data, colWidths=[120, 70, 70, 70, 70, 60])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#005b96')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f7f9fa')),
                ('GRID', (0,0), (-1,-1), 1, colors.lightgrey),
                ('FONTSIZE', (0,0), (-1,-1), 9),
            ]))
            story.append(t)
            story.append(Spacer(1, 15))
            
        doc.build(story)
        print(f"Generated PDF report saved to {pdf_path}")
        
    except ImportError:
        # reportlab not installed yet, write a warning and a text fallback
        print("reportlab package not available. Compiling simplified PDF binary fallback...")
        # Write a dummy/simple valid PDF file structure manually so that a pdf is present
        with open(pdf_path, "w", encoding="latin1") as f:
            f.write("%PDF-1.4\n")
            f.write("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
            f.write("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
            f.write("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")
            f.write("4 0 obj\n<< /Length 150 >>\nstream\nBT\n/F1 18 Tf\n70 700 Td\n(AuraMed: Pipeline Training & Validation Report) Tj\n/F1 12 Tf\n0 -30 Td\n(Model validation was completed successfully.) Tj\n0 -20 Td\n(For full results table, please read final_report.md) Tj\nET\nendstream\nendobj\n")
            f.write("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
            f.write("xref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000111 00000 n\n0000000250 00000 n\n0000000451 00000 n\n")
            f.write("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n524\n%%EOF\n")

if __name__ == "__main__":
    train_and_save_pipeline()
