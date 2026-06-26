import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import config

# Set plot style for professional aesthetics
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150
})

def plot_confusion_matrix(cm_dict, save_path=None):
    """
    Plots and saves confusion matrix heatmap.
    cm_dict: {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    """
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]
    
    cm_arr = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"],
        ax=ax
    )
    ax.set_title("Confusion Matrix Table")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_curves(roc_data, save_path=None):
    """
    Plots and saves ROC curves for comparison.
    roc_data: dict of model_name -> (y_test, y_prob)
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for (model_name, (y_test, y_prob)), color in zip(roc_data.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr, 
            tpr, 
            label=f"{model_name} (AUC = {roc_auc:.3f})", 
            color=color, 
            linewidth=2
        )
        
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(importances_dict, save_path=None):
    """
    Plots and saves a comparison of feature importances.
    importances_dict: dict of feature_name -> importance_score
    """
    # Sort features by importance
    sorted_features = sorted(
        importances_dict.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10] # Top 10 features
    
    features = [x[0] for x in sorted_features]
    scores = [x[1] for x in sorted_features]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=scores, y=features, hue=features, palette="viridis", legend=False, ax=ax)
    ax.set_xlabel("Relative Importance Score")
    ax.set_ylabel("Biometric Feature")
    ax.set_title("Top 10 Feature Importances Breakdown")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_eda_plots(df, dataset_name, save_path=None):
    """
    Plots target class distributions and correlations.
    """
    metadata = config.DATASET_METADATA[dataset_name]
    target_col = metadata["target_col"]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Class balance plot
    sns.countplot(
        x=target_col, 
        data=df, 
        hue=target_col,
        palette=["#2ca02c", "#d62728"], 
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title(f"Class Balance: {metadata['name']}")
    axes[0].set_xlabel("Diagnostic Outcome")
    axes[0].set_ylabel("Patient Count")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Negative", "Positive"])
    
    # Top correlations
    correlations = df.corr()[target_col].drop(target_col).fillna(0).sort_values(ascending=False)
    # top 6 and bottom 6
    top_corr = pd.concat([correlations.head(5), correlations.tail(5)]).drop_duplicates()
    
    sns.barplot(
        x=top_corr.values, 
        y=top_corr.index, 
        hue=top_corr.index,
        palette="coolwarm", 
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title("Correlation with Diagnostic Outcome")
    axes[1].set_xlabel("Pearson Correlation Coefficient")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
