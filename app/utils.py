import os
import subprocess
import config

def execute_pipeline():
    """
    Programmatically runs the pipeline training script from Streamlit.
    """
    script_path = os.path.join(config.ROOT_DIR, "src", "evaluation", "train.py")
    try:
        # Run python script as a subprocess
        result = subprocess.run(
            ["python", script_path], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return True, "Pipeline execution completed successfully!\n" + result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Pipeline execution failed:\nStderr: {e.stderr}\nStdout: {e.stdout}"
    except Exception as ex:
        return False, f"Unexpected error executing pipeline: {str(ex)}"

def check_plots_exist():
    """Checks if the required visual figures have been generated."""
    plots = [
        "confusion_matrix.png",
        "roc_curves.png",
        "feature_importance.png",
        "eda_plots.png"
    ]
    return all(os.path.exists(os.path.join(config.FIGURES_DIR, p)) for p in plots)
