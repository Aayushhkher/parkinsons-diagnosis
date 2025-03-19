import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, output_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(feature_importance, output_path):
    """Plot feature importance."""
    # Convert feature importance to DataFrame if it's not already
    if isinstance(feature_importance, list):
        feature_importance = pd.DataFrame(feature_importance)
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_prediction_distribution(predictions, probabilities, output_path):
    """Plot distribution of predictions and probabilities."""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Prediction Distribution
    plt.subplot(1, 2, 1)
    unique, counts = np.unique(predictions, return_counts=True)
    plt.bar(unique, counts)
    plt.title('Distribution of Predictions')
    plt.xlabel('Prediction (0: No PD, 1: PD)')
    plt.ylabel('Count')
    
    # Plot 2: Probability Distribution
    plt.subplot(1, 2, 2)
    plt.hist(probabilities, bins=20)
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Probability of PD')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Create output directory
    output_dir = "model_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model results
    from run_model import main as run_model
    results = run_model()
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['y_true'],
        results['predictions'],
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curve
    plot_roc_curve(
        results['y_true'],
        results['probabilities'],
        os.path.join(output_dir, 'roc_curve.png')
    )
    
    # Plot feature importance
    plot_feature_importance(
        results['feature_importance'],
        os.path.join(output_dir, 'feature_importance.png')
    )
    
    # Plot prediction distribution
    plot_prediction_distribution(
        results['predictions'],
        results['probabilities'],
        os.path.join(output_dir, 'prediction_distribution.png')
    )
    
    print(f"Visualizations have been saved in the {output_dir}/ directory")

if __name__ == "__main__":
    main() 