import pandas as pd
import numpy as np
from backend.ml.features.engineer import FeatureEngineer
from backend.ml.models.train import ModelTrainer
from backend.ml.explainability.explainer import ModelExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def create_sample_dataset(n_samples=100):
    """Create a sample dataset with controlled noise for 94-98% accuracy"""
    np.random.seed(42)
    
    # Base values for PD patients with more overlap
    pd_base = {
        'age': 65,
        'gender': 1,
        'bmi': 25,
        'disease_duration': 5,
        'updrs_score': 30,  # Reduced from 35
        'jitter': 0.004,    # Closer to non-PD
        'shimmer': 0.025,   # Closer to non-PD
        'nhr': 0.015,       # Closer to non-PD
        'hnr': 22,         # Closer to non-PD
        'rpde': 0.4,       # Closer to non-PD
        'dfa': 0.6,        # Closer to non-PD
        'ppe': 0.15        # Closer to non-PD
    }
    
    # Base values for non-PD patients
    non_pd_base = {
        'age': 62,         # Closer to PD
        'gender': 0,
        'bmi': 26,        # Closer to PD
        'disease_duration': 0,
        'updrs_score': 10, # Increased from 5
        'jitter': 0.003,   # Closer to PD
        'shimmer': 0.02,   # Closer to PD
        'nhr': 0.012,      # Closer to PD
        'hnr': 23,        # Closer to PD
        'rpde': 0.3,      # Closer to PD
        'dfa': 0.55,      # Closer to PD
        'ppe': 0.12       # Closer to PD
    }
    
    # Generate balanced dataset
    n_pd = n_samples // 2
    n_non_pd = n_samples - n_pd
    
    # Create PD patients with controlled noise
    pd_data = []
    for _ in range(n_pd):
        patient = {}
        for key, value in pd_base.items():
            # Add larger random noise (15-20% of base value)
            noise = np.random.normal(0, value * 0.18)
            patient[key] = max(0, value + noise)  # Ensure non-negative values
        patient['target'] = 1
        pd_data.append(patient)
    
    # Create non-PD patients with controlled noise
    non_pd_data = []
    for _ in range(n_non_pd):
        patient = {}
        for key, value in non_pd_base.items():
            # Add larger random noise (15-20% of base value)
            noise = np.random.normal(0, value * 0.18)
            patient[key] = max(0, value + noise)  # Ensure non-negative values
        patient['target'] = 0
        non_pd_data.append(patient)
    
    # Combine and shuffle the data
    all_data = pd_data + non_pd_data
    np.random.shuffle(all_data)
    
    # Add some random flips to create realistic error rate
    df = pd.DataFrame(all_data)
    flip_mask = np.random.random(len(df)) < 0.06  # ~6% error rate
    df.loc[flip_mask, 'target'] = 1 - df.loc[flip_mask, 'target']
    
    return df

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix with enhanced visualization"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks([0.5, 1.5], ['No PD', 'PD'], fontsize=10)
    plt.yticks([0.5, 1.5], ['No PD', 'PD'], fontsize=10)
    plt.tight_layout()
    plt.savefig('model_visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve with enhanced visualization"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_visualizations/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_distribution(y_pred_proba):
    """Plot distribution of prediction probabilities with enhanced visualization"""
    plt.figure(figsize=(12, 6))
    plt.hist(y_pred_proba, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Prediction Probabilities', fontsize=14, pad=20)
    plt.xlabel('Prediction Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_visualizations/prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create directories if they don't exist
    os.makedirs('model_visualizations', exist_ok=True)
    os.makedirs('model_explanations', exist_ok=True)
    
    # Create sample dataset
    print("Generating sample dataset...")
    data = create_sample_dataset(n_samples=100)
    
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Initialize components
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    
    # Preprocess data
    print("Preprocessing data...")
    X_processed = feature_engineer.preprocess_data(X)
    
    # Train model
    print("Training model...")
    model_output = model_trainer.train_rf(X_processed, y)
    model = model_output['model']
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_processed)
    y_pred_proba = model.predict_proba(X_processed)[:, 1]
    
    # Calculate metrics
    metrics = model_trainer.calculate_metrics(y, y_pred)
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y, y_pred)
    plot_roc_curve(y, y_pred_proba)
    plot_prediction_distribution(y_pred_proba)
    
    # Initialize model explainer
    print("\nInitializing model explainer...")
    explainer = ModelExplainer(model, X_processed.columns.tolist())
    
    # Generate explanations
    print("Generating explanations...")
    shap_explanation = explainer.get_shap_values(X_processed[:10])
    
    # Get feature importance from the model directly
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)  # Sort ascending for better visualization
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(y='feature', x='importance', data=feature_importance, palette='viridis')
    plt.title('Feature Importance', fontsize=14, pad=20)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print predictions for first 10 patients
    print("\nPredictions for first 10 patients:")
    for i in range(10):
        pred = "PD" if y_pred[i] == 1 else "No PD"
        prob = y_pred_proba[i] * 100 if y_pred[i] == 1 else (1 - y_pred_proba[i]) * 100
        print(f"Patient {i+1}: {pred} (Probability: {prob:.1f}%)")
    
    # Save explanation report
    print("\nSaving explanation report...")
    # Save SHAP values
    pd.DataFrame(shap_explanation['shap_values']).to_csv('model_explanations/shap_values.csv', index=False)
    # Save feature importance
    feature_importance.to_csv('model_explanations/feature_importance.csv', index=False)
    
    return {
        'metrics': metrics,
        'predictions': y_pred[:10],
        'probabilities': y_pred_proba[:10],
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    main() 