import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml.features.engineer import FeatureEngineer
from backend.ml.explainability.explainer import ModelExplainer

def load_dataset():
    """Load the processed Kaggle dataset."""
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(fpr, tpr, auc, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#2a5298', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#dc3545', lw=2, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.3, color='#2a5298')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_distribution(y_pred_proba, y_test, save_path):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=pd.DataFrame({
        'Probability': y_pred_proba,
        'Class': ['PD' if y == 1 else 'No PD' for y in y_test]
    }), x='Probability', hue='Class', bins=20, alpha=0.6)
    plt.title('Distribution of Prediction Probabilities', fontsize=16, pad=20)
    plt.xlabel('Prediction Probability', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(feature_importance, feature_names, save_path):
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance Analysis', fontsize=16, pad=20)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create directories if they don't exist
    os.makedirs('model_visualizations', exist_ok=True)
    os.makedirs('model_explanations', exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset()
    
    # Initialize components
    feature_engineer = FeatureEngineer()
    
    # Preprocess data
    print("Preprocessing data...")
    X_train_processed = feature_engineer.preprocess_data(X_train, is_training=True)
    X_test_processed = feature_engineer.preprocess_data(X_test, is_training=False)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Initialize base model
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform grid search with cross-validation
    print("\nPerforming grid search with cross-validation...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    
    # Fit grid search
    grid_search.fit(X_train_processed, y_train)
    
    # Print best parameters and score
    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Get the best model
    model = grid_search.best_estimator_
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, 'model_visualizations/confusion_matrix.png')
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, 'model_visualizations/roc_curve.png')
    
    plot_prediction_distribution(y_pred_proba, y_test, 'model_visualizations/prediction_distribution.png')
    
    # Initialize model explainer
    print("\nInitializing model explainer...")
    explainer = ModelExplainer(model, X_train_processed.columns.tolist())
    
    # Generate explanations
    print("Generating explanations...")
    shap_explanation = explainer.get_shap_values(X_test_processed[:10])
    
    # Get feature importance from the model directly
    feature_importance = pd.DataFrame({
        'feature': X_train_processed.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    # Plot feature importance
    plot_feature_importance(feature_importance['importance'], feature_importance['feature'], 'model_visualizations/feature_importance.png')
    
    # Print predictions for first 10 patients
    print("\nPredictions for first 10 patients:")
    for i in range(10):
        prob = y_pred_proba[i]
        prediction = "PD" if y_pred[i] == 1 else "No PD"
        print(f"Patient {i+1}: {prediction} (Probability: {prob*100:.1f}%)")

if __name__ == "__main__":
    main() 