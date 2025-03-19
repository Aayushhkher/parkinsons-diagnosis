import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import os

class ModelExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.lime_explainer = None
        
    def setup_shap(self, X: pd.DataFrame):
        """Initialize SHAP explainer."""
        if isinstance(self.model, (shap.TreeExplainer, shap.KernelExplainer)):
            self.explainer = self.model
        else:
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X)
            
    def setup_lime(self, X: pd.DataFrame):
        """Initialize LIME explainer."""
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=self.feature_names,
            class_names=['No PD', 'PD'],
            mode='classification'
        )
        
    def get_shap_values(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Calculate SHAP values for the input data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        if self.explainer is None:
            self.setup_shap(X)
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle binary classification case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (PD) for feature importance
            
        # Calculate mean absolute SHAP values for feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap.tolist()
        }).sort_values('importance', ascending=False)
        
        return {
            'shap_values': shap_values.tolist(),
            'feature_importance': feature_importance
        }
        
    def get_lime_explanation(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Generate LIME explanation for a single instance."""
        if self.lime_explainer is None:
            self.setup_lime(instance)
            
        # Generate explanation
        exp = self.lime_explainer.explain_instance(
            instance.values[0],
            self.model.predict_proba
        )
        
        # Get explanation as list of tuples (feature, weight)
        explanation = exp.as_list()
        
        return {
            'explanation': explanation,
            'prediction': exp.predict_proba[1],  # Probability of class 1 (PD)
            'intercept': exp.intercept[1]  # Intercept for class 1 (PD)
        }
        
    def plot_shap_summary(self, X: pd.DataFrame, output_path: str):
        """Generate and save SHAP summary plot."""
        shap_results = self.get_shap_values(X)
        shap_values = np.array(shap_results['shap_values'])
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_lime_explanation(self, instance: pd.DataFrame, output_path: str):
        """Generate and save LIME explanation plot."""
        explanation = self.get_lime_explanation(instance)
        
        plt.figure(figsize=(10, 6))
        exp = self.lime_explainer.explain_instance(
            instance.values[0],
            self.model.predict_proba
        )
        exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def generate_explanation_report(self, X: pd.DataFrame, y: np.ndarray, output_dir: str) -> Dict[str, Any]:
        """Generate a complete explanation report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get SHAP values and feature importance
        shap_results = self.get_shap_values(X)
        
        # Generate SHAP summary plot
        shap_plot_path = os.path.join(output_dir, 'shap_summary.png')
        self.plot_shap_summary(X, shap_plot_path)
        
        # Generate LIME explanation for a sample instance
        sample_instance = X.iloc[0:1]
        lime_plot_path = os.path.join(output_dir, 'lime_explanation.png')
        self.plot_lime_explanation(sample_instance, lime_plot_path)
        
        return {
            'feature_importance': shap_results['feature_importance'].to_dict('records'),
            'shap_values': shap_results['shap_values'],
            'lime_explanation': self.get_lime_explanation(sample_instance),
            'plots': {
                'shap_summary': shap_plot_path,
                'lime_explanation': lime_plot_path
            }
        } 