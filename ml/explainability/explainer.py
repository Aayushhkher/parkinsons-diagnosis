import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from typing import Dict, Any, List, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained ML model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.lime_explainer = None

    def setup_shap(self, X: pd.DataFrame) -> None:
        """
        Set up SHAP explainer for the model.
        
        Args:
            X: Training data used to fit the SHAP explainer
        """
        logger.info("Setting up SHAP explainer...")
        
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X
            )
        else:
            self.explainer = shap.TreeExplainer(self.model)

    def setup_lime(self, X: pd.DataFrame) -> None:
        """
        Set up LIME explainer for the model.
        
        Args:
            X: Training data used to fit the LIME explainer
        """
        logger.info("Setting up LIME explainer...")
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=self.feature_names,
            class_names=['No PD', 'PD'],
            mode='classification'
        )

    def get_shap_values(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate SHAP values for the input data.
        
        Args:
            X: Input data to explain
            
        Returns:
            Dictionary containing SHAP values and feature importance
        """
        if self.explainer is None:
            self.setup_shap(X)
            
        logger.info("Calculating SHAP values...")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance
        }

    def get_lime_explanation(self, instance: pd.Series) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Single data instance to explain
            
        Returns:
            Dictionary containing LIME explanation
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call setup_lime first.")
            
        logger.info("Generating LIME explanation...")
        
        # Generate explanation
        exp = self.lime_explainer.explain_instance(
            instance.values,
            self.model.predict_proba
        )
        
        # Get feature importance
        feature_importance = pd.DataFrame(
            exp.as_list(),
            columns=['feature', 'importance']
        )
        
        return {
            'explanation': exp,
            'feature_importance': feature_importance
        }

    def plot_shap_summary(self, shap_values: np.ndarray, X: pd.DataFrame,
                         output_path: str = None) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values to plot
            X: Input data
            output_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False
        )
        
        if output_path:
            plt.savefig(output_path)
        plt.close()

    def plot_lime_explanation(self, exp: Any, output_path: str = None) -> None:
        """
        Plot LIME explanation.
        
        Args:
            exp: LIME explanation object
            output_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        
        if output_path:
            plt.savefig(output_path)
        plt.close()

    def generate_explanation_report(self, X: pd.DataFrame, y: pd.Series,
                                  output_dir: str = 'reports/') -> Dict[str, Any]:
        """
        Generate a comprehensive explanation report.
        
        Args:
            X: Input data
            y: Target labels
            output_dir: Directory to save report files
            
        Returns:
            Dictionary containing report data
        """
        logger.info("Generating explanation report...")
        
        # Get SHAP values
        shap_results = self.get_shap_values(X)
        
        # Generate LIME explanations for a few examples
        lime_explanations = {}
        for i in range(min(5, len(X))):
            lime_results = self.get_lime_explanation(X.iloc[i])
            lime_explanations[f'instance_{i}'] = lime_results
            
        # Create plots
        self.plot_shap_summary(
            shap_results['shap_values'],
            X,
            f"{output_dir}/shap_summary.png"
        )
        
        for i, exp in enumerate(lime_explanations.values()):
            self.plot_lime_explanation(
                exp['explanation'],
                f"{output_dir}/lime_explanation_{i}.png"
            )
            
        return {
            'shap': shap_results,
            'lime': lime_explanations
        } 