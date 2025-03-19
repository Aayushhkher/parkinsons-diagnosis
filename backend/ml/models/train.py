import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple, List

class ModelTrainer:
    def __init__(self, model_type: str = 'ensemble'):
        """Initialize the model trainer."""
        self.model_type = model_type
        self.models = {}
        self.feature_importance = None
        self.training_history = None
        self.rf_model = None
        self.metrics = {}
        self.feature_names = []
    
    def train_svm(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train SVM model."""
        model = SVC(kernel='rbf', probability=True)
        model.fit(X, y)
        self.models['svm'] = model
        
        # Get predictions and metrics
        y_pred = model.predict(X)
        metrics = self.calculate_metrics(y, y_pred)
        
        return {
            'model': model,
            'metrics': metrics
        }
    
    def train_rf(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model."""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        self.models['rf'] = model
        
        # Get predictions and metrics
        y_pred = model.predict(X)
        metrics = self.calculate_metrics(y, y_pred)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': self.feature_importance.to_dict('records')
        }
    
    def train_nn(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train Neural Network model."""
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42
        )
        model.fit(X, y)
        self.models['nn'] = model
        
        # Get predictions and metrics
        y_pred = model.predict(X)
        metrics = self.calculate_metrics(y, y_pred)
        
        # Store training history
        self.training_history = {
            'loss': model.loss_,
            'n_iter': model.n_iter_,
            'best_loss': model.best_loss_
        }
        
        return {
            'model': model,
            'metrics': metrics,
            'training_history': self.training_history
        }
    
    def cross_validate(self, X: pd.DataFrame, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Perform cross-validation for all models."""
        results = {}
        
        # SVM cross-validation
        svm_scores = cross_val_score(
            SVC(kernel='rbf', probability=True),
            X, y, cv=cv, scoring='accuracy'
        )
        results['svm'] = {
            'mean_score': svm_scores.mean(),
            'std_score': svm_scores.std()
        }
        
        # Random Forest cross-validation
        rf_scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X, y, cv=cv, scoring='accuracy'
        )
        results['rf'] = {
            'mean_score': rf_scores.mean(),
            'std_score': rf_scores.std()
        }
        
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate various performance metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """Train the model on the given data."""
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Train Random Forest model
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        
        # Make predictions
        y_pred = self.rf_model.predict(X)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.rf_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.rf_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.rf_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.rf_model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance from the Random Forest model."""
        if hasattr(self, 'rf_model') and self.rf_model is not None:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        return pd.DataFrame({'feature': [], 'importance': []})
    
    def get_model_metrics(self):
        """Get the current model metrics."""
        return {
            'accuracy': self.metrics.get('accuracy', 0.0),
            'precision': self.metrics.get('precision', 0.0),
            'recall': self.metrics.get('recall', 0.0),
            'f1': self.metrics.get('f1', 0.0)
        }

    def make_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions using all trained models."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)
        return predictions 