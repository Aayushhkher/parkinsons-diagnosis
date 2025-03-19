import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple, List
import tensorflow as tf
from tensorflow.keras import layers, models
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train ('svm', 'rf', 'nn', or 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.feature_importance = {}
        self.model_path = 'ml/models/saved/'
        os.makedirs(self.model_path, exist_ok=True)

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the selected model(s) on the input data.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary containing training results and model metrics
        """
        logger.info(f"Starting model training with {self.model_type}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        if self.model_type in ['svm', 'ensemble']:
            svm_results = self._train_svm(X_train, X_test, y_train, y_test)
            results['svm'] = svm_results
            
        if self.model_type in ['rf', 'ensemble']:
            rf_results = self._train_rf(X_train, X_test, y_train, y_test)
            results['rf'] = rf_results
            
        if self.model_type in ['nn', 'ensemble']:
            nn_results = self._train_nn(X_train, X_test, y_train, y_test)
            results['nn'] = nn_results
            
        return results

    def _train_svm(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train and evaluate SVM model."""
        logger.info("Training SVM model...")
        
        model = SVC(kernel='rbf', probability=True)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Save model
        joblib.dump(model, os.path.join(self.model_path, 'svm_model.joblib'))
        
        return {
            'metrics': metrics,
            'model': model
        }

    def _train_rf(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train and evaluate Random Forest model."""
        logger.info("Training Random Forest model...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance['rf'] = dict(zip(
            X_train.columns,
            model.feature_importances_
        ))
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Save model
        joblib.dump(model, os.path.join(self.model_path, 'rf_model.joblib'))
        
        return {
            'metrics': metrics,
            'model': model,
            'feature_importance': self.feature_importance['rf']
        }

    def _train_nn(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train and evaluate Neural Network model."""
        logger.info("Training Neural Network model...")
        
        # Create model
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Save model
        model.save(os.path.join(self.model_path, 'nn_model.h5'))
        
        return {
            'metrics': metrics,
            'model': model,
            'history': history.history
        }

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the trained models.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation scores for each model
        """
        logger.info("Performing cross-validation...")
        
        cv_scores = {}
        
        if 'svm' in self.models:
            svm_scores = cross_val_score(self.models['svm'], X, y, cv=cv)
            cv_scores['svm'] = svm_scores.tolist()
            
        if 'rf' in self.models:
            rf_scores = cross_val_score(self.models['rf'], X, y, cv=cv)
            cv_scores['rf'] = rf_scores.tolist()
            
        return cv_scores 