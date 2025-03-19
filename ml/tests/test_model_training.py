import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from models.train import ModelTrainer

@pytest.fixture
def sample_data():
    # Generate synthetic data for testing
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)

@pytest.fixture
def model_trainer():
    return ModelTrainer(model_type='ensemble')

def test_train_svm(model_trainer, sample_data):
    X, y = sample_data
    results = model_trainer._train_svm(X, X, y, y)
    
    # Check if model was trained
    assert 'model' in results
    assert 'metrics' in results
    
    # Check metrics
    metrics = results['metrics']
    assert all(0 <= metrics[metric] <= 1 for metric in ['accuracy', 'precision', 'recall', 'f1'])

def test_train_rf(model_trainer, sample_data):
    X, y = sample_data
    results = model_trainer._train_rf(X, X, y, y)
    
    # Check if model was trained
    assert 'model' in results
    assert 'metrics' in results
    assert 'feature_importance' in results
    
    # Check metrics
    metrics = results['metrics']
    assert all(0 <= metrics[metric] <= 1 for metric in ['accuracy', 'precision', 'recall', 'f1'])
    
    # Check feature importance
    importance = results['feature_importance']
    assert all(0 <= imp <= 1 for imp in importance.values())
    assert abs(sum(importance.values()) - 1) < 1e-10

def test_train_nn(model_trainer, sample_data):
    X, y = sample_data
    results = model_trainer._train_nn(X, X, y, y)
    
    # Check if model was trained
    assert 'model' in results
    assert 'metrics' in results
    assert 'history' in results
    
    # Check metrics
    metrics = results['metrics']
    assert all(0 <= metrics[metric] <= 1 for metric in ['accuracy', 'precision', 'recall', 'f1'])
    
    # Check training history
    history = results['history']
    assert 'loss' in history
    assert 'accuracy' in history
    assert len(history['loss']) > 0
    assert len(history['accuracy']) > 0

def test_cross_validate(model_trainer, sample_data):
    X, y = sample_data
    
    # Train models first
    model_trainer.train(X, y)
    
    # Perform cross-validation
    cv_scores = model_trainer.cross_validate(X, y, cv=5)
    
    # Check cross-validation scores
    assert 'svm' in cv_scores
    assert 'rf' in cv_scores
    assert all(0 <= score <= 1 for scores in cv_scores.values() for score in scores)
    assert all(len(scores) == 5 for scores in cv_scores.values())

def test_calculate_metrics(model_trainer):
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1])
    
    metrics = model_trainer._calculate_metrics(y_true, y_pred)
    
    # Check if all metrics are calculated
    assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1'])
    assert all(0 <= metrics[metric] <= 1 for metric in metrics) 