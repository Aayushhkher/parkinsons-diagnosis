import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from ml.models.train import ModelTrainer

@pytest.fixture
def sample_data():
    """Create synthetic data for testing."""
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
    """Create a ModelTrainer instance."""
    return ModelTrainer(model_type='ensemble')

def test_train_svm(model_trainer, sample_data):
    """Test SVM training."""
    X, y = sample_data
    result = model_trainer.train_svm(X, y)
    
    # Check if model is trained
    assert 'model' in result
    assert 'metrics' in result
    
    # Check if metrics are valid
    metrics = result['metrics']
    assert all(0 <= value <= 1 for value in metrics.values())

def test_train_rf(model_trainer, sample_data):
    """Test Random Forest training."""
    X, y = sample_data
    result = model_trainer.train_rf(X, y)
    
    # Check if model is trained
    assert 'model' in result
    assert 'metrics' in result
    assert 'feature_importance' in result
    
    # Check if feature importance is valid
    feature_importance = result['feature_importance']
    assert len(feature_importance) > 0
    assert all(0 <= item['importance'] <= 1 for item in feature_importance)

def test_train_nn(model_trainer, sample_data):
    """Test Neural Network training."""
    X, y = sample_data
    result = model_trainer.train_nn(X, y)
    
    # Check if model is trained
    assert 'model' in result
    assert 'metrics' in result
    assert 'training_history' in result
    
    # Check if training history is valid
    history = result['training_history']
    assert 'loss' in history
    assert 'n_iter' in history
    assert 'best_loss' in history

def test_cross_validate(model_trainer, sample_data):
    """Test cross-validation."""
    X, y = sample_data
    results = model_trainer.cross_validate(X, y)
    
    # Check if results contain SVM and RF scores
    assert 'svm' in results
    assert 'rf' in results
    
    # Check if scores are valid
    for model in ['svm', 'rf']:
        assert 'mean_score' in results[model]
        assert 'std_score' in results[model]
        assert 0 <= results[model]['mean_score'] <= 1
        assert results[model]['std_score'] >= 0

def test_calculate_metrics(model_trainer):
    """Test metric calculation."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1])
    
    metrics = model_trainer.calculate_metrics(y_true, y_pred)
    
    # Check if all metrics are calculated
    assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1'])
    
    # Check if metrics are valid
    assert all(0 <= value <= 1 for value in metrics.values()) 