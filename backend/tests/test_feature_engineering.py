import pytest
import pandas as pd
import numpy as np
from ml.features.engineer import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'age': [65, 70, 75],
        'gender': [1, 0, 1],
        'bmi': [25.5, 26.0, 24.5],
        'disease_duration': [3, 5, 2],
        'updrs_score': [25, 30, 20],
        'jitter': [0.002, 0.003, 0.001],
        'shimmer': [0.015, 0.018, 0.012],
        'nhr': [0.02, 0.025, 0.015],
        'hnr': [20.5, 19.5, 21.0],
        'rpde': [0.4, 0.45, 0.35],
        'dfa': [0.7, 0.75, 0.65],
        'ppe': [0.15, 0.18, 0.12]
    })

@pytest.fixture
def feature_engineer():
    """Create a FeatureEngineer instance."""
    return FeatureEngineer(n_components=0.95)

def test_preprocess_data(feature_engineer, sample_data):
    """Test data preprocessing."""
    processed_data = feature_engineer.preprocess_data(sample_data)
    
    # Check if data is scaled
    assert processed_data.shape == sample_data.shape
    assert processed_data.columns.equals(sample_data.columns)
    
    # Check if data is approximately standardized (mean close to 0, std close to 1)
    assert abs(processed_data.mean().mean()) < 1e-10  # Mean should be close to 0
    assert abs(processed_data.std().mean() - 1) < 0.1  # Std should be approximately 1

def test_apply_pca(feature_engineer, sample_data):
    """Test PCA application."""
    processed_data = feature_engineer.preprocess_data(sample_data)
    pca_data, explained_variance = feature_engineer.apply_pca(processed_data)
    
    # Check if dimensionality is reduced
    assert pca_data.shape[1] <= processed_data.shape[1]
    
    # Check if explained variance ratio sum is close to n_components
    assert abs(explained_variance - 0.95) < 0.1

def test_select_features(feature_engineer, sample_data):
    """Test feature selection."""
    processed_data = feature_engineer.preprocess_data(sample_data)
    selected_data = feature_engineer.select_features(processed_data)
    
    # Check if selected features are fewer than or equal to original
    assert selected_data.shape[1] <= processed_data.shape[1]
    
    # Check if selected features are present in original data
    assert all(col in processed_data.columns for col in selected_data.columns)

def test_extract_medical_features(feature_engineer):
    """Test medical feature extraction."""
    # Create sample input data
    input_data = {
        'clinical_data': {
            'age': 65,
            'gender': 'M',
            'bmi': 25.5,
            'disease_duration': 3,
            'updrs_score': 25
        },
        'voice_data': {
            'jitter': 0.002,
            'shimmer': 0.015,
            'nhr': 0.02,
            'hnr': 20.5,
            'rpde': 0.4,
            'dfa': 0.7,
            'ppe': 0.15
        },
        'gait_data': {
            'stride_length': 1.2,
            'stride_time': 1.1,
            'step_length': 0.6,
            'step_time': 0.55,
            'cadence': 110
        }
    }
    
    # Extract features
    features = feature_engineer.extract_medical_features(input_data)
    
    # Check if output is a DataFrame
    assert isinstance(features, pd.DataFrame)
    
    # Check if all expected features are present
    expected_features = [
        'age', 'gender', 'bmi', 'disease_duration', 'updrs_score',
        'jitter', 'shimmer', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe',
        'stride_length', 'stride_time', 'step_length', 'step_time', 'cadence'
    ]
    assert all(feature in features.columns for feature in expected_features) 