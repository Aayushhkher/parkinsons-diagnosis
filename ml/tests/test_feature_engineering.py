import pytest
import pandas as pd
import numpy as np
from features.engineer import FeatureEngineer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [65, 70, 75],
        'gender': ['M', 'F', 'M'],
        'duration': [5, 8, 3],
        'motor_UPDRS': [25, 30, 20],
        'total_UPDRS': [35, 40, 30],
        'tremor': [2, 3, 1],
        'rigidity': [3, 4, 2],
        'bradykinesia': [4, 5, 3]
    })

@pytest.fixture
def feature_engineer():
    return FeatureEngineer(n_components=0.95)

def test_preprocess_data(feature_engineer, sample_data):
    processed_data = feature_engineer.preprocess_data(sample_data)
    
    # Check if data is scaled
    assert processed_data.mean().mean() < 1e-10  # Mean should be close to 0
    assert abs(processed_data.std().mean() - 1) < 1e-10  # Std should be close to 1
    
    # Check if all columns are preserved
    assert all(col in processed_data.columns for col in sample_data.columns)

def test_apply_pca(feature_engineer, sample_data):
    processed_data = feature_engineer.preprocess_data(sample_data)
    pca_data, explained_variance = feature_engineer.apply_pca(processed_data)
    
    # Check if PCA reduced dimensionality
    assert pca_data.shape[1] <= processed_data.shape[1]
    
    # Check if explained variance ratios sum to less than 1
    assert sum(explained_variance.values()) <= 1

def test_select_features(feature_engineer, sample_data):
    processed_data = feature_engineer.preprocess_data(sample_data)
    selected_data = feature_engineer.select_features(processed_data, threshold=0.01)
    
    # Check if features were selected
    assert selected_data.shape[1] <= processed_data.shape[1]
    
    # Check if selected features are in original data
    assert all(col in processed_data.columns for col in selected_data.columns)

def test_extract_medical_features(feature_engineer):
    test_data = {
        'clinical_data': {
            'age': 65,
            'gender': 'M',
            'duration': 5,
            'motor_UPDRS': 25,
            'total_UPDRS': 35,
            'tremor': 2,
            'rigidity': 3,
            'bradykinesia': 4
        },
        'imaging_data': {
            'mri_features': [0.1, 0.2, 0.3],
            'pet_features': [0.4, 0.5, 0.6]
        },
        'voice_data': {
            'jitter': 0.01,
            'shimmer': 0.02,
            'nhr': 0.03,
            'hnr': 0.04,
            'rpde': 0.05,
            'dfa': 0.06,
            'spread1': 0.07,
            'spread2': 0.08,
            'ppe': 0.09
        },
        'motion_data': {
            'acceleration': [0.1, 0.2, 0.3],
            'gyroscope': [0.4, 0.5, 0.6],
            'magnetometer': [0.7, 0.8, 0.9]
        }
    }
    
    features = feature_engineer.extract_medical_features(test_data)
    
    # Check if features were extracted
    assert isinstance(features, pd.DataFrame)
    assert not features.empty 