import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.explainability.explainer import ModelExplainer
import os
import shutil

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'feature1': np.random.rand(5),
        'feature2': np.random.rand(5),
        'feature3': np.random.rand(5),
        'feature4': np.random.rand(5),
        'feature5': np.random.rand(5)
    })

@pytest.fixture
def trained_model(sample_data):
    """Create a trained Random Forest model."""
    X = sample_data
    y = np.random.randint(0, 2, size=5)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

@pytest.fixture
def model_explainer(trained_model, sample_data):
    """Create a ModelExplainer instance."""
    return ModelExplainer(trained_model, feature_names=sample_data.columns.tolist())

@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary directory for output files."""
    return str(tmp_path / "explanations")

def test_setup_shap(model_explainer, sample_data):
    """Test SHAP explainer setup."""
    model_explainer.setup_shap(sample_data)
    assert model_explainer.explainer is not None

def test_setup_lime(model_explainer, sample_data):
    """Test LIME explainer setup."""
    model_explainer.setup_lime(sample_data)
    assert model_explainer.lime_explainer is not None

def test_get_shap_values(model_explainer, sample_data):
    """Test SHAP value calculation."""
    result = model_explainer.get_shap_values(sample_data)
    
    # Check if SHAP values are calculated
    assert 'shap_values' in result
    assert 'feature_importance' in result
    
    # Check if feature importance is valid
    feature_importance = result['feature_importance']
    assert len(feature_importance) > 0
    assert all('feature' in item and 'importance' in item for item in feature_importance)

def test_get_lime_explanation(model_explainer, sample_data):
    """Test LIME explanation generation."""
    instance = sample_data.iloc[0]
    result = model_explainer.get_lime_explanation(instance)
    
    # Check if explanation is generated
    assert 'explanation' in result
    assert 'feature_importance' in result
    
    # Check if feature importance is valid
    feature_importance = result['feature_importance']
    assert len(feature_importance) > 0
    assert all('feature' in item and 'importance' in item for item in feature_importance)

def test_plot_shap_summary(model_explainer, sample_data, output_dir):
    """Test SHAP summary plot generation."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get SHAP values
    shap_values = model_explainer.get_shap_values(sample_data)['shap_values']
    
    # Generate plot
    output_path = os.path.join(output_dir, 'shap_summary.png')
    model_explainer.plot_shap_summary(shap_values, sample_data, output_path)
    
    # Check if plot is generated
    assert os.path.exists(output_path)

def test_plot_lime_explanation(model_explainer, sample_data, output_dir):
    """Test LIME explanation plot generation."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get LIME explanation
    instance = sample_data.iloc[0]
    explanation = model_explainer.get_lime_explanation(instance)['explanation']
    
    # Generate plot
    output_path = os.path.join(output_dir, 'lime_explanation.png')
    model_explainer.plot_lime_explanation(explanation, output_path)
    
    # Check if plot is generated
    assert os.path.exists(output_path)

def test_generate_explanation_report(model_explainer, sample_data, output_dir):
    """Test complete explanation report generation."""
    # Generate report
    y = np.random.randint(0, 2, size=len(sample_data))
    report = model_explainer.generate_explanation_report(sample_data, y, output_dir)
    
    # Check if report contains all components
    assert 'shap' in report
    assert 'lime' in report
    
    # Check if SHAP components are present
    assert 'shap_values' in report['shap']
    assert 'feature_importance' in report['shap']
    
    # Check if LIME components are present
    assert len(report['lime']) > 0
    assert all('explanation' in item and 'feature_importance' in item for item in report['lime'])
    
    # Check if plots are generated
    assert os.path.exists(os.path.join(output_dir, 'shap_summary.png'))
    assert any(os.path.exists(os.path.join(output_dir, f'lime_explanation_{i}.png')) for i in range(5)) 