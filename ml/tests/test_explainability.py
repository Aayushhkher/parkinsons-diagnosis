import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from explainability.explainer import ModelExplainer

@pytest.fixture
def sample_data():
    # Generate synthetic data for testing
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'feature4': np.random.rand(100),
        'feature5': np.random.rand(100)
    })
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@pytest.fixture
def model_explainer(trained_model, sample_data):
    X, _ = sample_data
    return ModelExplainer(trained_model, X.columns.tolist())

def test_setup_shap(model_explainer, sample_data):
    X, _ = sample_data
    model_explainer.setup_shap(X)
    assert model_explainer.explainer is not None

def test_setup_lime(model_explainer, sample_data):
    X, _ = sample_data
    model_explainer.setup_lime(X)
    assert model_explainer.lime_explainer is not None

def test_get_shap_values(model_explainer, sample_data):
    X, _ = sample_data
    model_explainer.setup_shap(X)
    results = model_explainer.get_shap_values(X)
    
    # Check if SHAP values were calculated
    assert 'shap_values' in results
    assert 'feature_importance' in results
    
    # Check feature importance
    importance = results['feature_importance']
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert len(importance) == X.shape[1]

def test_get_lime_explanation(model_explainer, sample_data):
    X, _ = sample_data
    model_explainer.setup_lime(X)
    instance = X.iloc[0]
    results = model_explainer.get_lime_explanation(instance)
    
    # Check if LIME explanation was generated
    assert 'explanation' in results
    assert 'feature_importance' in results
    
    # Check feature importance
    importance = results['feature_importance']
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns

def test_plot_shap_summary(model_explainer, sample_data, tmp_path):
    X, _ = sample_data
    model_explainer.setup_shap(X)
    shap_results = model_explainer.get_shap_values(X)
    
    # Test plotting
    output_path = tmp_path / "shap_summary.png"
    model_explainer.plot_shap_summary(
        shap_results['shap_values'],
        X,
        str(output_path)
    )
    assert output_path.exists()

def test_plot_lime_explanation(model_explainer, sample_data, tmp_path):
    X, _ = sample_data
    model_explainer.setup_lime(X)
    instance = X.iloc[0]
    lime_results = model_explainer.get_lime_explanation(instance)
    
    # Test plotting
    output_path = tmp_path / "lime_explanation.png"
    model_explainer.plot_lime_explanation(
        lime_results['explanation'],
        str(output_path)
    )
    assert output_path.exists()

def test_generate_explanation_report(model_explainer, sample_data, tmp_path):
    X, y = sample_data
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    
    report = model_explainer.generate_explanation_report(
        X, y, str(output_dir)
    )
    
    # Check report contents
    assert 'shap' in report
    assert 'lime' in report
    
    # Check if plots were generated
    assert (output_dir / "shap_summary.png").exists()
    assert (output_dir / "lime_explanation_0.png").exists() 