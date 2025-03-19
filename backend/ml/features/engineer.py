import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from typing import Dict, List, Any, Tuple

class FeatureEngineer:
    def __init__(self, n_components: float = 0.95):
        """Initialize the feature engineering pipeline."""
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.variance_threshold = VarianceThreshold()
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data."""
        # Create a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Ensure data is numeric
        processed_data = processed_data.astype(float)
        
        # Scale the data
        processed_data = pd.DataFrame(
            self.scaler.fit_transform(processed_data),
            columns=processed_data.columns
        )
        
        return processed_data
    
    def apply_pca(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Apply PCA for dimensionality reduction."""
        # Fit and transform the data
        X_pca = self.pca.fit_transform(X)
        
        # Create DataFrame with component names
        component_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=component_names)
        
        # Calculate explained variance ratio
        explained_variance_ratio = self.pca.explained_variance_ratio_.sum()
        
        return X_pca_df, explained_variance_ratio
    
    def select_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Select features based on variance threshold."""
        # Fit and transform the data
        X_selected = self.variance_threshold.fit_transform(X)
        
        # Get selected feature names
        selected_features = X.columns[self.variance_threshold.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features)
    
    def extract_medical_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Extract medical features from structured input data."""
        features = {}
        
        # Extract basic clinical features
        if 'clinical_data' in data:
            clinical = data['clinical_data']
            features.update({
                'age': clinical.get('age', 0),
                'gender': 1 if clinical.get('gender') == 'M' else 0,
                'bmi': clinical.get('bmi', 0),
                'disease_duration': clinical.get('disease_duration', 0),
                'updrs_score': clinical.get('updrs_score', 0)
            })
        
        # Extract voice features
        if 'voice_data' in data:
            voice = data['voice_data']
            features.update({
                'jitter': voice.get('jitter', 0),
                'shimmer': voice.get('shimmer', 0),
                'nhr': voice.get('nhr', 0),
                'hnr': voice.get('hnr', 0),
                'rpde': voice.get('rpde', 0),
                'dfa': voice.get('dfa', 0),
                'ppe': voice.get('ppe', 0)
            })
        
        # Extract gait features
        if 'gait_data' in data:
            gait = data['gait_data']
            features.update({
                'stride_length': gait.get('stride_length', 0),
                'stride_time': gait.get('stride_time', 0),
                'step_length': gait.get('step_length', 0),
                'step_time': gait.get('step_time', 0),
                'cadence': gait.get('cadence', 0)
            })
        
        # Create DataFrame
        return pd.DataFrame([features])
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main method to engineer features from input data."""
        # If data is already a DataFrame, use it directly
        if isinstance(data, pd.DataFrame):
            X = data
        else:
            # Extract medical features if input is a dictionary
            X = self.extract_medical_features(data)
        
        # Preprocess the data
        X = self.preprocess_data(X)
        
        # Apply feature selection
        X = self.select_features(X)
        
        # Apply PCA
        X_pca, explained_variance = self.apply_pca(X)
        
        return X_pca 