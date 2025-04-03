import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from typing import Dict, List, Any, Tuple

class FeatureEngineer:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.variance_threshold = VarianceThreshold()
        self.is_fitted = False
        self.interaction_pairs = []
        self.feature_names = None
        
    def preprocess_data(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Preprocess the input features with advanced feature engineering.
        
        Args:
            X: Input feature matrix
            is_training: Whether this is training data
            
        Returns:
            Processed feature matrix
        """
        # Create a copy to avoid modifying the original data
        X_processed = X.copy()
        
        # 1. Handle outliers using IQR method
        for column in X_processed.columns:
            Q1 = X_processed[column].quantile(0.25)
            Q3 = X_processed[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X_processed[column] = X_processed[column].clip(lower_bound, upper_bound)
        
        # 2. Create interaction features for highly correlated pairs
        if is_training:
            # Only compute correlation pairs during training
            correlation_matrix = X_processed.corr()
            columns = X_processed.columns.tolist()
            self.interaction_pairs = []
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    if abs(correlation_matrix.loc[columns[i], columns[j]]) > 0.5:
                        self.interaction_pairs.append((columns[i], columns[j]))
        
        # Create interaction features using stored pairs
        new_features = {}
        for col1, col2 in self.interaction_pairs:
            if col1 in X_processed.columns and col2 in X_processed.columns:
                new_features[f'{col1}_{col2}_interaction'] = X_processed[col1] * X_processed[col2]
        
        # Add all interaction features at once
        for feature_name, feature_values in new_features.items():
            X_processed[feature_name] = feature_values
        
        # 3. Create polynomial features for important columns
        important_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)']
        squared_features = {}
        for feature in important_features:
            if feature in X_processed.columns:
                squared_features[f'{feature}_squared'] = X_processed[feature] ** 2
        
        # Add all squared features at once
        for feature_name, feature_values in squared_features.items():
            X_processed[feature_name] = feature_values
        
        if is_training:
            # Store feature names during training
            self.feature_names = X_processed.columns.tolist()
        else:
            # Ensure test data has same features as training
            missing_cols = set(self.feature_names) - set(X_processed.columns)
            for col in missing_cols:
                X_processed[col] = 0
            # Ensure columns are in same order
            X_processed = X_processed[self.feature_names]
        
        # 4. Scale the features
        if not self.is_fitted:
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=X_processed.columns
            )
            self.is_fitted = True
        else:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns
            )
        
        return X_processed
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get the importance of original features based on PCA loadings.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            return pd.DataFrame()
            
        # Get the loadings for each principal component
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.pca.feature_names_in_
        )
        
        # Calculate feature importance as the sum of absolute loadings
        importance = loadings.abs().sum(axis=1)
        
        return pd.DataFrame({
            'feature': importance.index,
            'importance': importance.values
        }).sort_values('importance', ascending=False)
    
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
        
        return X  # Remove PCA for now as it's not necessary
    
    def apply_pca(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply PCA transformation to the data."""
        if not self.is_fitted:
            X_pca = pd.DataFrame(
                self.pca.fit_transform(X),
                columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
            )
            self.is_fitted = True
        else:
            X_pca = pd.DataFrame(
                self.pca.transform(X),
                columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
            )
        
        return X_pca, self.pca.explained_variance_ratio_ 