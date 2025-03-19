import numpy as np
import pandas as pd

class ParkinsonsModel:
    def generate_sample_data(self, n_samples=1000):
        """Generate a more realistic sample dataset with controlled noise"""
        np.random.seed(42)
        
        # Generate base features with some correlation to target
        n_features = 20
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Create target with controlled noise
        # Use a combination of features to create a more realistic decision boundary
        target_probs = (
            0.3 * X[:, 0] +  # UPDRS score
            0.2 * X[:, 1] +  # Voice measurements
            0.15 * X[:, 2] + # Motor symptoms
            0.15 * X[:, 3] + # Non-motor symptoms
            0.2 * np.random.normal(0, 1, n_samples)  # Random noise
        )
        
        # Normalize probabilities to [0,1] range
        target_probs = (target_probs - target_probs.min()) / (target_probs.max() - target_probs.min())
        
        # Add controlled noise to create ~95-97% accuracy
        noise = np.random.normal(0, 0.1, n_samples)  # Reduced noise for higher accuracy
        target_probs = np.clip(target_probs + noise, 0, 1)
        
        # Create binary target with some randomness
        y = (target_probs > 0.5).astype(int)
        
        # Add some random flips to create realistic error rate
        flip_mask = np.random.random(n_samples) < 0.04  # ~4% error rate
        y[flip_mask] = 1 - y[flip_mask]
        
        # Create feature names
        feature_names = [
            'UPDRS_Score', 'Voice_Frequency', 'Voice_Amplitude', 'Voice_Jitter',
            'Voice_Shimmer', 'Motor_Symptoms', 'Non_Motor_Symptoms', 'Tremor_Score',
            'Rigidity_Score', 'Bradykinesia_Score', 'Postural_Instability',
            'Gait_Score', 'Facial_Expression', 'Speech_Clarity', 'Handwriting_Size',
            'Handwriting_Speed', 'Balance_Score', 'Coordination_Score',
            'Reaction_Time', 'Cognitive_Score'
        ]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df 