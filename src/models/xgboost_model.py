'''
Simplistic XGBoost wrapper
Designed to take as input test and train data and output predictions for both
Only takes train output values as input (of course)
'''

import xgboost as xgb
import torch
from tqdm import tqdm
import numpy as np

class XGBoostWrapper:
    def __init__(self, n_estimators=100, random_seed=42, batch_size=1000):
        self.batch_size = batch_size
        self.model = None  # Will be initialized in train_and_predict with correct num_class
        self.n_estimators = n_estimators
        self.random_seed = random_seed
    
    def predict_in_batches(self, X):
        """Predict probabilities in batches to manage memory"""
        all_predictions = []
        n_samples = len(X)
        
        for i in tqdm(range(0, n_samples, self.batch_size), desc="Predicting"):
            batch_X = X[i:i + self.batch_size]
            batch_pred = self.model.predict_proba(batch_X)
            all_predictions.append(batch_pred)
            
        return np.vstack(all_predictions)
    
    def train_and_predict(self, X_train, y_train, X_test):
        # Determine number of classes from training data
        num_classes = len(np.unique(y_train.numpy() if torch.is_tensor(y_train) else y_train))
        
        # Initialize model with correct number of classes
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_seed,
            tree_method='hist',
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            enable_categorical=False,
            num_class=num_classes
        )
        
        # Reshape input if it's not 2D
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
            
        # Convert to numpy if tensors
        if torch.is_tensor(X_train):
            X_train = X_train.cpu().numpy()
        if torch.is_tensor(X_test):
            X_test = X_test.cpu().numpy()
        if torch.is_tensor(y_train):
            y_train = y_train.cpu().numpy()
        
        print(f"\nXGBoost Training Details:")
        print(f"Training XGBoost model with {self.n_estimators} trees...")
        print(f"Number of classes: {num_classes}")
        print(f"Input shape: {X_train.shape}")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=True
        )
        
        print("\nGenerating predictions...")
        train_pred = self.predict_in_batches(X_train)
        test_pred = self.predict_in_batches(X_test)
        
        return {
            'train_pred': torch.FloatTensor(train_pred),
            'test_pred': torch.FloatTensor(test_pred)
        }