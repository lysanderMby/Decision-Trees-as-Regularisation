'''
Main training logic for the neural network
Note that the XGBoost model is instead output in the xgboost_model.py file in the defining class
'''

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from tqdm import tqdm

def get_layer_stats(model):
    """Compute norm statistics for each layer in the model"""
    layer_stats = {}
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only tracking weight matrices, not biases
            norm = torch.norm(param.data).item()
            std = torch.std(param.data).item()
            layer_stats[name] = {
                'norm': norm,
                'std': std
            }
    return layer_stats

class Trainer:
    def __init__(self, num_epochs, batch_size, learning_rate, stats_frequency=50):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.stats_frequency = stats_frequency
        self.criterion = nn.CrossEntropyLoss()
        self.model = None
        self.optimizer = None
        
    def set_model(self, model):
        """Initialize model and optimizer"""
        self.model = model
        # Use Adam optimizer with weight decay for CNNs
        if hasattr(model, 'architecture_type') and model.architecture_type == 'cnn':
            self.optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-4  # L2 regularization
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.learning_rate
            )
        
    def train(self, X_train, y_train, X_test, y_test, y_true_train, y_true_test):
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer not initialized. Call set_model() first.")
            
        train_losses = []
        test_losses = []
        train_true_losses = []
        test_true_losses = []
        layer_stats_history = []
        epochs_sampled = []
        
        pbar = tqdm(range(self.num_epochs), desc="Training")
        
        for epoch in pbar:
            # Training phase
            self.model.train()
            epoch_train_losses = []
            
            # loop through predefined batches
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i + self.batch_size]
                batch_y = y_train[i:i + self.batch_size]
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_train_losses.append(loss.item())
            
            # Collect layer statistics periodically. Used in layer_statistics collection
            if epoch % self.stats_frequency == 0:
                layer_stats_history.append(get_layer_stats(self.model))
                epochs_sampled.append(epoch)
            
            # Evaluation phase
            # This is used to collect the training and testing losses on training and test data
            # This is expected to be different depending on whether the original or mimic model is being trained
            self.model.eval()
            with torch.no_grad():
                train_outputs = self.model(X_train)
                train_loss = self.criterion(train_outputs, y_train)
                train_losses.append(train_loss.item())
                
                test_outputs = self.model(X_test)
                test_loss = self.criterion(test_outputs, y_test)
                test_losses.append(test_loss.item())
                
                train_true_loss = self.criterion(train_outputs, y_true_train)
                test_true_loss = self.criterion(test_outputs, y_true_test)
                train_true_losses.append(train_true_loss.item())
                test_true_losses.append(test_true_loss.item())
            
            pbar.set_postfix({
                'train_loss': f'{train_loss.item():.4f}',
                'test_loss': f'{test_loss.item():.4f}',
                'true_test_loss': f'{test_true_loss.item():.4f}'
            })
        # passed to the 
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_true_losses': train_true_losses,
            'test_true_losses': test_true_losses,
            'layer_stats_history': layer_stats_history,
            'epochs_sampled': epochs_sampled
        }