'''
This file is used to load the data for the project.
It is currently only designed to work with the sklearn datasets, but could be extended to other datasets in the future.

While configured to load cifar-10, this is currently far too slow to be useful.
Looking into alternatiively approaches to extend these techniques to image data.
'''

import torch
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import urllib.request
import tarfile
import pickle

class DataLoader:
    def __init__(self, dataset_config, test_size=0.2, random_seed=42):
        self.dataset_config = dataset_config # defined in main.py
        self.test_size = test_size # default 20% of data for testing
        self.random_seed = random_seed # default seed for reproducibility
        self.scaler = StandardScaler() # standard scaler for data normalisation. mean 0 variance 1
        self.data_dir = Path('data') # data directory at project root - not relevant for sklearn datasets
        self.data_dir.mkdir(exist_ok=True)
    
    def download_cifar10(self):
        """Download and extract CIFAR-10 zip dataset"""
        dataset_path = self.data_dir / 'cifar-10-batches-py'
        if not dataset_path.exists():
            print("Downloading CIFAR-10 dataset...")
            file_path = self.data_dir / 'cifar-10-python.tar.gz'
            
            # Download
            urllib.request.urlretrieve(self.dataset_config['url'], file_path)
            
            # Extract
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=self.data_dir)
            
            # Cleanup
            file_path.unlink()
            print("Download complete and files extracted.")
    
    def load_cifar10_data(self):
        """Load CIFAR-10 dataset from files with memory efficiency"""
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        dataset_path = self.data_dir / 'cifar-10-batches-py'
        if not dataset_path.exists(): # do not redownload if already present in project root
            self.download_cifar10()
        
        # Load test data first to determine split size
        test_batch = unpickle(dataset_path / 'test_batch')
        X_test = test_batch[b'data'].astype('float32') / 255.0  # Normalize here
        y_test = np.array(test_batch[b'labels'])
        
        # Calculate how many training samples we need
        test_size = len(X_test)
        train_size = int(test_size * (1 - self.test_size) / self.test_size)
        
        # Load only enough training data
        X_train = []
        y_train = []
        for i in range(1, 6):
            batch = unpickle(dataset_path / f'data_batch_{i}')
            X_train.append(batch[b'data'])
            y_train.extend(batch[b'labels'])
            if len(y_train) >= train_size:
                break
        
        X_train = np.vstack(X_train)[:train_size].astype('float32') / 255.0
        y_train = np.array(y_train[:train_size])
        
        # Reshape data to (N, C, H, W) format if it isn't already
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(-1, 3, 32, 32)
            X_test = X_test.reshape(-1, 3, 32, 32)
        
        return X_train, y_train, X_test, y_test
    
    def load_sklearn_dataset(self):
        """Load and process sklearn datasets"""
        # extend the dataset_loaders dictionary as needed. Note that this is only configured to work for sklearn datasets
        dataset_loaders = {
            'breast_cancer': load_breast_cancer,
            'digits': load_digits,
            'wine': load_wine,
            'california_housing': fetch_california_housing
        }
        
        if self.dataset_config['name'] not in dataset_loaders:
            raise ValueError(f"Dataset {self.dataset_config['name']} not supported")
            
        dataset = dataset_loaders[self.dataset_config['name']]()
        X = dataset.data
        y = dataset.target
        
        # Convert regression targets to classification for housing dataset
        # This is a bit of a hack, but it works for now. These experiments are only designed to work with classification problems.
        if self.dataset_config['name'] == 'california_housing':
            y = np.digitize(y, bins=np.quantile(y, [0.25, 0.5, 0.75])) # Convert to 4 classes
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed
        )
        
        # dataset_info is used to store information about the dataset. 
        # Used in summary records and experiment directories
        dataset_info = {
            'name': self.dataset_config['name'], 
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y)),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_names': list(dataset.feature_names),
            'target_names': list(dataset.target_names) if hasattr(dataset, 'target_names') else None,
            'is_regression': self.dataset_config['name'] == 'california_housing'
        }
        
        return X_train, y_train, X_test, y_test, dataset_info
    
    def load_data(self):
        """Load and preprocess the configured dataset"""
        # Dictionary mapping data sources to their loading functions
        data_loaders = {
            'sklearn': self.load_sklearn_dataset,
            'local': {
                'cifar10': self.load_cifar10_data
                # Add more local datasets here as needed
            }
        }
        
        # Get the appropriate loader
        if self.dataset_config['source'] not in data_loaders:
            raise ValueError(f"Data source {self.dataset_config['source']} not supported")
        
        if self.dataset_config['source'] == 'sklearn':
            X_train, y_train, X_test, y_test, dataset_info = data_loaders['sklearn']()
        else:
            # Handle local datasets
            dataset_name = self.dataset_config['name']
            if dataset_name not in data_loaders['local']:
                raise ValueError(f"Local dataset {dataset_name} not supported")
            
            X_train, y_train, X_test, y_test = data_loaders['local'][dataset_name]()
            
            # Create dataset_info for non-sklearn datasets
            dataset_info = {
                'name': dataset_name,
                'num_features': X_train.shape[1],
                'num_classes': len(np.unique(y_train)),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': [f'feature_{i}' for i in range(X_train.shape[1])],
                'target_names': [f'class_{i}' for i in range(len(np.unique(y_train)))],
                'is_regression': False
            }
        
        if self.dataset_config['name'] == 'cifar10':
            X_train, y_train, X_test, y_test = self.load_cifar10_data()
            # For CIFAR-10, normalize using mean/std per channel instead of StandardScaler
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            
            # Normalize per channel
            mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
            std = X_train.std(dim=(0, 2, 3), keepdim=True)
            X_train = (X_train - mean) / (std + 1e-7)
            X_test = (X_test - mean) / (std + 1e-7)
            
            dataset_info = {
                'name': 'cifar10',
                'num_features': X_train.shape[1:],  # (C, H, W)
                'num_classes': len(np.unique(y_train)),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': [f'pixel_{i}' for i in range(np.prod(X_train.shape[1:]))],
                'target_names': [f'class_{i}' for i in range(len(np.unique(y_train)))],
                'is_regression': False
            }
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': torch.LongTensor(y_train),
                'y_test': torch.LongTensor(y_test),
                'dataset_info': dataset_info
            }
        
        # For other datasets, use StandardScaler as before
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'X_train': torch.FloatTensor(X_train_scaled),
            'X_test': torch.FloatTensor(X_test_scaled),
            'y_train': torch.LongTensor(y_train),
            'y_test': torch.LongTensor(y_test),
            'dataset_info': dataset_info
        }