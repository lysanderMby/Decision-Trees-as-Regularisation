'''
*** Run this file to run the overall project. ***

This file is used to create the experiment directories and run the training and evaluation of the models.

Future improvements:
I would like to make this more versitile, importantly have it function on CIFAR-10.
At present, this is entirely geared towards small dataset classification problems.
I think that this could potentially be extended to regression problems, but this would require a lot of work.
Extending this to image classification would be interesting, but XGBoost would be a poor teacher for this task.
'''

from pathlib import Path
from datetime import datetime
import json
import urllib.request
import tarfile
import os
import numpy as np

# Dataset Configuration for loading CIFAR-10
# DATASET_CONFIG = {
#     'name': 'cifar10',
#     'source': 'download',
#     'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', # this should really be done through torch datasets
#     'num_classes': 10, # it is not generally easy to get this information directly from the dataset itself
#     'input_size': 3072  # 32x32x3
# }

# Use a variant of this config with the name set to your desired dataset
DATASET_CONFIG = {
    'name': 'breast_cancer',  # Options: 'breast_cancer', 'digits', 'wine', 'california_housing'
    'source': 'sklearn',
    'url': None,
    'num_classes': None,  # Will be determined from data
    'input_size': None   # Will be determined from data
}

# Dataset Configuration
# DATASET_CONFIG = {
#     'name': 'cifar10',
#     'source': 'local',  # Changed from 'sklearn'. Also possibly download?
#     'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
#     'num_classes': 10,
#     'input_size': 3072  # 32x32x3
# }

from src.data.data_loader import DataLoader
from src.models.neural_network import NeuralNetwork
from src.models.xgboost_model import XGBoostWrapper
from src.utils.plotting import Plotter
from src.utils.training import Trainer
from src.utils.summary import write_summary

def create_experiment_dir(dataset_info):
    experiments_dir = Path("experiments")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{dataset_info['name']}_{timestamp}"
    exp_dir = experiments_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def get_model_architecture(dataset_info):
    """Define model architecture based on dataset"""
    input_size = dataset_info['num_features']
    num_classes = dataset_info['num_classes']
    
    if dataset_info['name'] == 'cifar10':
        return {
            'type': 'cnn',
            'input_channels': 3,
            'architecture': [
                ('conv', 32, 3),  # (channels, kernel_size)
                ('pool', 2),      # (kernel_size)
                ('conv', 64, 3),
                ('pool', 2),
                ('conv', 64, 3),
                ('flatten',),
                ('linear', 64),
                ('linear', num_classes)
            ]
        }
    
    # Default architectures for other datasets (fully connected)
    architectures = {
        'breast_cancer': [input_size, 32, 16, num_classes],
        'digits': [input_size, 128, 64, num_classes],
        'wine': [input_size, 32, 16, num_classes],
        'california_housing': [input_size, 64, 32, num_classes],
    }
    
    if dataset_info['name'] not in architectures:
        raise ValueError(f"No default architecture for dataset {dataset_info['name']}")
    
    return {
        'type': 'mlp', # default architecture is used for sklearn classification problems
        'architecture': architectures[dataset_info['name']]
    }

def numpy_safe_json(obj):
    """Convert numpy types JSON conversion. No, I'm not proud of this"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def main():
    # Initialize components. Needs to be versitile for different datasets
    data_loader = DataLoader(DATASET_CONFIG)
    data = data_loader.load_data()
    
    # Create experiment directory with dataset info
    exp_dir = create_experiment_dir(data['dataset_info'])
    
    # Create models with dataset-specific architecture
    nn_architecture = get_model_architecture(data['dataset_info'])
    model_mimic = NeuralNetwork(None, nn_architecture)  # input_size is not needed, it's in the architecture dict
    model_original = NeuralNetwork(None, nn_architecture)
    
    # Adjust training parameters based on dataset
    num_epochs = 500 if data['dataset_info']['name'] == 'breast_cancer' else 1000
    
    trainer = Trainer(
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=0.001
    )
    
    xgb_model = XGBoostWrapper()
    xgb_predictions = xgb_model.train_and_predict(
        data['X_train'], data['y_train'], data['X_test']
    )
    
    plotter = Plotter(exp_dir)
    
    # Training and evaluation
    results = {}
    

    # Train mimic model (trained on XGBoost predictions)
    trainer.set_model(model_mimic)
    mimic_results = trainer.train(
        data['X_train'], xgb_predictions['train_pred'],
        data['X_test'], xgb_predictions['test_pred'],
        data['y_train'], data['y_test']
    )
    
    # Train original model (trained on raw data)
    trainer.set_model(model_original)
    original_results = trainer.train(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        xgb_predictions['train_pred'], xgb_predictions['test_pred']  # XGBoost predictions as true_train/test
    )
    
    # Combine results for plotting
    combined_results = {
        # Mimic model results
        'mimic_train_losses': mimic_results['train_losses'],
        'mimic_test_losses': mimic_results['test_losses'],
        'mimic_train_true_losses': mimic_results['train_true_losses'],
        'mimic_test_true_losses': mimic_results['test_true_losses'],
        
        # Original model results
        'original_train_losses': original_results['train_losses'],
        'original_test_losses': original_results['test_losses'],
        'original_train_xgb_losses': original_results['train_true_losses'],
        'original_test_xgb_losses': original_results['test_true_losses'],
        
        'mimic_layer_stats': mimic_results['layer_stats_history'],
        'original_layer_stats': original_results['layer_stats_history'],
        'epochs_sampled': mimic_results['epochs_sampled']
    }

    # Plot comparison curves
    plotter.plot_comparison_curves(combined_results, 'Loss Comparison')
    plotter.plot_layer_stats_comparison(
        combined_results['mimic_layer_stats'],
        combined_results['original_layer_stats'],
        combined_results['epochs_sampled'],
        'Layer Statistics'
    )

    
    # Save extended metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset': data['dataset_info'],
        'parameters': {
            'nn_layers': nn_architecture,
            'num_epochs': trainer.num_epochs,
            'batch_size': trainer.batch_size,
            'learning_rate': trainer.learning_rate
        },
        'results': {
            'mimic_model': {
                'final_xgb_test_loss': float(mimic_results['test_losses'][-1]),
                'final_raw_test_loss': float(mimic_results['test_true_losses'][-1])
            },
            'original_model': {
                'final_raw_test_loss': float(original_results['test_losses'][-1]),
                'final_xgb_test_loss': float(original_results['test_true_losses'][-1])
            }
        }
    }
    
    with open(exp_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4, default=numpy_safe_json)
    
    write_summary(exp_dir, combined_results, metadata)

if __name__ == "__main__":
    main()