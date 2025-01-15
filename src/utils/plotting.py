'''
Currently only plots learning curves and weight matrices
'''

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import torch.nn as nn

class Plotter:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir

    def plot_comparison_curves(self, results_dict, title):
        """Plot multiple loss curves for comparison"""
        plt.figure(figsize=(12, 8))
        
        # Plot mimic model performance
        plt.subplot(2, 1, 1)
        plt.plot(results_dict['mimic_train_losses'], label='Train Loss on XGBoost')
        plt.plot(results_dict['mimic_test_losses'], label='Test Loss on XGBoost')
        plt.plot(results_dict['mimic_train_true_losses'], label='Train Loss on Raw Data')
        plt.plot(results_dict['mimic_test_true_losses'], label='Test Loss on Raw Data')
        plt.title(f'Mimic Model Performance - {title}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot original model performance
        plt.subplot(2, 1, 2)
        plt.plot(results_dict['original_train_losses'], label='Train Loss on Raw Data')
        plt.plot(results_dict['original_test_losses'], label='Test Loss on Raw Data')
        plt.plot(results_dict['original_train_xgb_losses'], label='Train Loss on XGBoost')
        plt.plot(results_dict['original_test_xgb_losses'], label='Test Loss on XGBoost')
        plt.title(f'Original Model Performance - {title}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{title.replace(' ', '_')}_comparison.png")
        plt.close()
        
    def plot_curves(self, train_values, test_values, ylabel, title):
        plt.figure(figsize=(10, 6))
        plt.plot(train_values, label='Train')
        plt.plot(test_values, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(self.save_dir / f"{title.replace(' ', '_')}.png")
        plt.close()
        
    def plot_weight_matrices(self, model, title):
        linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
        num_linear_layers = len(linear_layers)

        plt.figure(figsize=(15, 5 * num_linear_layers))
        for i, layer in enumerate(linear_layers):
            weights = layer.weight.detach().cpu().numpy()
            plt.subplot(num_linear_layers, 1, i+1)
            sns.heatmap(weights, cmap='coolwarm', center=0)
            plt.title(f'Layer {i+1} Weights')
            plt.xlabel('Input Features')
            plt.ylabel('Neurons')
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{title.replace(' ', '_')}_weights.png")
        plt.close()
        
    def plot_layer_stats_comparison(self, mimic_stats, original_stats, epochs_sampled, title):
        """Plot layer statistics comparison between mimic and original models."""
        plt.figure(figsize=(15, 10))
        
        # Plot norms
        plt.subplot(2, 1, 1)
        for layer_name in mimic_stats[0].keys():
            # Extract layer number from the name (e.g., 'layers.0.weight' -> '1')
            layer_num = str(int(layer_name.split('.')[1]) + 1)
            
            mimic_norms = [stats[layer_name]['norm'] for stats in mimic_stats]
            original_norms = [stats[layer_name]['norm'] for stats in original_stats]
            
            plt.plot(epochs_sampled, mimic_norms, label=f'Mimic Layer #{layer_num}', linestyle='-')
            plt.plot(epochs_sampled, original_norms, label=f'Original Layer #{layer_num}', linestyle='--')
        
        plt.title('Layer Norms Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('Norm')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot standard deviations
        plt.subplot(2, 1, 2)
        for layer_name in mimic_stats[0].keys():
            # Extract layer number from the name
            layer_num = str(int(layer_name.split('.')[1]) + 1)
            
            mimic_stds = [stats[layer_name]['std'] for stats in mimic_stats]
            original_stds = [stats[layer_name]['std'] for stats in original_stats]
            
            plt.plot(epochs_sampled, mimic_stds, label=f'Mimic Layer #{layer_num}', linestyle='-')
            plt.plot(epochs_sampled, original_stds, label=f'Original Layer #{layer_num}', linestyle='--')
        
        plt.title('Layer Standard Deviations Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('Standard Deviation')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{title.replace(' ', '_')}_layer_stats.png", bbox_inches='tight')
        plt.close()