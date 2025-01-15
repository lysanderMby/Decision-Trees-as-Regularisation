'''
Functions for creating and writing experiment summaries
'''

from pathlib import Path

def write_summary(exp_dir: Path, combined_results: dict, metadata: dict):
    """
    Write a clear summary of the experiment results.
    
    Args:
        exp_dir (Path): Directory to save the summary
        combined_results (dict): Dictionary containing all training results
        metadata (dict): Dictionary containing experiment metadata
    """
    with open(exp_dir / 'experiment_summary.txt', 'w') as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=================\n\n")
        
        # Write model architecture details
        f.write("Model Architecture:\n")
        arch = metadata['parameters']['nn_architecture']
        if arch['type'] == 'mlp':
            layers = arch['architecture']
            f.write(f"Input Size: {layers[0]}\n")
            f.write(f"Hidden Layers: {layers[1:-1]}\n")
            f.write(f"Output Size: {layers[-1]}\n")
        else:  # CNN
            f.write("CNN Architecture:\n")
            f.write(f"Input Channels: {arch['input_channels']}\n")
            for layer in arch['architecture']:
                if layer[0] == 'conv':
                    f.write(f"Conv2D: {layer[1]} channels, {layer[2]}x{layer[2]} kernel\n")
                elif layer[0] == 'pool':
                    f.write(f"MaxPool: {layer[1]}x{layer[1]}\n")
                elif layer[0] == 'flatten':
                    f.write("Flatten\n")
                elif layer[0] == 'linear':
                    f.write(f"Linear: {layer[1]} units\n")
        
        f.write("\nTraining Parameters:\n")
        f.write(f"Epochs: {metadata['parameters']['num_epochs']}\n")
        f.write(f"Batch Size: {metadata['parameters']['batch_size']}\n")
        f.write(f"Learning Rate: {metadata['parameters']['learning_rate']}\n")
        
        # Key Results
        f.write("\nKEY RESULTS\n")
        f.write("-----------\n")
        f.write("Performance on Raw Data (Test Set):\n")
        f.write(f"Original Model Loss: {metadata['results']['original_model']['final_raw_test_loss']:.4f}\n")
        f.write(f"XGBoost Mimic Model Loss: {metadata['results']['mimic_model']['final_raw_test_loss']:.4f}\n\n")
        
        relative_improvement = ((metadata['results']['original_model']['final_raw_test_loss'] - 
                               metadata['results']['mimic_model']['final_raw_test_loss']) / 
                              metadata['results']['original_model']['final_raw_test_loss'] * 100)
        
        f.write(f"Relative Improvement using XGBoost as Teacher: {relative_improvement:.1f}%\n\n")
        
        # Additional Metrics
        f.write("Additional Metrics:\n")
        f.write("XGBoost Mimic Model:\n")
        f.write(f"- Final XGBoost Test Loss: {metadata['results']['mimic_model']['final_xgb_test_loss']:.4f}\n")
        f.write("\nOriginal Model:\n")
        f.write(f"- Final XGBoost Test Loss: {metadata['results']['original_model']['final_xgb_test_loss']:.4f}\n") 
        
        # Add layer statistics summary
        f.write("\nFINAL LAYER STATISTICS\n")
        f.write("=====================\n")
        
        mimic_stats = combined_results['mimic_layer_stats'][-1]
        original_stats = combined_results['original_layer_stats'][-1]
        
        f.write("\nMimic Model:\n")
        for layer_name, stats in mimic_stats.items():
            f.write(f"{layer_name}:\n")
            f.write(f"  Norm: {stats['norm']:.4f}\n")
            f.write(f"  Std: {stats['std']:.4f}\n")
        
        f.write("\nOriginal Model:\n")
        for layer_name, stats in original_stats.items():
            f.write(f"{layer_name}:\n")
            f.write(f"  Norm: {stats['norm']:.4f}\n")
            f.write(f"  Std: {stats['std']:.4f}\n") 