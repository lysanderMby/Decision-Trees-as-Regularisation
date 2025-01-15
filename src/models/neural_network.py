'''
Very simplistic architecture. MLP which takes as input a vector of layer sizes with an input size
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=None, architecture=None):
        super(NeuralNetwork, self).__init__()
        if architecture is None:
            raise ValueError("Architecture must be provided")
            
        self.architecture_type = architecture['type']
        
        if self.architecture_type == 'mlp':
            self.layers = self._build_mlp(architecture['architecture'])
        elif self.architecture_type == 'cnn':
            self.layers = self._build_cnn(architecture)
        else:
            raise ValueError(f"Unknown architecture type: {self.architecture_type}")
    
    def _build_mlp(self, layer_sizes):
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No ReLU after last layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def _build_cnn(self, architecture):
        layers = []
        in_channels = architecture['input_channels']
        current_size = 32  # CIFAR-10 image size
        
        for layer_type, *params in architecture['architecture']:
            if layer_type == 'conv':
                out_channels, kernel_size = params
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
                layers.append(nn.ReLU())
                in_channels = out_channels
            elif layer_type == 'pool':
                kernel_size = params[0]
                layers.append(nn.MaxPool2d(kernel_size))
                current_size //= kernel_size
            elif layer_type == 'flatten':
                layers.append(nn.Flatten())
            elif layer_type == 'linear':
                out_features = params[0]
                # Find the last layer with output features by skipping activation layers
                if isinstance(layers[-1], nn.Flatten):
                    in_features = in_channels * current_size * current_size
                else:
                    # Look backwards through layers to find the last layer with out_features
                    for layer in reversed(layers):
                        if hasattr(layer, 'out_features'):
                            in_features = layer.out_features
                            break
                
                layers.append(nn.Linear(in_features, out_features))
                if out_features != architecture['architecture'][-1][1]:  # Not last layer
                    layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.architecture_type == 'cnn':
            # Reshape input if needed (batch_size, channels, height, width)
            if len(x.shape) == 2:
                x = x.view(-1, 3, 32, 32)
        return self.layers(x)