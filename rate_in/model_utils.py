"""
Neural Network Model Utilities

This module provides utilities for manipulating and analyzing neural network models,
particularly focused on dropout layer management and feature map analysis.
"""

import torch
import copy
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from adaptive_dropout import AdaptiveInformationDropout  # Import the correct implementation

class CatchFeatureMap(torch.nn.Module):
    """
    A module that captures feature maps during forward pass and stores them in a global dictionary.
    
    Attributes:
        name (str): Unique identifier for the feature map
    """
    
    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name
        self.feature_map = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that stores the feature map and passes through the input unchanged.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Unchanged input tensor
        """
        self.feature_map = x.detach() 

        return x


def update_dropout_rates(
    model: torch.nn.Module,
    rates: Dict[str, float],
    parent_name: str = ''
) -> torch.nn.Module:
    """
    Recursively update dropout rates in a model.
    
    Args:
        model: The neural network model
        rates: Dictionary mapping layer names to new dropout rates
        parent_name: Name of parent module for recursive calls
        
    Returns:
        Updated model
    """
    for name, layer in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        if (full_name in rates.keys()) and isinstance(layer, (torch.nn.Dropout, AdaptiveInformationDropout)):
            layer.p = torch.tensor(rates[full_name])  # Update using the parameter
            print(f"Updated dropout rate of layer '{full_name}' to {rates[full_name]}")
        elif len(list(layer.children())) > 0:
            update_dropout_rates(layer, rates, full_name)
            
    return model


def extract_dropout_rates(
    model: torch.nn.Module,
    parent_name: str = ''
) -> List[Tuple[str, torch.Tensor]]:
    """
    Extract dropout rates from all dropout layers in a model.
    
    Args:
        model: The neural network model
        parent_name: Name of parent module for recursive calls
        
    Returns:
        List of tuples containing layer names and their dropout rates
    """
    dropout_rates = []
    
    for name, layer in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        if isinstance(layer, (torch.nn.Dropout, AdaptiveInformationDropout)):
            dropout_rates.append((full_name, layer.p))  # Access the parameter directly
        elif len(list(layer.children())) > 0:
            dropout_rates.extend(extract_dropout_rates(layer, full_name))
    
    return dropout_rates

def extract_feature_maps(
    model: torch.nn.Module,
    parent_name: str = ''
) -> List[Tuple[str, torch.Tensor]]:
    """
    Extract feature maps from all hook layers in a model.
    
    Args:
        model: The neural network model
        parent_name: Name of parent module for recursive calls
        
    Returns:
        List of tuples containing layer names and their feature maps
    """
    feature_maps = []
    
    for name, layer in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        if isinstance(layer, (CatchFeatureMap)):
            feature_maps.append((full_name, layer.feature_map))  # Access the parameter directly
        elif len(list(layer.children())) > 0:
            feature_maps.extend(extract_feature_maps(layer, full_name))
    
    return feature_maps

def replace_dropout_layers(
    model: torch.nn.Module,
    layer_type: str = 'adaptive',
    clone_model: bool = True,
    parent_name: str = '',
    layer_properties: Dict[str, dict] = {}
) -> torch.nn.Module:
    """
    Replace standard dropout layers with custom dropout implementations.
    
    Args:
        model: The neural network model
        layer_type: Type of replacement layer ('adaptive' or 'catcher')
        clone_model: Whether to create a copy of the model
        parent_name: Name of parent module for recursive calls
        layer_properties: Properties for new layers
        
    Returns:
        Model with replaced dropout layers
    """
    if clone_model:
        model = copy.deepcopy(model)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            attr_name = name.split('.')[-1]
            
            if layer_type == 'adaptive':
                properties = layer_properties.get(name, {})
                setattr(parent, attr_name, 
                       AdaptiveInformationDropout(
                           initial_p=module.p,
                           calc_information_loss=properties.get('calc_information_loss'),
                           information_loss_threshold=properties.get('information_loss_threshold', 0.01),
                           optimizer_config=properties.get('optimizer_config'),
                           name=name,
                           verbose=properties.get('verbose', 0),
                           properties=properties.get('properties', {})
                       ))
            elif layer_type == 'catcher':
                setattr(parent, attr_name, CatchFeatureMap(name=name))
    
    return model


def add_dropout_layers(
    model: torch.nn.Module,
    dropoutLayer: torch.nn.Module,
    placement_layers: List[str],
    parent_name: str = ''
) -> torch.nn.Module:
    """
    Add dropout layers after specified layers in the model.
    
    Args:
        model: The neural network model
        dropoutLayer: Dropout layer to add
        placement_layers: Names of layers after which to add dropout
        parent_name: Name of parent module for recursive calls
        
    Returns:
        Model with added dropout layers
    """
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        if list(module.children()):
            add_dropout_layers(module, dropoutLayer, placement_layers, full_name)
        elif full_name in placement_layers:
            new_dropout = copy.deepcopy(dropoutLayer)
            if hasattr(new_dropout, 'name'):
                new_dropout.name = f"{full_name}.dropout"
            
            new_module = torch.nn.Sequential(
                module,
                new_dropout
            )
            setattr(model, name, new_module)
    
    return model


def calc_CoV(x: torch.Tensor) -> float:
    """
    Calculate the Coefficient of Variation (CoV) for a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Coefficient of Variation
    """
    return x.std() / torch.abs(x).mean()
