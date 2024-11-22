"""
Baseline Dropout Layer Implementations

This module provides custom dropout layer implementations for neural networks,
including constant and scheduled dropout strategies. These implementations
extend PyTorch's Module class and can be used as drop-in replacements for
standard dropout layers.

The module includes:
- ScheduledDropout: Dropout with linearly declining probability over iterations
"""

import torch
from typing import Optional, Union
from torch import Tensor


class ConstantDropout(torch.nn.Module):
    """
    Implements a dropout layer with constant dropout probability, as described in "Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research" (Srivastava et al., 2014).
    
    This layer maintains a fixed dropout rate throughout training, similar to
    standard PyTorch dropout but with additional tracking capabilities.
    
    Args:
        p (float): Dropout probability (default: 0.5)
        name (str): Identifier for the layer (default: "")
        verbose (int): Verbosity level for logging (default: 0)
        
    Attributes:
        initial_dropout_rate (float): The initial and fixed dropout rate
        dropout_rate (nn.Parameter): Dropout rate as a PyTorch parameter
        name (str): Layer identifier
        verbose (int): Verbosity level
    
    Example:
        >>> layer = ConstantDropout(p=0.3, name="dropout_1")
        >>> x = torch.randn(10, 20)
        >>> output = layer(x)
    """
    
    def __init__(
        self,
        p: float = 0.5,
        name: str = "",
        verbose: int = 0
    ):
        super().__init__()
        
        if not 0 <= p <= 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        
        self.initial_dropout_rate = p
        self.dropout_rate = torch.nn.Parameter(torch.tensor(p), requires_grad=False)
        self.name = name
        self.verbose = verbose
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout to the input tensor.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output with dropout applied
        """
        return torch.nn.functional.dropout(
            x,
            p=self.dropout_rate.item(),
            training=self.training
        )
    
    def extra_repr(self) -> str:
        """Provide extra information for string representation."""
        return f'p={self.dropout_rate.item()}, name="{self.name}"'


class ScheduledDropout(torch.nn.Module):
    """
    Implements a dropout layer with linearly declining dropout probability.
    
    The dropout rate decreases linearly from the initial probability to 0
    over a specified number of iterations.
    
    Args:
        p (float): Initial dropout probability (default: 0.5)
        reps (int): Number of iterations for complete decline (default: 30)
        name (str): Identifier for the layer (default: "")
        verbose (int): Verbosity level for logging (default: 0)
        
    Attributes:
        initial_dropout_rate (float): The initial dropout rate
        dropout_rate (nn.Parameter): Current dropout rate as a PyTorch parameter
        reps (int): Total iterations for dropout decline
        iter (int): Current iteration count
        name (str): Layer identifier
        verbose (int): Verbosity level
    
    Example:
        >>> layer = ScheduledDropout(p=0.5, reps=100, name="scheduled_1")
        >>> for i in range(100):
        ...     output = layer(input_tensor)  # dropout rate decreases each iteration
    """
    
    def __init__(
        self,
        p: float = 0.5,
        reps: int = 30,
        name: str = "",
        verbose: int = 0
    ):
        super().__init__()
        
        if not 0 <= p <= 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        if reps < 1:
            raise ValueError("Number of repetitions must be positive")
            
        self.initial_dropout_rate = p
        self.dropout_rate = torch.nn.Parameter(torch.tensor(p), requires_grad=False)
        self.reps = reps
        self.iter = 0
        self.name = name
        self.verbose = verbose
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout to the input tensor with declining probability, as described in "Annealed dropout training of deep networks. In 2014 IEEE Spoken Language Technology Workshop (SLT)" (Rennie et al., 2014).
        
        Updates the dropout rate based on the current iteration and applies
        dropout to the input tensor.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output with dropout applied
        """
        # Update iteration counter
        self.iter = min(self.iter + 1, self.reps)
        
        # Calculate new dropout rate
        decline_factor = (1 - (self.iter - 1) / (self.reps - 1))
        new_rate = self.initial_dropout_rate * decline_factor
        self.dropout_rate.data = torch.tensor(new_rate)
        
        # Log if verbose
        if self.verbose >= 1:
            print(f"{self.name} dropout rate: {self.dropout_rate.item():.4f}")
        
        # Apply dropout
        return torch.nn.functional.dropout(
            x,
            p=self.dropout_rate.item(),
            training=self.training
        )
    
    def extra_repr(self) -> str:
        """Provide extra information for string representation."""
        return (f'p={self.dropout_rate.item()}, reps={self.reps}, '
                f'iter={self.iter}, name="{self.name}"')
    
    def reset_schedule(self):
        """Reset the iteration counter and dropout rate to initial values."""
        self.iter = 0
        self.dropout_rate.data = torch.tensor(self.initial_dropout_rate)
