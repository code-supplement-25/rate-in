import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class OptimizerConfig:
    """Configuration for the dropout rate optimization process."""
    max_iterations: int = 30
    learning_rate: float = 0.10
    decay_rate: float = 0.9
    stopping_error: float = 0.01

class AdaptiveInformationDropout(torch.nn.Module):
    """
    Implements an adaptive dropout layer that adjusts the dropout rate based on information loss.
    
    The layer dynamically adjusts the dropout rate to maintain a specified information loss threshold
    using an optimization process.
    """
    
    def __init__(
        self,
        initial_p: float,
        calc_information_loss: Callable,
        information_loss_threshold: float = 0.01,
        optimizer_config: Optional[OptimizerConfig] = None,
        name: str = "",
        verbose: int = 0,
        **kwargs
    ):
        super().__init__()
        
        # Validate inputs
        if not 0 <= initial_p <= 1:
            raise ValueError("Initial dropout probability must be between 0 and 1")
        if not callable(calc_information_loss):
            raise ValueError("calc_information_loss must be a callable")
            
        # Initialize parameters
        self.p = torch.nn.Parameter(torch.tensor(initial_p), requires_grad=False)
        self.calc_information_loss = calc_information_loss
        self.information_loss_threshold = information_loss_threshold
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.name = name
        self.verbose = verbose
        self.additional_properties = kwargs.get('properties', {})
        
    def _apply_dropout(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        """Apply dropout with the given rate."""
        return torch.nn.functional.dropout(x, p=rate, training=self.training)
    
    def _optimize_dropout_rate(self, x: torch.Tensor) -> float:
        """Optimize the dropout rate to achieve the desired information loss threshold."""
        pre_dropout = x.detach()
        config = self.optimizer_config
        
        for iteration in range(config.max_iterations):
            # Apply current dropout rate
            current_rate = np.clip(self.p.item(), 0, 1)
            post_dropout = self._apply_dropout(pre_dropout, current_rate)
            
            # Calculate information loss
            info_loss = self.calc_information_loss(
                pre_dropout=pre_dropout,
                post_dropout=post_dropout,
                properties=self.additional_properties
            )
            
            # Calculate error and update learning rate
            error = info_loss.item() - self.information_loss_threshold
            current_lr = (config.learning_rate * config.decay_rate 
                         if iteration % 10 == 0 else config.learning_rate)
            
            # Update dropout rate
            updated_rate = current_rate - current_lr / (1 + abs(error)) * error
            self.p.data = np.clip(torch.tensor(updated_rate), 0, 1) 
            
            # Log progress if verbose
            if self.verbose >= 2:
                self._log_progress(current_rate, info_loss.item())
                
            # Check stopping condition
            if abs(error) < config.stopping_error:
                break
                
        # Final rate and logging
        final_rate = np.clip(self.p.item(), 0, 1)
        if self.verbose >= 1:
            self._log_progress(final_rate, info_loss.item(), final=True)
            
        return final_rate
    
    def _log_progress(self, rate: float, loss: float, final: bool = False):
        """Log optimization progress."""
        prefix = "Final" if final else "Current"
        print(f"{self.name}: {prefix} Dropout Rate: {100*rate:.1f}% | Loss: {loss:.3f}")
        if final:
            print()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the adaptive dropout layer."""
        if self.training:
            optimized_rate = self._optimize_dropout_rate(x)
            return self._apply_dropout(x, optimized_rate)
        return x
