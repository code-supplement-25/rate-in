# Rate-In: Information-Driven Adaptive Dropout Rates

This repository contains the official implementation of Rate-In, an adaptive dropout rate mechanism for improved inference-time uncertainty estimation.

## Overview

Rate-In is a novel approach to dropout that:
- Automatically adjusts dropout rates based on information preservation
- Can work with various information metrics (MI, MSE, SSIM, etc.) and neural network architectures
- Designed for inference use, independent of ground truth labels
- Works with pre-trained models by adding dropout layers post-training

## Installation

```bash
# Clone the repository
git clone https://github.com/code-supplement-25/rate-in.git
cd rate-in

# Install requirements
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

Rate-In can be easily integrated into existing neural networks. Here's a simple example:

```python
from rate_in.adaptive_dropout import AdaptiveInformationDropout, OptimizerConfig
from rate_in.model_utils import add_dropout_layers

# Configure Rate-In
optimizer_config = OptimizerConfig(
    max_iterations=100,
    learning_rate=0.10,
    decay_rate=0.9,
    stopping_error=0.01
)

# Create adaptive dropout layer
adaptive_dropout = AdaptiveInformationDropout(
    initial_p=0.5,
    information_loss_threshold=0.1,
    calc_information_loss=your_loss_function,
    optimizer_config=optimizer_config
)

# Add to your model
model_with_dropout = add_dropout_layers(
    model=your_model,
    dropoutLayer=adaptive_dropout,
    placement_layers=['layer1', 'layer2']
)
```

## Examples

We provide two comprehensive examples demonstrating Rate-In's usage:

1. [MLP with Mutual Information Loss](examples/tutorial_mlp_with_mi_loss.ipynb)
   - Simple regression task using dense neural networks
   - Mutual Information-based loss metric
   - Comparison with standard dropout approaches

2. [CNN with SSIM Loss](examples/tutorial_cnn_with_ssim_loss.ipynb)
   - Image processing task using convolutional neural networks
   - SSIM-based information loss metric
   - Demonstration of Rate-In in convolutional architectures

## Repository Structure

```
rate_in/
├── rate_in/
│   ├── adaptive_dropout.py    # Main Rate-In implementation
│   ├── baseline_dropouts.py   # Baseline dropout implementations
│   ├── model_utils.py         # Utility functions
│   └── evaluation_metrics.py  # Evaluation metrics
└── examples/                  # Tutorial notebooks
```

## Requirements

Main dependencies:
- Python ≥ 3.8
- PyTorch ≥ 1.9
- torchmetrics
- numpy
- pandas
- matplotlib

Full dependencies are listed in `requirements.txt`.
