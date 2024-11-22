"""
Comprehensive Evaluation Metrics for Machine Learning Models

This module provides implementations of various evaluation metrics for machine learning models, based on popular Python libraries.,
particularly focused on predictive power and uncertainty quantification in regression,
classification, and segmentation tasks.

The module includes metrics for:
1. Predictive Power:
    - Mean Squared Error (MSE)
    - Dice Similarity Coefficient (DSC)
    - Prediction Accuracy (ACC)
    
2. Predictive Uncertainty:
    - Expected Calibration Error (ECE)
    - Area Under the Accuracy-Rejection Curve (AUAR)
    - Boundary Uncertainty Consistency (BUC)
    - Predictive Interval Coverage Probability (PICP)
    - Interval Width
    - Interval Efficiency Ratio (IER)
"""

import numpy as np
import torch
from torchmetrics import CalibrationError, MeanSquaredError
from torchmetrics.classification import Dice

from sklearn.metrics import accuracy_score, auc
import cv2
from typing import Tuple, Union, Optional, List, Dict
from numpy.typing import NDArray

# Type aliases
NumericArray = NDArray[Union[np.float32, np.float64]]
BinaryArray = NDArray[np.uint8]
TensorType = Union[torch.Tensor, np.ndarray]

class PredictivePowerMetrics:
    """
    Collection of metrics for evaluating model's predictive power.
    """
    
    @staticmethod
    def mse(y_pred: TensorType, y_true: TensorType) -> float:
        """
        Calculate Mean Squared Error using torchmetrics implementation.
        
        Args:
            y_pred: Predicted values
            y_true: Ground truth values
            
        Returns:
            float: MSE value
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
            
        metric = MeanSquaredError()
        return float(metric(y_pred, y_true))
    
    @staticmethod
    def dice_score(pred_mask: TensorType, true_mask: TensorType) -> float:
        """
        Calculate Dice Similarity Coefficient using torchmetrics implementation.
        
        Args:
            pred_mask: Predicted binary mask
            true_mask: Ground truth binary mask
            
        Returns:
            float: Dice score
        """
        if isinstance(pred_mask, np.ndarray):
            pred_mask = torch.from_numpy(pred_mask)
        if isinstance(true_mask, np.ndarray):
            true_mask = torch.from_numpy(true_mask)
            
        metric = Dice()
        return float(metric(pred_mask, true_mask))
    
    @staticmethod
    def accuracy(y_pred: TensorType, y_true: TensorType) -> float:
        """
        Calculate classification accuracy using sklearn implementation.
        
        Args:
            y_pred: Predicted class labels
            y_true: Ground truth class labels
            
        Returns:
            float: Accuracy score
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
            
        return float(accuracy_score(y_true, y_pred))

class UncertaintyMetrics:
    """
    Collection of metrics for evaluating model's uncertainty estimates.
    """
    
    @staticmethod
    def expected_calibration_error(
        uncert: torch.Tensor,
        errors: torch.Tensor,
        n_bins: int = 15
    ) -> float:
        """
        Calculate Expected Calibration Error using torchmetrics implementation. 
        
        Args:
            uncert: Uncertainty scores (scaled between 0 to 1)
            errors: Prediction errors
            n_bins: Number of bins for calibration
            
        Returns:
            float: ECE value
        """
        metric = CalibrationError(
            task='binary',
            norm='l1',
            n_bins=n_bins
        )
        return float(metric(uncert, errors))
    
    @staticmethod
    def auar(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        uncertainty_scores: np.ndarray,
        n_thresholds: int = 100
    ) -> float:
        """
        Calculate Area Under the Accuracy-Rejection Curve, as descrbed in "Bias-reduced uncertainty estimation for deep neural classifiers" (Geifman et al., 2018).
        
        Args:
            y_pred: Predicted class labels
            y_true: Ground truth class labels
            uncertainty_scores: Uncertainty scores for each prediction
            n_thresholds: Number of threshold points for the curve
            
        Returns:
            float: AUAR score
        """
        percentiles = np.linspace(0, 100, n_thresholds)
        thresholds = np.percentile(uncertainty_scores, percentiles)
        
        accuracies = []
        rejection_fractions = []
        
        for threshold in thresholds:
            mask = uncertainty_scores <= threshold
            if mask.sum() > 0:
                acc = accuracy_score(y_true[mask], y_pred[mask])
            else:
                acc = 1.0
            
            accuracies.append(acc)
            rejection_fractions.append(1 - mask.mean())
        
        return float(auc(rejection_fractions, accuracies))
    
    @staticmethod
    def boundary_uncertainty_consistency(
        uncertainty_map: NumericArray,
        ground_truth_mask: BinaryArray,
        thickness: int = 5
    ) -> float:
        """
        Calculate Boundary Uncertainty Consistency (BUC).
        
        Args:
            uncertainty_map: 2D array of uncertainty values
            ground_truth_mask: Binary mask of ground truth
            thickness: Boundary thickness in pixels
            
        Returns:
            float: BUC score
        """
        kernel = np.ones((thickness, thickness), np.uint8)
        dilated = cv2.dilate(ground_truth_mask.astype(np.uint8), kernel)
        eroded = cv2.erode(ground_truth_mask.astype(np.uint8), kernel)
        boundary = dilated - eroded
        
        boundary_uncertainty = uncertainty_map[boundary == 1].mean()
        interior_uncertainty = uncertainty_map[
            (ground_truth_mask == 1) & (boundary == 0)
        ].mean()
        
        return float(boundary_uncertainty / (boundary_uncertainty + interior_uncertainty))
    
    @staticmethod
    def predictive_interval_metrics(
        pred_mean: NumericArray,
        pred_std: NumericArray,
        y_true: NumericArray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate predictive interval metrics (PICP, Width, and IER).
        
        Args:
            pred_mean: Predicted mean values
            pred_std: Predicted standard deviations
            y_true: True target values
            confidence_level: Confidence level for intervals
            
        Returns:
            Dict containing PICP, Width, and IER values
        """
        z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + confidence_level) / 2))
        
        # Calculate interval width
        width = 2 * z_score * pred_std
        mean_width = torch.mean(width)
        
        # Calculate PICP
        lower_bound = pred_mean - width/2
        upper_bound = pred_mean + width/2
        within_interval = ((y_true >= lower_bound) & (y_true <= upper_bound))

        picp = torch.mean(torch.as_tensor(within_interval, dtype = torch.float))
        
        # Calculate IER
        ier = mean_width / (picp + 1e-8)
        
        return {
            'picp': picp,
            'width': mean_width,
            'ier': ier
        }
