import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.preprocessing import binarize
from typing import Dict, List, Tuple, Optional
import torch

class EvaluationMetrics:
    """
    Class for calculating various evaluation metrics for crowd risk prediction
    """
    def __init__(self):
        pass

    def calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Area Under the ROC Curve
        Args:
            y_true: Ground truth binary labels (0 or 1)
            y_pred: Predicted probabilities or scores
        Returns:
            AUC score
        """
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            # Handle edge case where only one class is present
            return 0.5  # Random classifier score

    def calculate_precision_recall_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Area Under the Precision-Recall Curve
        Args:
            y_true: Ground truth binary labels (0 or 1)
            y_pred: Predicted probabilities or scores
        Returns:
            PR AUC score
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        return auc(recall, precision)

    def calculate_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Tuple[float, float, float]:
        """
        Calculate Precision, Recall, and F1-score
        Args:
            y_true: Ground truth binary labels (0 or 1)
            y_pred: Predicted probabilities or scores
            threshold: Threshold to convert probabilities to binary predictions
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        y_pred_binary = binarize(y_pred.reshape(1, -1), threshold=threshold)[0]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score

    def calculate_false_alarm_rate(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate false alarm rate (false positive rate)
        Args:
            y_true: Ground truth binary labels (0 or 1)
            y_pred: Predicted probabilities or scores
            threshold: Threshold to convert probabilities to binary predictions
        Returns:
            False alarm rate
        """
        y_pred_binary = binarize(y_pred.reshape(1, -1), threshold=threshold)[0]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        # False alarm rate = false positives / total negatives
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return far

    def calculate_detection_accuracy(self, predicted_times: np.ndarray, actual_times: np.ndarray, tolerance: float = 2.0) -> float:
        """
        Calculate accuracy of time-to-event predictions
        Args:
            predicted_times: Predicted times to events (in seconds)
            actual_times: Actual times to events (in seconds)
            tolerance: Tolerance for considering prediction correct (in seconds)
        Returns:
            Detection accuracy within tolerance
        """
        if len(predicted_times) != len(actual_times):
            raise ValueError("Predicted and actual times must have the same length")
        
        if len(predicted_times) == 0:
            return 0.0
        
        # Calculate absolute errors
        errors = np.abs(predicted_times - actual_times)
        
        # Calculate accuracy within tolerance
        accurate_predictions = np.sum(errors <= tolerance)
        accuracy = accurate_predictions / len(predicted_times)
        
        return accuracy

    def calculate_heatmap_iou(self, pred_heatmap: np.ndarray, true_heatmap: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate Intersection over Union for heatmaps
        Args:
            pred_heatmap: Predicted heatmap
            true_heatmap: Ground truth heatmap
            threshold: Threshold to binarize heatmaps
        Returns:
            IoU score
        """
        # Binarize heatmaps
        pred_binary = (pred_heatmap > threshold).astype(int)
        true_binary = (true_heatmap > threshold).astype(int)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, true_binary).sum()
        union = np.logical_or(pred_binary, true_binary).sum()
        
        # Calculate IoU
        iou = intersection / union if union != 0 else 0.0
        
        return iou

    def calculate_pixel_accuracy(self, pred_heatmap: np.ndarray, true_heatmap: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate pixel-wise accuracy between heatmaps
        Args:
            pred_heatmap: Predicted heatmap
            true_heatmap: Ground truth heatmap
            threshold: Threshold to binarize heatmaps
        Returns:
            Pixel accuracy
        """
        # Binarize heatmaps
        pred_binary = (pred_heatmap > threshold).astype(int)
        true_binary = (true_heatmap > threshold).astype(int)
        
        # Calculate accuracy
        correct_pixels = np.sum(pred_binary == true_binary)
        total_pixels = pred_binary.size
        
        accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return accuracy

    def calculate_mae_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Mean Absolute Error and Mean Squared Error
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        Returns:
            Tuple of (MAE, MSE)
        """
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        
        return mae, mse

    def calculate_correlation_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Pearson correlation coefficient
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        Returns:
            Correlation coefficient
        """
        # Handle edge case where one array is constant
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            if np.array_equal(y_true, y_pred):
                return 1.0
            else:
                return 0.0
        
        correlation_matrix = np.corrcoef(y_true, y_pred)
        return correlation_matrix[0, 1]

    def evaluate_risk_prediction(self, 
                               ciri_predictions: np.ndarray, 
                               ground_truth_risk: np.ndarray,
                               risk_threshold: float = 0.7) -> Dict[str, float]:
        """
        Evaluate the overall risk prediction performance
        Args:
            ciri_predictions: Predicted CIRI values
            ground_truth_risk: Ground truth risk values (binary or continuous)
            risk_threshold: Threshold to classify as high-risk event
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure inputs are numpy arrays
        ciri_predictions = np.asarray(ciri_predictions)
        ground_truth_risk = np.asarray(ground_truth_risk)
        
        # If ground truth is continuous, binarize it based on threshold
        if len(np.unique(ground_truth_risk)) > 2:
            y_true = (ground_truth_risk > risk_threshold).astype(int)
        else:
            y_true = ground_truth_risk.astype(int)
        
        y_pred = ciri_predictions
        
        # Calculate various metrics
        metrics = {}
        
        # Classification metrics
        metrics['auc_roc'] = self.calculate_auc(y_true, y_pred)
        metrics['auc_pr'] = self.calculate_precision_recall_auc(y_true, y_pred)
        
        precision, recall, f1 = self.calculate_precision_recall_f1(y_true, y_pred)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        metrics['false_alarm_rate'] = self.calculate_false_alarm_rate(y_true, y_pred)
        
        # Regression metrics if ground truth is continuous
        if len(np.unique(ground_truth_risk)) > 2:
            mae, mse = self.calculate_mae_mse(ground_truth_risk, ciri_predictions)
            metrics['mae'] = mae
            metrics['mse'] = mse
            metrics['correlation'] = self.calculate_correlation_coefficient(ground_truth_risk, ciri_predictions)
        
        # Accuracy metrics
        metrics['accuracy'] = self.calculate_pixel_accuracy(
            ciri_predictions, 
            ground_truth_risk if len(np.unique(ground_truth_risk)) <= 2 else (ground_truth_risk > risk_threshold).astype(float)
        )
        
        return metrics

    def evaluate_sequence_prediction(self, 
                                  predicted_sequences: List[np.ndarray], 
                                  true_sequences: List[np.ndarray],
                                  frame_interval: float = 0.1) -> Dict[str, float]:
        """
        Evaluate sequence prediction performance (temporal consistency)
        Args:
            predicted_sequences: List of predicted heatmaps for each frame
            true_sequences: List of true heatmaps for each frame
            frame_interval: Time interval between frames in seconds
        Returns:
            Dictionary of sequence evaluation metrics
        """
        if len(predicted_sequences) != len(true_sequences):
            raise ValueError("Predicted and true sequences must have the same length")
        
        # Calculate metrics for each frame and average
        all_aucs = []
        all_pixel_accuracies = []
        all_ious = []
        
        for pred, true in zip(predicted_sequences, true_sequences):
            # Flatten arrays for classification metrics
            y_pred_flat = pred.flatten()
            y_true_flat = true.flatten()
            
            # Calculate AUC
            try:
                auc_score = self.calculate_auc(y_true_flat, y_pred_flat)
                all_aucs.append(auc_score)
            except:
                all_aucs.append(0.5)  # Default to random classifier if calculation fails
            
            # Calculate pixel accuracy
            pix_acc = self.calculate_pixel_accuracy(pred, true)
            all_pixel_accuracies.append(pix_acc)
            
            # Calculate IoU
            iou = self.calculate_heatmap_iou(pred, true)
            all_ious.append(iou)
        
        # Aggregate metrics
        seq_metrics = {
            'mean_auc': np.mean(all_aucs),
            'std_auc': np.std(all_aucs),
            'mean_pixel_accuracy': np.mean(all_pixel_accuracies),
            'std_pixel_accuracy': np.std(all_pixel_accuracies),
            'mean_iou': np.mean(all_ious),
            'std_iou': np.std(all_ious),
            'temporal_consistency': self._calculate_temporal_consistency(predicted_sequences)
        }
        
        return seq_metrics

    def _calculate_temporal_consistency(self, sequences: List[np.ndarray]) -> float:
        """
        Calculate temporal consistency of predictions across frames
        Args:
            sequences: List of prediction arrays for consecutive frames
        Returns:
            Temporal consistency score
        """
        if len(sequences) < 2:
            return 1.0  # Perfect consistency for single frame
        
        diffs = []
        for i in range(1, len(sequences)):
            # Calculate the difference between consecutive frames
            diff = np.mean(np.abs(sequences[i].astype(float) - sequences[i-1].astype(float)))
            diffs.append(diff)
        
        # Lower mean difference indicates higher temporal consistency
        # Normalize to [0, 1] range (higher is better)
        max_possible_diff = 1.0  # Since our values are normalized to [0, 1]
        mean_diff = np.mean(diffs)
        consistency = max(0.0, 1.0 - mean_diff / max_possible_diff)
        
        return consistency

    def calculate_confidence_intervals(self, 
                                    metric_values: np.ndarray, 
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for a metric
        Args:
            metric_values: Array of metric values (e.g., from cross-validation)
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(metric_values, lower_percentile)
        upper_bound = np.percentile(metric_values, upper_percentile)
        
        return lower_bound, upper_bound


def evaluate_model_performance(predictions: Dict, targets: Dict) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance using all available metrics
    Args:
        predictions: Dictionary containing model predictions
        targets: Dictionary containing ground truth values
    Returns:
        Nested dictionary of evaluation results
    """
    evaluator = EvaluationMetrics()
    
    results = {}
    
    # Evaluate CIRI predictions
    if 'ciri_predictions' in predictions and 'ground_truth_risk' in targets:
        results['ciri_evaluation'] = evaluator.evaluate_risk_prediction(
            predictions['ciri_predictions'],
            targets['ground_truth_risk']
        )
    
    # Evaluate sequence predictions
    if 'predicted_sequences' in predictions and 'true_sequences' in targets:
        results['sequence_evaluation'] = evaluator.evaluate_sequence_prediction(
            predictions['predicted_sequences'],
            targets['true_sequences']
        )
    
    # Calculate additional metrics if specific arrays are provided
    if 'y_true' in targets and 'y_pred' in predictions:
        y_true = targets['y_true']
        y_pred = predictions['y_pred']
        
        results['classification_metrics'] = {
            'auc_roc': evaluator.calculate_auc(y_true, y_pred),
            'auc_pr': evaluator.calculate_precision_recall_auc(y_true, y_pred),
            'precision': evaluator.calculate_precision_recall_f1(y_true, y_pred)[0],
            'recall': evaluator.calculate_precision_recall_f1(y_true, y_pred)[1],
            'f1_score': evaluator.calculate_precision_recall_f1(y_true, y_pred)[2],
            'false_alarm_rate': evaluator.calculate_false_alarm_rate(y_true, y_pred)
        }
        
        # If ground truth is continuous, add regression metrics
        if len(np.unique(y_true)) > 2:
            mae, mse = evaluator.calculate_mae_mse(y_true, y_pred)
            results['regression_metrics'] = {
                'mae': mae,
                'mse': mse,
                'correlation': evaluator.calculate_correlation_coefficient(y_true, y_pred)
            }
    
    return results