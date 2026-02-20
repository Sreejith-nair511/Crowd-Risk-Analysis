import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from .transformer import SpatioTemporalTransformer

class CIRIModel(nn.Module):
    """
    Crowd Instability Risk Index (CIRI) Model
    Implements the weighted combination of instability features
    """
    def __init__(self, 
                 num_features: int = 6,  # D, H_d, FOI, LMCS, ∇D, Δv
                 use_mlp_weights: bool = True,
                 mlp_hidden_size: int = 32,
                 manual_weights: Optional[list] = None):
        super().__init__()
        
        self.num_features = num_features
        self.use_mlp_weights = use_mlp_weights
        
        if manual_weights is not None:
            # Use manually specified weights
            if len(manual_weights) != num_features:
                raise ValueError(f"Expected {num_features} manual weights, got {len(manual_weights)}")
            self.register_buffer('manual_weights', torch.tensor(manual_weights, dtype=torch.float32))
            self.learnable_weights = None
        else:
            # Learnable weights for the CIRI formula
            self.learnable_weights = nn.Parameter(torch.ones(num_features) / num_features)
            self.manual_weights = None
        
        # Alternative: Small MLP to learn feature combinations
        if use_mlp_weights:
            self.feature_combiner = nn.Sequential(
                nn.Linear(num_features, mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_size // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.feature_combiner = None

    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute CIRI = w1*D + w2*H_d + w3*FOI + w4*LMCS + w5*∇D + w6*Δv
        Args:
            features_dict: Dictionary containing all instability features
        Returns:
            CIRI map with values normalized to [0, 1]
        """
        # Extract features
        density = features_dict.get('density_map', features_dict.get('density', torch.zeros_like(features_dict[list(features_dict.keys())[0]])))
        directional_entropy = features_dict.get('directional_entropy_map', features_dict.get('entropy', torch.zeros_like(density)))
        foi = features_dict.get('foi_map', torch.zeros_like(density))
        lmcs = features_dict.get('lmcs_map', torch.zeros_like(density))
        density_grad = features_dict.get('density_grad_map', torch.zeros_like(density))
        accel_spikes = features_dict.get('acceleration_spikes', torch.zeros_like(density))
        
        # Stack all features along a new dimension
        stacked_features = torch.stack([density, directional_entropy, foi, lmcs, density_grad, accel_spikes], dim=-1)
        
        if self.feature_combiner is not None:
            # Use MLP to combine features
            batch_size, height, width, num_feats = stacked_features.shape
            # Reshape to (batch_size*height*width, num_features) for MLP
            reshaped_features = stacked_features.view(-1, num_feats)
            ciri_values = self.feature_combiner(reshaped_features)
            # Reshape back to spatial dimensions
            ciri_map = ciri_values.view(batch_size, height, width, 1).squeeze(-1)
        else:
            # Use weighted linear combination
            if self.manual_weights is not None:
                weights = self.manual_weights
            else:
                weights = torch.softmax(self.learnable_weights, dim=0)
            
            # Expand weights to match spatial dimensions
            expanded_weights = weights.view(1, 1, 1, -1).expand_as(stacked_features)
            
            # Compute weighted sum
            weighted_features = stacked_features * expanded_weights
            ciri_map = torch.sum(weighted_features, dim=-1)
            
            # Normalize to [0, 1] using sigmoid
            ciri_map = torch.sigmoid(ciri_map)
        
        return ciri_map

    def compute_ciri_with_weights(self, 
                                density: torch.Tensor,
                                directional_entropy: torch.Tensor,
                                foi: torch.Tensor,
                                lmcs: torch.Tensor,
                                density_grad: torch.Tensor,
                                accel_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute CIRI using individual feature tensors
        """
        features_dict = {
            'density_map': density,
            'directional_entropy_map': directional_entropy,
            'foi_map': foi,
            'lmcs_map': lmcs,
            'density_grad_map': density_grad,
            'acceleration_spikes': accel_spikes
        }
        return self.forward(features_dict)


class CIRIPredictor(nn.Module):
    """
    Complete CIRI prediction system that combines feature extraction and CIRI computation
    """
    def __init__(self, 
                 transformer_model: Optional[SpatioTemporalTransformer] = None,
                 use_temporal_prediction: bool = True):
        super().__init__()
        
        self.ciri_model = CIRIModel()
        self.use_temporal_prediction = use_temporal_prediction
        
        if transformer_model is not None:
            self.spatiotemporal_transformer = transformer_model
        else:
            self.spatiotemporal_transformer = SpatioTemporalTransformer(
                input_channels=6,  # density, entropy, foi, lmcs, density_grad, accel
                seq_length=8
            )
    
    def forward(self, 
                feature_sequence: torch.Tensor,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for CIRI prediction
        Args:
            feature_sequence: Sequence of feature tensors (B, T, H, W, C)
            return_intermediates: Whether to return intermediate results
        Returns:
            Dictionary with prediction results
        """
        batch_size, seq_len, height, width, channels = feature_sequence.shape
        
        # If using transformer for temporal prediction
        if self.use_temporal_prediction and seq_len > 1:
            current_risk, future_risk = self.spatiotemporal_transformer(feature_sequence)
        else:
            # Just process the last frame if not using temporal prediction
            last_frame_features = feature_sequence[:, -1:, :, :, :]  # (B, 1, H, W, C)
            # Reshape to (B, H, W, C) to pass to CIRI model
            reshaped_features = last_frame_features.squeeze(1)
            
            # Create features dict for CIRI model
            features_dict = {
                'density_map': reshaped_features[:, :, :, 0],
                'directional_entropy_map': reshaped_features[:, :, :, 1],
                'foi_map': reshaped_features[:, :, :, 2],
                'lmcs_map': reshaped_features[:, :, :, 3],
                'density_grad_map': reshaped_features[:, :, :, 4],
                'acceleration_spikes': reshaped_features[:, :, :, 5]
            }
            
            current_risk = self.ciri_model(features_dict).unsqueeze(-1)  # Add channel dim back
            future_risk = current_risk  # Same as current if not using temporal prediction
        
        results = {
            'current_risk': current_risk,
            'future_risk': future_risk
        }
        
        if return_intermediates:
            # Add intermediate computations if requested
            results['intermediate_features'] = feature_sequence
        
        return results

    def predict_single_frame(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict CIRI for a single frame given feature dictionary
        """
        return self.ciri_model(features_dict)


class CIRILoss(nn.Module):
    """
    Loss function for training CIRI model
    Combines multiple objectives for instability prediction
    """
    def __init__(self, 
                 future_weight: float = 0.5,
                 current_weight: float = 0.5,
                 stability_margin: float = 0.1):
        super().__init__()
        self.future_weight = future_weight
        self.current_weight = current_weight
        self.stability_margin = stability_margin
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, 
                predicted_current: torch.Tensor,
                predicted_future: torch.Tensor,
                target_current: torch.Tensor,
                target_future: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for CIRI predictions
        """
        # Current risk prediction loss
        current_loss = self.bce_loss(predicted_current, target_current)
        
        # Future risk prediction loss
        future_loss = self.bce_loss(predicted_future, target_future)
        
        # Combined loss
        total_loss = self.current_weight * current_loss + self.future_weight * future_loss
        
        return total_loss


def create_default_ciri_predictor() -> CIRIPredictor:
    """
    Create a default CIRI predictor with standard configuration
    """
    transformer = SpatioTemporalTransformer(
        input_channels=6,
        seq_length=8,
        embed_dim=256,
        num_heads=8,
        num_layers=6
    )
    
    return CIRIPredictor(transformer_model=transformer)