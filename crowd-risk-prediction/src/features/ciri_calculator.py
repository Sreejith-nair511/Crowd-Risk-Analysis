import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from ..models.ciri_model import CIRIModel
from ..features.instability_features import InstabilityFeatures

class CIRICalculator:
    """
    Calculator for Crowd Instability Risk Index (CIRI)
    Integrates all instability features into the final risk index
    """
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.ciri_model = CIRIModel()
        self.instability_features = InstabilityFeatures()
        
        # Load trained model if path provided
        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            self.ciri_model.load_state_dict(checkpoint)
        
        self.ciri_model.to(device)
        self.ciri_model.eval()

    def calculate_ciri(self, 
                      density_map: np.ndarray,
                      flow_data: Dict,
                      velocity_fields: List[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Calculate Crowd Instability Risk Index for a single frame
        Args:
            density_map: Normalized density heatmap
            flow_data: Dictionary containing flow vectors, magnitude, angles, etc.
            velocity_fields: List of velocity magnitude maps from consecutive frames
        Returns:
            Tuple of (CIRI map, feature breakdown dict)
        """
        # Extract all instability features
        features_dict = self.instability_features.extract_all_features(
            density_map, flow_data, velocity_fields
        )
        
        # Convert numpy arrays to torch tensors
        torch_features = {}
        for key, value in features_dict.items():
            if isinstance(value, np.ndarray):
                torch_features[key] = torch.from_numpy(value).float().to(self.device)
            else:
                # For scalar values like directional_entropy, expand to match spatial dimensions
                if key == 'directional_entropy':
                    # Create a map filled with the entropy value
                    entropy_map = np.full(density_map.shape, value, dtype=np.float32)
                    torch_features[f'{key}_map'] = torch.from_numpy(entropy_map).float().to(self.device)
                else:
                    torch_features[key] = torch.from_numpy(np.array(value)).float().to(self.device)
        
        # Ensure all feature maps have the same spatial dimensions
        spatial_dims = density_map.shape
        for key, tensor in torch_features.items():
            if tensor.dim() == 0:  # Scalar
                torch_features[key] = tensor.expand(spatial_dims[0], spatial_dims[1])
            elif tensor.dim() == 2:  # Already a 2D map
                pass  # Correct dimension
            elif tensor.dim() == 3:  # Has extra dimension
                if tensor.shape[0] == 1:
                    torch_features[key] = tensor.squeeze(0)
        
        # Add batch dimension to all tensors
        for key in torch_features:
            if torch_features[key].dim() == 2:
                torch_features[key] = torch_features[key].unsqueeze(0).unsqueeze(-1)  # Add batch and channel dims
        
        # Calculate CIRI using the model
        with torch.no_grad():
            ciri_map = self.ciri_model(torch_features).squeeze(-1)  # Remove channel dimension
            ciri_map_np = ciri_map.cpu().numpy()
        
        # Prepare feature breakdown
        feature_breakdown = {
            'density_contribution': float(np.mean(density_map)),
            'directional_entropy': features_dict['directional_entropy'],
            'avg_foi': float(np.mean(features_dict['foi_map'])),
            'avg_lmcs': float(np.mean(features_dict['lmcs_map'])),
            'avg_density_gradient': float(np.mean(features_dict['density_grad_map'])),
            'avg_acceleration': float(np.mean(features_dict['acceleration_spikes']))
        }
        
        return ciri_map_np, feature_breakdown

    def calculate_sequence_ciri(self, 
                              density_maps: List[np.ndarray],
                              flow_data_list: List[Dict],
                              window_size: int = 8) -> List[Tuple[np.ndarray, Dict]]:
        """
        Calculate CIRI for a sequence of frames using sliding window approach
        Args:
            density_maps: List of density heatmaps for consecutive frames
            flow_data_list: List of flow data dictionaries for consecutive frames
            window_size: Number of frames to consider for temporal analysis
        Returns:
            List of (CIRI map, feature breakdown) tuples for each frame
        """
        results = []
        
        for i in range(len(density_maps)):
            # Determine the window for velocity fields (for acceleration calculation)
            start_idx = max(0, i - window_size + 1)
            velocity_fields = []
            for j in range(start_idx, i + 1):
                if j < len(flow_data_list):
                    velocity_fields.append(flow_data_list[j]['magnitude_map'])
            
            # Calculate CIRI for current frame
            ciri_map, feature_breakdown = self.calculate_ciri(
                density_maps[i], 
                flow_data_list[i], 
                velocity_fields
            )
            
            results.append((ciri_map, feature_breakdown))
        
        return results

    def get_risk_level(self, ciri_value: float) -> str:
        """
        Convert CIRI value to risk level
        Args:
            ciri_value: CIRI value between 0 and 1
        Returns:
            Risk level as string
        """
        if ciri_value < 0.3:
            return "LOW"
        elif ciri_value < 0.6:
            return "MODERATE"
        elif ciri_value < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"

    def calculate_aggregate_metrics(self, ciri_sequence: List[np.ndarray]) -> Dict:
        """
        Calculate aggregate metrics from a sequence of CIRI maps
        Args:
            ciri_sequence: List of CIRI maps
        Returns:
            Dictionary of aggregate metrics
        """
        if not ciri_sequence:
            return {}
        
        # Flatten all CIRI maps to calculate statistics
        all_values = np.concatenate([ciri_map.flatten() for ciri_map in ciri_sequence])
        
        # Calculate metrics
        metrics = {
            'mean_ciri': float(np.mean(all_values)),
            'std_ciri': float(np.std(all_values)),
            'min_ciri': float(np.min(all_values)),
            'max_ciri': float(np.max(all_values)),
            'median_ciri': float(np.median(all_values)),
            'percentile_95': float(np.percentile(all_values, 95)),
            'percentile_99': float(np.percentile(all_values, 99)),
            'high_risk_percentage': float(np.sum(all_values > 0.7) / len(all_values) * 100),
            'critical_risk_percentage': float(np.sum(all_values > 0.9) / len(all_values) * 100)
        }
        
        return metrics

    def detect_instability_events(self, ciri_sequence: List[np.ndarray], 
                                threshold: float = 0.7, 
                                min_duration: int = 3) -> List[Dict]:
        """
        Detect periods of high instability in the sequence
        Args:
            ciri_sequence: List of CIRI maps
            threshold: Threshold for considering a region as unstable
            min_duration: Minimum number of consecutive frames for an event
        Returns:
            List of detected instability events
        """
        events = []
        current_event = None
        
        for frame_idx, ciri_map in enumerate(ciri_sequence):
            # Check if any region exceeds the threshold
            high_risk_regions = ciri_map > threshold
            has_high_risk = np.any(high_risk_regions)
            
            if has_high_risk and current_event is None:
                # Start new event
                current_event = {
                    'start_frame': frame_idx,
                    'peak_ciri': float(np.max(ciri_map)),
                    'high_risk_area_ratio': float(np.sum(high_risk_regions) / high_risk_regions.size)
                }
            elif not has_high_risk and current_event is not None:
                # End current event if it meets minimum duration
                current_event['end_frame'] = frame_idx - 1
                current_event['duration'] = current_event['end_frame'] - current_event['start_frame'] + 1
                
                if current_event['duration'] >= min_duration:
                    events.append(current_event)
                
                current_event = None
            elif has_high_risk and current_event is not None:
                # Update peak value for ongoing event
                current_event['peak_ciri'] = max(current_event['peak_ciri'], float(np.max(ciri_map)))
                current_event['high_risk_area_ratio'] = max(
                    current_event['high_risk_area_ratio'],
                    float(np.sum(high_risk_regions) / high_risk_regions.size)
                )
        
        # Handle event that goes to the end of sequence
        if current_event is not None:
            current_event['end_frame'] = len(ciri_sequence) - 1
            current_event['duration'] = current_event['end_frame'] - current_event['start_frame'] + 1
            
            if current_event['duration'] >= min_duration:
                events.append(current_event)
        
        return events