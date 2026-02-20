import numpy as np
import torch
from scipy.ndimage import sobel
from scipy.stats import entropy
from sklearn.cluster import KMeans
from typing import Dict, Tuple
import cv2

class InstabilityFeatures:
    def __init__(self):
        pass

    def calculate_directional_entropy(self, flow_angles: np.ndarray, bins: int = 16) -> float:
        """
        Calculate directional entropy (H_d) over quantized motion directions
        Args:
            flow_angles: Array of flow angles in radians
            bins: Number of bins for histogram quantization
        Returns:
            Directional entropy value
        """
        # Flatten the angle array and remove any NaN values
        flat_angles = flow_angles.flatten()
        flat_angles = flat_angles[~np.isnan(flat_angles)]
        
        if len(flat_angles) == 0:
            return 0.0
            
        # Create histogram of flow angles
        hist, _ = np.histogram(flat_angles, bins=bins, range=(0, 2*np.pi))
        
        # Normalize histogram to get probability distribution
        hist = hist.astype(float)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        
        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        
        # Calculate entropy
        if len(hist) > 0:
            directional_entropy = -np.sum(hist * np.log(hist + 1e-10))  # Adding small epsilon to prevent log(0)
        else:
            directional_entropy = 0.0
            
        return directional_entropy

    def calculate_flow_opposition_index(self, flow_vectors: np.ndarray, window_size: int = 32) -> np.ndarray:
        """
        Calculate percentage of vectors opposing dominant direction in local window
        Args:
            flow_vectors: Array of flow vectors (H, W, 2)
            window_size: Size of local window for analysis
        Returns:
            FOI map with same dimensions as input
        """
        h, w, _ = flow_vectors.shape
        foi_map = np.zeros((h, w))
        
        # Calculate dominant direction globally
        global_avg_flow = np.mean(flow_vectors, axis=(0, 1))
        global_angle = np.arctan2(global_avg_flow[1], global_avg_flow[0])
        
        # Pad the flow vectors to handle border cases
        pad_size = window_size // 2
        padded_flow = np.pad(flow_vectors, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 
                             mode='constant', constant_values=0)
        
        for i in range(h):
            for j in range(w):
                # Extract local window
                window = padded_flow[i:i+window_size, j:j+window_size, :]
                
                # Calculate local average flow direction
                local_avg_flow = np.mean(window, axis=(0, 1))
                local_angle = np.arctan2(local_avg_flow[1], local_avg_flow[0])
                
                # Calculate opposition: vectors that are roughly 180 degrees from local dominant direction
                local_flows = window.reshape(-1, 2)
                
                # Calculate angles of all vectors in the window
                vector_angles = np.arctan2(local_flows[:, 1], local_flows[:, 0])
                
                # Find vectors that oppose the local dominant direction (within 45 degrees of opposite)
                opposition_threshold = np.pi / 4  # 45 degrees tolerance
                opposite_direction = local_angle + np.pi
                
                # Normalize angles to [-pi, pi]
                vector_angles = (vector_angles + np.pi) % (2 * np.pi) - np.pi
                opposite_direction = (opposite_direction + np.pi) % (2 * np.pi) - np.pi
                
                # Calculate angular differences
                angle_diffs = np.abs(vector_angles - opposite_direction)
                angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
                
                # Count vectors that are opposing
                opposing_count = np.sum(angle_diffs <= opposition_threshold)
                total_count = len(vector_angles)
                
                foi_map[i, j] = opposing_count / total_count if total_count > 0 else 0.0
        
        return foi_map

    def calculate_local_motion_compression_score(self, density_map: np.ndarray, 
                                               velocity_magnitude: np.ndarray,
                                               alpha: float = 0.7) -> np.ndarray:
        """
        Calculate LMCS: High density gradient + decreasing velocity magnitude
        Args:
            density_map: Normalized density map
            velocity_magnitude: Velocity magnitude map
            alpha: Weight for combining density and velocity components
        Returns:
            LMCS map
        """
        # Calculate density gradient
        density_grad_x = sobel(density_map, axis=1)
        density_grad_y = sobel(density_map, axis=0)
        density_gradient = np.sqrt(density_grad_x**2 + density_grad_y**2)
        
        # Calculate velocity gradient (areas where velocity decreases)
        vel_grad_x = sobel(velocity_magnitude, axis=1)
        vel_grad_y = sobel(velocity_magnitude, axis=0)
        velocity_gradient = np.sqrt(vel_grad_x**2 + vel_grad_y**2)
        
        # Areas of high density gradient and low velocity (compression zones)
        # We want areas where density is high AND velocity is low
        # Normalize both maps to [0, 1] range
        density_gradient_norm = density_gradient / (np.max(density_gradient) + 1e-10)
        velocity_magnitude_norm = velocity_magnitude / (np.max(velocity_magnitude) + 1e-10)
        
        # Low velocity areas contribute more to compression
        low_velocity_weight = 1 - velocity_magnitude_norm
        
        # Combine density gradient and inverse velocity
        lmcs_map = alpha * density_gradient_norm + (1 - alpha) * low_velocity_weight
        
        return lmcs_map

    def calculate_density_gradient(self, density_map: np.ndarray) -> np.ndarray:
        """
        Calculate spatial gradient magnitude of density map
        Args:
            density_map: Normalized density map
        Returns:
            Density gradient magnitude map
        """
        grad_x = sobel(density_map, axis=1)
        grad_y = sobel(density_map, axis=0)
        density_gradient = np.sqrt(grad_x**2 + grad_y**2)
        return density_gradient

    def calculate_acceleration_spike_map(self, velocity_fields: list) -> np.ndarray:
        """
        Calculate temporal velocity changes (Δv) - acceleration spikes
        Args:
            velocity_fields: List of velocity magnitude maps from consecutive frames
        Returns:
            Acceleration spike map
        """
        if len(velocity_fields) < 2:
            raise ValueError("Need at least 2 velocity fields to compute acceleration")
        
        # Calculate acceleration as change in velocity between frames
        acceleration_maps = []
        for i in range(1, len(velocity_fields)):
            # Difference in velocity magnitude between consecutive frames
            vel_diff = velocity_fields[i] - velocity_fields[i-1]
            # Take absolute value to get magnitude of change
            abs_vel_diff = np.abs(vel_diff)
            acceleration_maps.append(abs_vel_diff)
        
        # Average the acceleration maps to get a single map
        avg_acceleration = np.mean(acceleration_maps, axis=0)
        
        # Normalize to [0, 1] range
        if np.max(avg_acceleration) > 0:
            avg_acceleration = avg_acceleration / np.max(avg_acceleration)
        
        return avg_acceleration

    def extract_all_features(self, 
                           density_map: np.ndarray, 
                           flow_data: Dict,
                           velocity_fields: list = None) -> Dict:
        """
        Extract all instability features from density map and flow data
        Args:
            density_map: Normalized density heatmap
            flow_data: Dictionary containing flow vectors, magnitude, etc.
            velocity_fields: List of velocity magnitude maps from consecutive frames
        Returns:
            Dictionary containing all extracted features
        """
        # Calculate directional entropy
        directional_entropy = self.calculate_directional_entropy(flow_data['angle_map'])
        
        # Calculate flow opposition index
        foi_map = self.calculate_flow_opposition_index(flow_data['flow_vectors'])
        
        # Calculate velocity magnitude from flow
        velocity_magnitude = flow_data['magnitude_map']
        
        # Calculate local motion compression score
        lmcs_map = self.calculate_local_motion_compression_score(density_map, velocity_magnitude)
        
        # Calculate density gradient
        density_grad_map = self.calculate_density_gradient(density_map)
        
        # Calculate acceleration spike map
        if velocity_fields is None:
            velocity_fields = [velocity_magnitude]  # Use current field if not provided
        acceleration_spikes = self.calculate_acceleration_spike_map(velocity_fields)
        
        return {
            'directional_entropy': directional_entropy,
            'foi_map': foi_map,
            'lmcs_map': lmcs_map,
            'density_grad_map': density_grad_map,
            'acceleration_spikes': acceleration_spikes,
            'density_map': density_map
        }

    def compute_instability_mask(self, features_dict: Dict, 
                               thresholds: Dict = None) -> np.ndarray:
        """
        Compute binary mask highlighting unstable regions based on features
        Args:
            features_dict: Dictionary containing all extracted features
            thresholds: Dictionary of thresholds for each feature (optional)
        Returns:
            Binary mask indicating unstable regions
        """
        if thresholds is None:
            # Default thresholds based on empirical observations
            thresholds = {
                'foi_threshold': 0.3,
                'lmcs_threshold': 0.5,
                'density_grad_threshold': 0.4,
                'acceleration_threshold': 0.3
            }
        
        # Combine feature maps based on thresholds
        foi_mask = features_dict['foi_map'] > thresholds['foi_threshold']
        lmcs_mask = features_dict['lmcs_map'] > thresholds['lmcs_threshold']
        density_grad_mask = features_dict['density_grad_map'] > thresholds['density_grad_threshold']
        accel_mask = features_dict['acceleration_spikes'] > thresholds['acceleration_threshold']
        
        # Combine masks - instability occurs when multiple features align
        combined_mask = np.logical_or.reduce((
            foi_mask,
            lmcs_mask,
            density_grad_mask,
            accel_mask
        ))
        
        return combined_mask.astype(np.float32)