import cv2
import numpy as np
import torch
from typing import Tuple, Optional

class OpticalFlowProcessor:
    def __init__(self, method='farneback'):
        """
        Initialize optical flow processor
        Args:
            method: 'farneback' for OpenCV implementation or 'raft' for RAFT (placeholder for now)
        """
        self.method = method
        if method == 'farneback':
            # Farneback parameters
            self.prev_frame = None
        elif method == 'raft':
            # In a real implementation, we would initialize RAFT here
            # For now, we'll use Farneback as fallback since RAFT requires additional dependencies
            print("RAFT not fully implemented in this version, using Farneback as default")
            self.method = 'farneback'
            self.prev_frame = None

    def compute_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> dict:
        """
        Compute optical flow between two frames
        Args:
            prev_frame: Previous frame as grayscale numpy array
            curr_frame: Current frame as grayscale numpy array
        Returns:
            Dictionary containing flow vectors, magnitude map, and direction clusters
        """
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame

        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame

        if self.method == 'farneback':
            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, 
                levels=3, 
                winsize=15, 
                iterations=3, 
                poly_n=5, 
                poly_sigma=1.2, 
                flags=0
            )

            # Compute magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Compute direction clusters using k-means
            direction_clusters = self._compute_direction_clusters(angle)

            return {
                'flow_vectors': flow,
                'magnitude_map': magnitude,
                'angle_map': angle,
                'direction_clusters': direction_clusters
            }

    def _compute_direction_clusters(self, angle_map: np.ndarray, k: int = 8) -> np.ndarray:
        """
        Compute direction clusters using k-means clustering on angles
        Args:
            angle_map: Angle map from optical flow
            k: Number of clusters
        Returns:
            Cluster labels for each pixel
        """
        from sklearn.cluster import KMeans
        
        # Reshape angle map to (height*width, 1) for clustering
        h, w = angle_map.shape
        angles_flat = angle_map.reshape(-1, 1)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(angles_flat)
        
        # Reshape back to original dimensions
        cluster_map = cluster_labels.reshape(h, w)
        
        return cluster_map

    def compute_velocity_variance(self, flow_vectors: np.ndarray, window_size: int = 32) -> np.ndarray:
        """
        Calculate local velocity variance in sliding windows
        Args:
            flow_vectors: Flow vectors from optical flow computation
            window_size: Size of the sliding window
        Returns:
            Velocity variance map
        """
        h, w, _ = flow_vectors.shape
        var_map = np.zeros((h, w))

        # Pad the flow vectors to handle border cases
        pad_size = window_size // 2
        padded_flow = np.pad(flow_vectors, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 
                             mode='constant', constant_values=0)

        for i in range(h):
            for j in range(w):
                # Extract window
                window = padded_flow[i:i+window_size, j:j+window_size, :]
                
                # Calculate velocity magnitude in window
                vel_mag = np.sqrt(window[:, :, 0]**2 + window[:, :, 1]**2)
                
                # Calculate variance
                var_map[i, j] = np.var(vel_mag)

        return var_map

    def detect_acceleration_spikes(self, flows: list) -> np.ndarray:
        """
        Detect acceleration spikes (frame-to-frame delta)
        Args:
            flows: List of flow vectors from consecutive frames
        Returns:
            Acceleration spike map
        """
        if len(flows) < 2:
            raise ValueError("Need at least 2 flow frames to compute acceleration")

        # Calculate difference between consecutive flow frames
        acc_spikes = []
        for i in range(1, len(flows)):
            # Calculate the difference in flow between consecutive frames
            flow_diff = np.abs(flows[i] - flows[i-1])
            # Sum the differences in x and y directions
            acc_spike = np.sqrt(np.sum(flow_diff**2, axis=2))
            acc_spikes.append(acc_spike)

        # Average the acceleration spikes over the sequence
        avg_acc_spikes = np.mean(acc_spikes, axis=0)
        return avg_acc_spikes

    def process_frame_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> dict:
        """
        Process a pair of frames to compute optical flow features
        Args:
            frame1: First frame
            frame2: Second frame
        Returns:
            Dictionary with all optical flow features
        """
        flow_data = self.compute_flow(frame1, frame2)
        
        velocity_var = self.compute_velocity_variance(flow_data['flow_vectors'])
        
        return {
            'flow_vectors': flow_data['flow_vectors'],
            'magnitude_map': flow_data['magnitude_map'],
            'angle_map': flow_data['angle_map'],
            'direction_clusters': flow_data['direction_clusters'],
            'velocity_variance': velocity_var
        }


class RAFTOpticalFlow:
    """
    Placeholder class for RAFT optical flow implementation.
    In a real implementation, this would interface with the RAFT model.
    """
    def __init__(self):
        print("RAFT optical flow would be implemented here with a PyTorch model.")
        print("For now, using OpenCV's Farneback method as a substitute.")
        # In a real implementation:
        # self.model = load_RAFT_model()
        # self.model.eval()

    def compute_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> dict:
        """
        Compute optical flow using RAFT method
        """
        # This is a simplified implementation using OpenCV as placeholder
        # Real RAFT would require loading a specific model
        processor = OpticalFlowProcessor(method='farneback')
        return processor.compute_flow(prev_frame, curr_frame)