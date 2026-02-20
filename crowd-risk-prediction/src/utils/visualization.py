import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple, Optional
import torch
import io
from PIL import Image

def visualize_heatmap(heatmap: np.ndarray, 
                    title: str = "Risk Heatmap", 
                    colormap: str = 'jet',
                    save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize a heatmap using matplotlib
    Args:
        heatmap: 2D numpy array representing the heatmap
        title: Title for the visualization
        colormap: Matplotlib colormap to use
        save_path: Optional path to save the visualization
    Returns:
        RGB image as numpy array
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap=colormap, interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Risk Level')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Convert plot to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close()
    
    return img_array

def overlay_heatmap_on_frame(frame: np.ndarray, 
                           heatmap: np.ndarray, 
                           alpha: float = 0.6,
                           colormap: str = 'jet') -> np.ndarray:
    """
    Overlay a heatmap on top of a video frame
    Args:
        frame: Original video frame (H, W, 3) in RGB format
        heatmap: Heatmap to overlay (H, W)
        alpha: Transparency factor for overlay
        colormap: Colormap for heatmap visualization
    Returns:
        Frame with overlaid heatmap
    """
    # Resize heatmap to match frame dimensions if needed
    if heatmap.shape[:2] != frame.shape[:2]:
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    else:
        heatmap_resized = heatmap
    
    # Normalize heatmap to [0, 1]
    heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8)
    
    # Apply colormap to heatmap
    colormap_func = plt.cm.get_cmap(colormap)
    colored_heatmap = colormap_func(heatmap_normalized)
    colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel and convert to uint8
    
    # Convert original frame from RGB to BGR if needed (OpenCV uses BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if np.allclose(frame[:,:,0], frame[:,:,0]) else frame
    else:
        frame_bgr = frame
    
    # Blend the original frame with the colored heatmap
    overlay = cv2.addWeighted(frame_bgr, 1 - alpha, colored_heatmap, alpha, 0)
    
    # Convert back to RGB
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb

def create_risk_timeline(risk_scores: List[float], 
                        timestamps: Optional[List] = None,
                        title: str = "Risk Over Time") -> np.ndarray:
    """
    Create a timeline plot of risk scores over time/frames
    Args:
        risk_scores: List of risk scores
        timestamps: Optional list of timestamps for x-axis
        title: Title for the plot
    Returns:
        Plot as numpy array
    """
    plt.figure(figsize=(12, 6))
    
    x_axis = range(len(risk_scores)) if timestamps is None else timestamps
    plt.plot(x_axis, risk_scores, linewidth=2, color='red')
    
    # Add horizontal lines for risk level thresholds
    plt.axhline(y=0.3, color='green', linestyle='--', label='Low Risk', alpha=0.7)
    plt.axhline(y=0.6, color='orange', linestyle='--', label='Moderate Risk', alpha=0.7)
    plt.axhline(y=0.8, color='red', linestyle='--', label='High Risk', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Frame Number' if timestamps is None else 'Time')
    plt.ylabel('Risk Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Convert plot to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close()
    
    return img_array

def create_feature_comparison_grid(heatmaps: Dict[str, np.ndarray], 
                                 titles: Dict[str, str],
                                 cols: int = 2) -> np.ndarray:
    """
    Create a grid comparing different feature heatmaps
    Args:
        heatmaps: Dictionary mapping feature names to their heatmaps
        titles: Dictionary mapping feature names to their display titles
        cols: Number of columns in the grid
    Returns:
        Grid visualization as numpy array
    """
    rows = int(np.ceil(len(heatmaps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    
    # Handle case where there's only one subplot
    if len(heatmaps) == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if len(heatmaps) > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for idx, (feature_name, heatmap) in enumerate(heatmaps.items()):
        axes[idx].imshow(heatmap, cmap='jet', interpolation='nearest')
        axes[idx].set_title(titles[feature_name])
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(heatmaps), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close()
    
    return img_array

def generate_report_visualizations(analysis_results: Dict) -> Dict[str, np.ndarray]:
    """
    Generate a set of visualizations for a complete analysis report
    Args:
        analysis_results: Dictionary containing analysis results
    Returns:
        Dictionary mapping visualization names to their images
    """
    visualizations = {}
    
    # Risk timeline
    if 'risk_timeline' in analysis_results:
        risk_scores = [item['ciri_score'] for item in analysis_results['risk_timeline']]
        visualizations['risk_timeline'] = create_risk_timeline(risk_scores)
    
    # Sample heatmaps if available
    if 'frames_analyzed' in analysis_results and analysis_results['frames_analyzed']:
        sample_frame = analysis_results['frames_analyzed'][0]
        # Here we'd need to have the actual heatmaps stored in the results
        # For now, we'll just note that this would generate the visualizations
    
    return visualizations

def draw_risk_zones(frame: np.ndarray, 
                   ciri_map: np.ndarray, 
                   threshold: float = 0.7,
                   color: Tuple[int, int, int] = (0, 0, 255),
                   thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes around high-risk zones on the frame
    Args:
        frame: Input frame to draw on
        ciri_map: CIRI map for the same frame
        threshold: Threshold for identifying high-risk zones
        color: Color for drawing (BGR format)
        thickness: Thickness of drawn contours
    Returns:
        Frame with drawn risk zones
    """
    # Find regions with high CIRI values
    high_risk_mask = (ciri_map > threshold).astype(np.uint8)
    
    # Find contours of high-risk regions
    contours, _ = cv2.findContours(high_risk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the frame
    frame_with_zones = frame.copy()
    cv2.drawContours(frame_with_zones, contours, -1, color, thickness)
    
    # Also draw bounding rectangles for each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame_with_zones, (x, y), (x+w, y+h), color, thickness)
    
    return frame_with_zones

def create_risk_colormap() -> np.ndarray:
    """
    Create a custom colormap for risk visualization
    Returns:
        Custom colormap as numpy array
    """
    # Define colors for different risk levels: green -> yellow -> orange -> red -> purple
    colors = np.array([
        [0, 255, 0],    # Green (low risk)
        [128, 255, 0],  # Yellow-green
        [255, 255, 0],  # Yellow
        [255, 128, 0],  # Orange
        [255, 0, 0],    # Red
        [255, 0, 128],  # Red-purple
        [255, 0, 255]   # Purple (high risk)
    ], dtype=np.uint8)
    
    # Interpolate between these colors to create a smooth colormap
    colormap = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        ratio = i / 255.0
        # Map 0-1 to color segments
        segment_size = 1.0 / (len(colors) - 1)
        segment_idx = int(ratio / segment_size)
        local_ratio = (ratio % segment_size) / segment_size if segment_size > 0 else 0
        
        if segment_idx >= len(colors) - 1:
            colormap[i] = colors[-1]
        else:
            # Interpolate between two colors
            c1 = colors[segment_idx]
            c2 = colors[segment_idx + 1]
            interpolated = c1 * (1 - local_ratio) + c2 * local_ratio
            colormap[i] = np.clip(interpolated, 0, 255).astype(np.uint8)
    
    return colormap

def apply_risk_colormap(heatmap: np.ndarray) -> np.ndarray:
    """
    Apply the custom risk colormap to a normalized heatmap
    Args:
        heatmap: Normalized heatmap with values in [0, 1]
    Returns:
        Colored heatmap as RGB image
    """
    # Ensure heatmap is in [0, 1] range
    heatmap = np.clip(heatmap, 0, 1)
    
    # Scale to [0, 255]
    scaled_heatmap = (heatmap * 255).astype(np.uint8)
    
    # Get the custom colormap
    colormap = create_risk_colormap()
    
    # Apply colormap
    colored_heatmap = colormap[scaled_heatmap]
    
    return colored_heatmap