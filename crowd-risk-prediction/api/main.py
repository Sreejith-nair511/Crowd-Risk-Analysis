from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import uvicorn
import numpy as np
import cv2
import torch
import uuid
import os
from datetime import datetime
import json
from PIL import Image

# Import our modules
from src.features.ciri_calculator import CIRICalculator
from src.models.ciri_model import CIRIPredictor, create_default_ciri_predictor
from src.models.csrnet import DensityEstimator
from src.features.optical_flow import OpticalFlowProcessor
from src.features.instability_features import InstabilityFeatures
from src.utils.visualization import visualize_heatmap

# Initialize the app
app = FastAPI(
    title="Crowd Risk Prediction API",
    description="API for predicting crowd instability using computer vision and spatio-temporal modeling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store models and analysis results
models = {}
analysis_cache = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models when the API starts"""
    global models
    
    print("Initializing models...")
    
    # Initialize CSRNet density estimator
    models['density_estimator'] = DensityEstimator()
    
    # Initialize optical flow processor
    models['optical_flow'] = OpticalFlowProcessor(method='farneback')
    
    # Initialize instability features calculator
    models['instability_features'] = InstabilityFeatures()
    
    # Initialize CIRI predictor
    models['ciri_predictor'] = create_default_ciri_predictor()
    
    print("Models initialized successfully!")

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for crowd risk analysis
    Returns a video ID for tracking the analysis
    """
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Uploaded file must be a video")
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Create upload directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(upload_dir, f"{video_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Store video info in cache
        analysis_cache[video_id] = {
            'file_path': file_path,
            'filename': file.filename,
            'upload_time': datetime.now().isoformat(),
            'status': 'uploaded',
            'total_frames': 0,
            'fps': 0,
            'duration': 0
        }
        
        # Get video properties
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        analysis_cache[video_id]['total_frames'] = total_frames
        analysis_cache[video_id]['fps'] = fps
        analysis_cache[video_id]['duration'] = duration
        
        cap.release()
        
        return {
            "video_id": video_id,
            "message": "Video uploaded successfully",
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading video: {str(e)}")

@app.get("/analyze-video/{video_id}")
async def analyze_video(video_id: str, start_frame: int = 0, end_frame: int = -1):
    """
    Analyze a video and compute crowd risk metrics
    """
    if video_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        video_info = analysis_cache[video_id]
        file_path = video_info['file_path']
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Update status
        analysis_cache[video_id]['status'] = 'analyzing'
        
        # Open video file
        cap = cv2.VideoCapture(file_path)
        
        # Set start frame if specified
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Determine end frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame == -1 or end_frame > total_frames:
            end_frame = total_frames
        
        # Initialize results storage
        results = {
            'frames_analyzed': [],
            'risk_timeline': [],
            'average_risk': 0.0,
            'max_risk': 0.0,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Prepare for sequence analysis
        frame_sequence = []
        feature_sequence = []
        
        for frame_idx in range(start_frame, min(end_frame, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate density map
            density_map = models['density_estimator'].estimate_density(frame_rgb)
            
            # Calculate optical flow (requires previous frame)
            if len(frame_sequence) > 0:
                flow_data = models['optical_flow'].process_frame_pair(frame_sequence[-1], frame_rgb)
                
                # Calculate instability features
                features = models['instability_features'].extract_all_features(
                    density_map, flow_data
                )
                
                # Create feature tensor for CIRI model
                feature_tensor = np.stack([
                    features['density_map'],
                    features['directional_entropy'],  # This needs to be a map, not scalar
                    features['foi_map'],
                    features['lmcs_map'],
                    features['density_grad_map'],
                    features['acceleration_spikes']
                ], axis=-1)
                
                # Convert to torch tensor
                feature_tensor = torch.from_numpy(feature_tensor).float().unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
                
                # Add to sequence
                feature_sequence.append(feature_tensor)
                
                # Calculate CIRI for this frame
                with torch.no_grad():
                    ciri_result = models['ciri_predictor'].predict_single_frame({
                        'density_map': torch.from_numpy(features['density_map']).float().unsqueeze(0).unsqueeze(-1),
                        'directional_entropy_map': torch.from_numpy(np.full_like(features['density_map'], features['directional_entropy'])).float().unsqueeze(0).unsqueeze(-1),
                        'foi_map': torch.from_numpy(features['foi_map']).float().unsqueeze(0).unsqueeze(-1),
                        'lmcs_map': torch.from_numpy(features['lmcs_map']).float().unsqueeze(0).unsqueeze(-1),
                        'density_grad_map': torch.from_numpy(features['density_grad_map']).float().unsqueeze(0).unsqueeze(-1),
                        'acceleration_spikes': torch.from_numpy(features['acceleration_spikes']).float().unsqueeze(0).unsqueeze(-1)
                    })
                    
                    ciri_value = float(torch.mean(ciri_result).item())
                
                # Store frame analysis
                frame_analysis = {
                    'frame_number': frame_idx,
                    'density_map_shape': density_map.shape,
                    'ciri_score': ciri_value,
                    'instability_features': {
                        'directional_entropy': float(features['directional_entropy']),
                        'avg_foi': float(np.mean(features['foi_map'])),
                        'avg_lmcs': float(np.mean(features['lmcs_map'])),
                        'avg_density_grad': float(np.mean(features['density_grad_map'])),
                        'avg_acceleration': float(np.mean(features['acceleration_spikes']))
                    }
                }
                
                results['frames_analyzed'].append(frame_analysis)
                results['risk_timeline'].append({
                    'frame': frame_idx,
                    'ciri_score': ciri_value
                })
            else:
                # For the first frame, we can't compute flow yet
                density_map_normalized = density_map / (np.max(density_map) + 1e-8)  # Avoid division by zero
                ciri_value = float(np.mean(density_map_normalized))  # Use density as proxy for first frame
                
                frame_analysis = {
                    'frame_number': frame_idx,
                    'density_map_shape': density_map.shape,
                    'ciri_score': ciri_value,
                    'instability_features': {
                        'directional_entropy': 0.0,  # Placeholder
                        'avg_foi': 0.0,
                        'avg_lmcs': 0.0,
                        'avg_density_grad': float(np.mean(models['instability_features'].calculate_density_gradient(density_map))),
                        'avg_acceleration': 0.0
                    }
                }
                
                results['frames_analyzed'].append(frame_analysis)
                results['risk_timeline'].append({
                    'frame': frame_idx,
                    'ciri_score': ciri_value
                })
            
            # Keep only the last frame for flow calculation
            frame_sequence = [frame_rgb] if len(frame_sequence) == 0 else [frame_rgb]
        
        cap.release()
        
        # Calculate aggregate metrics
        if results['risk_timeline']:
            ciri_scores = [item['ciri_score'] for item in results['risk_timeline']]
            results['average_risk'] = float(np.mean(ciri_scores))
            results['max_risk'] = float(np.max(ciri_scores))
        
        # Update cache with results
        analysis_cache[video_id]['analysis_results'] = results
        analysis_cache[video_id]['status'] = 'completed'
        
        return results
        
    except Exception as e:
        analysis_cache[video_id]['status'] = 'error'
        analysis_cache[video_id]['error'] = str(e)
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")

@app.get("/frame-analysis/{video_id}/{frame_num}")
async def get_frame_analysis(video_id: str, frame_num: int):
    """
    Get analysis for specific frame
    """
    if video_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if 'analysis_results' not in analysis_cache[video_id]:
        raise HTTPException(status_code=400, detail="Video has not been analyzed yet")
    
    results = analysis_cache[video_id]['analysis_results']
    
    # Find the requested frame
    frame_analysis = None
    for frame_data in results['frames_analyzed']:
        if frame_data['frame_number'] == frame_num:
            frame_analysis = frame_data
            break
    
    if frame_analysis is None:
        raise HTTPException(status_code=404, detail="Frame not found in analysis results")
    
    return frame_analysis

@app.get("/risk-heatmap/{video_id}/{frame_num}")
async def get_risk_heatmap(video_id: str, frame_num: int):
    """
    Get risk heatmap for specific frame as JSON
    """
    if video_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if 'analysis_results' not in analysis_cache[video_id]:
        raise HTTPException(status_code=400, detail="Video has not been analyzed yet")
    
    # In a real implementation, we would return the actual heatmap
    # For now, we'll return a mock response
    return {
        "video_id": video_id,
        "frame_num": frame_num,
        "heatmap_data_url": f"/heatmap-data/{video_id}/{frame_num}",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics/{video_id}")
async def get_metrics(video_id: str):
    """
    Get evaluation metrics for video analysis
    """
    if video_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if 'analysis_results' not in analysis_cache[video_id]:
        raise HTTPException(status_code=400, detail="Video has not been analyzed yet")
    
    results = analysis_cache[video_id]['analysis_results']
    
    # Calculate additional metrics
    metrics = {
        "video_id": video_id,
        "total_frames_analyzed": len(results['frames_analyzed']),
        "average_risk_score": results['average_risk'],
        "max_risk_score": results['max_risk'],
        "high_risk_frames": len([r for r in results['risk_timeline'] if r['ciri_score'] > 0.7]),
        "medium_risk_frames": len([r for r in results['risk_timeline'] if 0.3 <= r['ciri_score'] <= 0.7]),
        "low_risk_frames": len([r for r in results['risk_timeline'] if r['ciri_score'] < 0.3]),
        "analysis_duration": (datetime.fromisoformat(results['analysis_timestamp']) - 
                              datetime.fromisoformat(analysis_cache[video_id]['upload_time'])).total_seconds()
    }
    
    return metrics

@app.get("/status/{video_id}")
async def get_analysis_status(video_id: str):
    """
    Get the current status of video analysis
    """
    if video_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {
        "video_id": video_id,
        "status": analysis_cache[video_id]['status'],
        "upload_time": analysis_cache[video_id]['upload_time'],
        "total_frames": analysis_cache[video_id]['total_frames'],
        "progress": len(analysis_cache[video_id].get('analysis_results', {}).get('frames_analyzed', [])) / analysis_cache[video_id]['total_frames'] if analysis_cache[video_id]['total_frames'] > 0 else 0
    }

@app.get("/")
async def root():
    """
    Root endpoint to check API status
    """
    return {
        "message": "Crowd Risk Prediction API",
        "status": "running",
        "models_loaded": len(models) > 0,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)