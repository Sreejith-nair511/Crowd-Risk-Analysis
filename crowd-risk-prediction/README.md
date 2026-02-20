# Vision-Based Early Crowd Collapse Risk Prediction via Spatio-Temporal Instability Modeling

## Project Overview

This project implements a real-time computer vision system that predicts crowd instability 2-5 seconds before potential collapse using density modeling, optical flow, and spatio-temporal transformers. The system computes a novel Crowd Instability Risk Index (CIRI) and generates a live risk heatmap overlay.

## Features

- **CSRNet-based density estimation** for crowd counting and density mapping
- **Optical flow analysis** using RAFT and Farneback algorithms
- **Instability feature extraction** including directional entropy, flow opposition index, and local motion compression score
- **Spatio-temporal transformer** for temporal pattern recognition
- **Crowd Instability Risk Index (CIRI)** combining multiple instability indicators
- **Real-time dashboard** with heatmap overlays and risk timelines
- **Synthetic scenario generator** for stress-testing and training
- **Comprehensive evaluation suite** with multiple metrics

## Mathematical Formulation

The Crowd Instability Risk Index (CIRI) is computed as:

```
CIRI = w1·D + w2·H_d + w3·FOI + w4·LMCS + w5·∇D + w6·Δv
```

Where:
- `D` = Density map
- `H_d` = Directional Entropy
- `FOI` = Flow Opposition Index
- `LMCS` = Local Motion Compression Score
- `∇D` = Density Gradient
- `Δv` = Acceleration Spike Map
- `w1-w6` = Learnable weights

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended for optimal performance)
- 16GB RAM or more

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd crowd-risk-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Node.js dependencies for the frontend:
```bash
cd frontend
npm install
cd ..
```

## Usage

### Running the API Server

```bash
cd crowd-risk-prediction
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Running the Frontend

```bash
cd frontend
npm run dev
```

### Training the Model

```bash
python experiments/train_ciri.py --config configs/training_config.yaml
```

### Evaluating the Model

```bash
python experiments/evaluate_model.py --model-path outputs/ciri_model_best.pth
```

### Running Ablation Studies

```bash
python experiments/ablation_study.py
```

## Architecture

### Backend (FastAPI)
- `/upload-video/` - Upload video files for analysis
- `/analyze-video/{video_id}` - Analyze uploaded videos
- `/frame-analysis/{video_id}/{frame_num}` - Get analysis for specific frame
- `/risk-heatmap/{video_id}/{frame_num}` - Get risk heatmap for specific frame
- `/metrics/{video_id}` - Get evaluation metrics for video

### Frontend (React)
- Dashboard with video player and heatmap overlay
- Risk timeline visualization
- Control panel for analysis modes
- Upload page for video submission

### Core Modules
- `src/models/csrnet.py` - Density estimation using CSRNet
- `src/features/optical_flow.py` - Optical flow computation
- `src/features/instability_features.py` - Instability feature extraction
- `src/models/transformer.py` - Spatio-temporal transformer
- `src/models/ciri_model.py` - CIRI calculation and prediction

## Configuration

The system can be configured using YAML files in the `configs/` directory:

- `model_config.yaml` - Model architecture and hyperparameters
- `training_config.yaml` - Training procedure and parameters

## Docker Deployment

To run the application using Docker:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000` and the frontend at `http://localhost`.

## Evaluation Metrics

The system implements the following evaluation metrics:

- AUC (Area Under the Curve)
- Precision/Recall
- False Alarm Rate
- Time-to-Collapse Detection Accuracy
- Heatmap IoU (Intersection over Union)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

## Synthetic Data Generation

The system includes a synthetic scenario generator that can create:

- Bidirectional crowd flow scenarios
- Bottleneck compression simulations
- Stress-test videos with instability patterns
- Augmented real dataset with instability patterns

## Research Output

The system generates:

- LaTeX-ready equations for the CIRI formulation
- Architecture diagrams using Mermaid
- Experiment configuration templates
- Ablation study support
- Baseline comparison scripts

## Performance Optimizations

- Efficient inference targeting <100ms per frame
- GPU acceleration support
- Batch processing capabilities
- Memory-efficient implementations
- Configurable performance parameters

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{crowd-risk-prediction,
  title={Vision-Based Early Crowd Collapse Risk Prediction via Spatio-Temporal Instability Modeling},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2026}
}
```

## Acknowledgments

- The CSRNet implementation is adapted from the original paper
- Some optical flow techniques are based on OpenCV implementations
- The transformer architecture is inspired by recent advances in spatio-temporal modeling