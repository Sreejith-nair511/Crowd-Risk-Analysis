# Crowd Risk Analysis: Real-Time Crowd Collapse Prediction System

<div align="center">

![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.12+-red)
![License](https://img.shields.io/badge/license-MIT-green)

**Vision-Based Early Crowd Collapse Risk Prediction via Spatio-Temporal Instability Modeling**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture)

</div>

---

## Project Overview

This project implements a **real-time computer vision system** that predicts crowd instability **2-5 seconds before potential collapse** using density modeling, optical flow, and spatio-temporal transformers. The system computes a novel **Crowd Instability Risk Index (CIRI)** and generates live risk heatmap overlays for immediate threat detection.

### Problem Statement
Large-scale crowd events face significant risks of stampedes and collapses. Our system provides **early warning detection** to prevent casualties by identifying instability patterns before they escalate to critical levels.

---

## Features

- **CSRNet-based Density Estimation** - Accurate crowd counting and density mapping
- **Optical Flow Analysis** - RAFT and Farneback algorithms for motion detection
- **Instability Feature Extraction** - Directional entropy, flow opposition index, motion compression
- **Spatio-Temporal Transformer** - Deep learning-based temporal pattern recognition
- **Crowd Instability Risk Index (CIRI)** - Novel risk metric combining multiple indicators
- **Real-Time Dashboard** - Live heatmap overlays and risk timelines
- **Synthetic Scenario Generator** - Stress-testing and training data generation
- **Comprehensive Evaluation Suite** - Multiple validation metrics and benchmarks
- **Docker Support** - Easy containerization and deployment
- **GPU Optimized** - High-performance inference <100ms/frame

## Mathematical Formulation

The **Crowd Instability Risk Index (CIRI)** is computed as:

$$\text{CIRI} = w_1 \cdot D + w_2 \cdot H_d + w_3 \cdot \text{FOI} + w_4 \cdot \text{LMCS} + w_5 \cdot \nabla D + w_6 \cdot \Delta v$$

### Components:
| Symbol | Description |
|--------|-------------|
| D | Crowd density map |
| H_d | Directional entropy of motion |
| FOI | Flow Opposition Index |
| LMCS | Local Motion Compression Score |
| ∇D | Density gradient |
| Δv | Acceleration spike map |
| w₁-w₆ | Learnable weights |

**Output**: Risk score in [0, 1] | **Threshold**: Alert triggered at CIRI > 0.7

## Installation

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.8+ | Recommended 3.9 or 3.10 |
| PyTorch | 1.12+ | With CUDA support recommended |
| GPU Memory | 8GB+ | For efficient inference |
| System RAM | 16GB+ | Recommended for training |

### Setup

**Step 1: Clone the repository**
```bash
git clone https://github.com/Sreejith-nair511/Crowd-Risk-Analysis.git
cd crowd-risk-prediction
```

**Step 2: Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 3: Install Python dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

**Step 5: Download pre-trained models (optional)**
```bash
python scripts/download_models.py
```

## Usage

### Running the API Server

```bash
cd crowd-risk-prediction
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API available at: http://localhost:8000
Swagger Docs: http://localhost:8000/docs

### Running the Frontend

```bash
cd frontend
npm run dev
```
Dashboard available at: http://localhost:5173

### Training the Model

```bash
python experiments/train_ciri.py --config configs/training_config.yaml
```
Training logs saved to `outputs/logs/`

### Evaluating the Model

```bash
python experiments/evaluate_model.py --model-path outputs/ciri_model_best.pth
```

### Running Ablation Studies

```bash
python experiments/ablation_study.py --config configs/training_config.yaml
```

### Docker Deployment

```bash
docker-compose up --build
```
Full stack running in containers!
- API: http://localhost:8000
- Frontend: http://localhost

## System Architecture

### Backend (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| /upload-video/ | POST | Upload video files for analysis |
| /analyze-video/{video_id} | POST | Analyze uploaded videos |
| /frame-analysis/{video_id}/{frame_num} | GET | Get analysis for specific frame |
| /risk-heatmap/{video_id}/{frame_num} | GET | Get risk heatmap for specific frame |
| /metrics/{video_id} | GET | Get evaluation metrics for video |
| /health | GET | System health check |

### Frontend (React + Vite)

- **Video Player** - Stream and analyze video content
- **Heatmap Overlay** - Real-time risk visualization
- **Risk Timeline** - Historical risk trends
- **Control Panel** - Analysis configuration
- **Upload Manager** - Batch video processing

### Core Modules

```
src/
├── models/
│   ├── csrnet.py              # Density estimation
│   ├── ciri_model.py          # CIRI calculation
│   └── transformer.py         # Spatio-temporal attention
├── features/
│   ├── optical_flow.py        # Motion analysis
│   ├── instability_features.py # Risk metrics
│   └── ciri_calculator.py     # CIRI computation
├── synthetic/
│   └── scenario_generator.py  # Training data generation
└── utils/
    ├── evaluation_metrics.py  # Validation metrics
    └── visualization.py       # Visualization tools
```

## Configuration

Configuration files are located in `configs/` directory:

**model_config.yaml**
```yaml
density_model: csrnet
optical_flow_method: raft
transformer_layers: 4
hidden_dim: 512
```

**training_config.yaml**
```yaml
batch_size: 16
learning_rate: 1e-4
epochs: 100
validation_split: 0.2
```

Edit these files to customize model architecture and training parameters.

## Docker Deployment

Deploy the entire application stack using Docker:

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

**Service Endpoints:**
- API: http://localhost:8000
- Frontend: http://localhost
- Database: localhost:5432

**Included Services:**
- FastAPI backend
- React frontend
- PostgreSQL database
- Redis cache
- NGINX reverse proxy

## Evaluation Metrics

The system implements comprehensive evaluation metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| AUC | Area Under ROC Curve | > 0.95 |
| Precision | True positives / predicted positives | > 0.90 |
| Recall | True positives / actual positives | > 0.88 |
| False Alarm Rate | False positives / negatives | < 5% |
| Detection Time | Early warning seconds before collapse | 2-5s |
| Heatmap IoU | Intersection over Union | > 0.85 |
| MAE | Mean Absolute Error | < 0.1 |
| MSE | Mean Squared Error | < 0.02 |

Run evaluations:
```bash
python experiments/evaluate_model.py --metrics all --visualize
```

## Synthetic Data Generation

Generate training data with various crowd scenarios:

```bash
python src/synthetic/scenario_generator.py \
  --scenario bidirectional \
  --frames 500 \
  --output data/synthetic/
```

**Available Scenarios:**
- **Bidirectional Flow** - Two-directional crowd movement
- **Bottleneck** - Compression and congestion
- **Stress Test** - Progressive instability
- **Event Simulation** - Concert/stadium scenarios
- **Panic** - Rapid evacuation patterns
- **Customizable** - Define your own patterns

**Output:** Synthetic videos with ground-truth CIRI annotations

## Research Output

The system generates publication-ready materials:

- **LaTeX Equations** - Mathematical formulations
- **Architecture Diagrams** - System visualization (Mermaid)
- **Configuration Templates** - Experiment reproducibility
- **Ablation Studies** - Component impact analysis
- **Baseline Comparisons** - Performance benchmarks
- **Technical Reports** - Detailed methodology

## Performance Optimizations

- **<100ms/frame** - Efficient inference on GPU
- **Model Quantization** - INT8 precision support
- **Batch Processing** - Multi-video analysis
- **Memory-Efficient** - Optimized tensor operations
- **Configurable** - Performance vs accuracy tradeoff
- **CPU Fallback** - Runs on CPU if GPU unavailable

**Benchmark Results (RTX 3090):**
```
Frame Resolution: 1920x1080
Throughput: 25-30 FPS
Latency: 35-40ms per frame
Memory: ~4GB VRAM
```

## Contributing

We welcome contributions! Follow these steps:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit changes (git commit -m 'Add amazing feature')
4. Push to branch (git push origin feature/amazing-feature)
5. Open a Pull Request

**Before contributing:**
- Run tests: `pytest tests/`
- Check formatting: `black .`
- Verify linting: `flake8 .`
- Update documentation

**Guidelines:**
- Write clear commit messages
- Add tests for new features
- Update README if needed
- Follow PEP 8 style guide

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{crowd-risk-prediction-2026,
  title={Vision-Based Early Crowd Collapse Risk Prediction via Spatio-Temporal Instability Modeling},
  author={Sreejith Nair},
  journal={IEEE/CVF Conference},
  year={2026},
  url={https://github.com/Sreejith-nair511/Crowd-Risk-Analysis}
}
```

---

## Acknowledgments

- **CSRNet** - Original density estimation paper authors
- **RAFT** - Optical flow methodology
- **PyTorch Team** - Deep learning framework
- **FastAPI Community** - Web framework excellence
- Contributors and testers who provided feedback

---

## Support & Contact

- **Email**: sreejith.nair@example.com
- **Issues**: [GitHub Issues](https://github.com/Sreejith-nair511/Crowd-Risk-Analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sreejith-nair511/Crowd-Risk-Analysis/discussions)
- **Social**: [@SreejithNair](https://twitter.com)

---

## Related Publications & Resources

- [CSRNet Paper](https://arxiv.org/abs/1807.10985) - Density Estimation
- [RAFT Paper](https://arxiv.org/abs/2003.12039) - Optical Flow
- [Vision Transformers](https://arxiv.org/abs/2010.11929) - Transformer Architecture
- [Crowd Analysis Survey](https://arxiv.org/abs/) - Literature Review

---

## Roadmap

- [x] Core CIRI model implementation
- [x] Real-time dashboard
- [x] Docker deployment
- [ ] Multi-camera support
- [ ] Edge device optimization
- [ ] Mobile app integration
- [ ] Advanced visualization features
- [ ] Integration with emergency systems

---

<div align="center">

Made with love by the Crowd Risk Analysis Team

</div>