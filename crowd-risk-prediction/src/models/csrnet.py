import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict

# VGG backbone for CSRNet
class VGGBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGBackbone, self).__init__()
        
        # First few layers of VGG16
        self.front_end = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Dilation layers
        self.back_end = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.front_end(x)
        x = self.back_end(x)
        return x

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.backend_feat = [512, 512, 512, 256, 128, 64, 1]
        self.frontend = VGGBackbone(pretrained=True)
        self.backend = make_layers(self.backend_feat, in_channels=512, batch_norm=False, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if load_weights:
            # Load pretrained weights here if available
            pass

    def forward(self, x):
        front_features = self.frontend(x)
        backend_features = self.backend(front_features)
        output = self.output_layer(backend_features)
        return output

def make_layers(cfg, in_channels, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class DensityEstimator:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize CSRNet density estimator
        Args:
            model_path: Path to pretrained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = CSRNet(load_weights=model_path is not None)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        # Transform for input images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def estimate_density(self, frame):
        """
        Process input frame and return normalized density heatmap (0-1)
        Args:
            frame: Input frame as numpy array (H, W, C) or torch tensor
        Returns:
            Density heatmap as numpy array normalized to [0, 1]
        """
        import numpy as np
        
        # Convert to tensor if needed
        if isinstance(frame, np.ndarray):
            frame_tensor = self.transform(frame).unsqueeze(0)
        else:
            frame_tensor = frame.unsqueeze(0) if len(frame.shape) == 3 else frame
            
        frame_tensor = frame_tensor.to(self.device)
        
        with torch.no_grad():
            density_map = self.model(frame_tensor)
            # Apply sigmoid to normalize to [0, 1]
            density_map = torch.sigmoid(density_map)
            # Move to CPU and convert to numpy
            density_map = density_map.cpu().squeeze().numpy()
        
        return density_map