import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path

from src.models.ciri_model import CIRIPredictor, CIRILoss
from src.models.transformer import SpatioTemporalTransformer
from src.features.instability_features import InstabilityFeatures
from src.models.csrnet import DensityEstimator
from src.features.optical_flow import OpticalFlowProcessor

class CrowdRiskDataset(Dataset):
    """Dataset class for crowd risk prediction training"""
    def __init__(self, data_path, sequence_length=8, transform=None):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # In a real implementation, this would load video sequences and annotations
        # For now, we'll generate synthetic data
        self.samples = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic training data"""
        samples = []
        
        # Generate 1000 synthetic sequences for demonstration
        for i in range(1000):
            # Create synthetic feature sequence
            # Shape: (seq_length, height, width, features)
            seq = np.random.rand(self.sequence_length, 64, 64, 6).astype(np.float32)
            
            # Create synthetic target (risk scores)
            current_target = np.random.rand(64, 64, 1).astype(np.float32)
            future_target = np.random.rand(64, 64, 1).astype(np.float32)
            
            samples.append({
                'sequence': seq,
                'current_target': current_target,
                'future_target': future_target
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        sequence = torch.from_numpy(sample['sequence'])
        current_target = torch.from_numpy(sample['current_target'])
        future_target = torch.from_numpy(sample['future_target'])
        
        return {
            'sequence': sequence,
            'current_target': current_target,
            'future_target': future_target
        }

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        sequence = batch['sequence'].to(device)
        current_target = batch['current_target'].to(device)
        future_target = batch['future_target'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequence, return_intermediates=False)
        current_pred = outputs['current_risk']
        future_pred = outputs['future_risk']
        
        # Calculate loss
        loss = criterion(current_pred, future_pred, current_target, future_target)
        
        # Backward pass
        loss.backward()
        
        # Optimize
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move data to device
            sequence = batch['sequence'].to(device)
            current_target = batch['current_target'].to(device)
            future_target = batch['future_target'].to(device)
            
            # Forward pass
            outputs = model(sequence, return_intermediates=False)
            current_pred = outputs['current_risk']
            future_pred = outputs['future_risk']
            
            # Calculate loss
            loss = criterion(current_pred, future_pred, current_target, future_target)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train CIRI model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs/',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'input_channels': 6,
                'seq_length': 8,
                'embed_dim': 256,
                'num_heads': 8,
                'num_layers': 6
            },
            'training': {
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs
            }
        }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize model
    transformer = SpatioTemporalTransformer(
        input_channels=config['model']['input_channels'],
        seq_length=config['model']['seq_length'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers']
    )
    
    model = CIRIPredictor(transformer_model=transformer).to(device)
    
    # Initialize loss function
    criterion = CIRILoss(
        future_weight=0.5,
        current_weight=0.5
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=1e-5
    )
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize datasets
    train_dataset = CrowdRiskDataset(data_path="data/train")
    val_dataset = CrowdRiskDataset(data_path="data/val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / f"ciri_model_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"ciri_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()