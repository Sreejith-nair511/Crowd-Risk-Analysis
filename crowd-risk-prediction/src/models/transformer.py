import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PatchEmbedding(nn.Module):
    """Convert input feature maps into patches for transformer processing"""
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, embed_dim, 1, 1))  # Simplified pos encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, height/patch_size, width/patch_size)
        x = x + self.positional_encoding
        # Reshape to (batch_size, seq_len, embed_dim)
        batch_size, embed_dim, h_patches, w_patches = x.shape
        x = x.view(batch_size, embed_dim, h_patches * w_patches).transpose(1, 2)
        return x

class SpatioTemporalAttention(nn.Module):
    """Multi-head attention mechanism for both spatial and temporal dimensions"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, 
                spatial_mask: Optional[torch.Tensor] = None,
                temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            spatial_mask: Optional spatial attention mask
            temporal_mask: Optional temporal attention mask
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate qkv
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply optional masks
        if spatial_mask is not None:
            attn_scores = attn_scores.masked_fill(spatial_mask == 0, float('-inf'))
        if temporal_mask is not None:
            attn_scores = attn_scores.masked_fill(temporal_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        output = attn_weights @ v  # (batch_size, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.proj(output)
        output = self.proj_dropout(output)
        
        return output

class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Single transformer block with both spatial and temporal attention"""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = SpatioTemporalAttention(embed_dim, num_heads, dropout)
        self.temporal_attn = SpatioTemporalAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial attention (applied independently to each time step)
        residual = x
        x = self.norm1(x)
        x = self.spatial_attn(x)
        x = self.dropout(x) + residual
        
        # Temporal attention (applied across time steps for each spatial position)
        residual = x
        x = self.norm2(x)
        x = self.temporal_attn(x)
        x = self.dropout(x) + residual
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        
        return x

class SpatioTemporalTransformer(nn.Module):
    """
    Spatio-Temporal Transformer for crowd instability prediction
    Processes sequences of feature maps to predict future risk
    """
    def __init__(self, 
                 input_channels: int = 6,  # Density, FOI, LMCS, density_grad, accel_spikes, velocity
                 patch_size: int = 16,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 ff_dim: int = 512,
                 seq_length: int = 8,
                 output_size: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.output_size = output_size
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(patch_size, input_channels, embed_dim)
        
        # Learnable temporal positional embeddings
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norm before final prediction
        self.norm = nn.LayerNorm(embed_dim)
        
        # Prediction heads
        # Head for current risk prediction
        self.current_risk_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_size),
            nn.Sigmoid()  # Output normalized risk score [0, 1]
        )
        
        # Head for future risk prediction (2-5 seconds ahead)
        self.future_risk_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_size),
            nn.Sigmoid()  # Output normalized risk score [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the spatio-temporal transformer
        Args:
            x: Input tensor of shape (batch_size, seq_length, height, width, input_channels)
        Returns:
            Tuple of (current_risk_map, future_risk_map) both with shape (batch_size, height, width, 1)
        """
        batch_size, seq_len, height, width, channels = x.shape
        
        # Reshape to combine batch and sequence dimensions for patch embedding
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Apply patch embedding
        x = self.patch_embed(x)  # Shape: (batch_size * seq_len, num_patches, embed_dim)
        
        # Reshape back to separate batch and sequence dimensions
        num_patches = x.size(1)
        x = x.view(batch_size, seq_len, num_patches, self.embed_dim)
        
        # Add temporal positional encoding
        x = x + self.temporal_pos_embed[:, :seq_len, :].unsqueeze(2)
        
        # Reshape to apply transformer: (batch_size, seq_len * num_patches, embed_dim)
        x = x.view(batch_size, seq_len * num_patches, self.embed_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Split back into sequence and spatial dimensions
        x = x.view(batch_size, seq_len, num_patches, self.embed_dim)
        
        # Process separately for current and future predictions
        # Take the last time step for current prediction
        current_features = x[:, -1, :, :]  # (batch_size, num_patches, embed_dim)
        current_risk = self.current_risk_head(current_features)  # (batch_size, num_patches, output_size)
        
        # For future prediction, use temporal aggregation
        # Average across time dimension
        temporal_features = x.mean(dim=1)  # (batch_size, num_patches, embed_dim)
        future_risk = self.future_risk_head(temporal_features)  # (batch_size, num_patches, output_size)
        
        # Reshape back to spatial dimensions (assuming square patches)
        patch_h = patch_w = int(math.sqrt(num_patches))
        current_risk = current_risk.view(batch_size, patch_h, patch_w, self.output_size)
        future_risk = future_risk.view(batch_size, patch_h, patch_w, self.output_size)
        
        # Upsample to original resolution if needed
        if patch_h != height // self.patch_size or patch_w != width // self.patch_size:
            current_risk = F.interpolate(
                current_risk.permute(0, 3, 1, 2), 
                size=(height // self.patch_size, width // self.patch_size), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
            
            future_risk = F.interpolate(
                future_risk.permute(0, 3, 1, 2), 
                size=(height // self.patch_size, width // self.patch_size), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            current_risk = current_risk.permute(0, 1, 2, 3)
            future_risk = future_risk.permute(0, 1, 2, 3)
        
        return current_risk, future_risk

    def predict_instability_sequence(self, feature_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to predict both current and future instability
        Args:
            feature_sequence: Sequence of feature tensors (density, flow, etc.)
        Returns:
            Tuple of (current_risk, future_risk) maps
        """
        return self.forward(feature_sequence)