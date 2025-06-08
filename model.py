import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedMNISTCNN(nn.Module):
    """
    Optimized MNIST CNN model - adapted for iDLG attacks
    """
    def __init__(self, num_classes=10):
        super(OptimizedMNISTCNN, self).__init__()
        
        # Use Sigmoid activation function, more suitable for gradient inversion attacks
        act = nn.Sigmoid
        
        # Convolutional layers - similar to LeNet architecture
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 12, kernel_size=5, padding=2, stride=2),  # 28x28 -> 14x14
            act(),
            
            # Second convolutional layer
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),  # 14x14 -> 7x7
            act(),
            
            # Third convolutional layer
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1),  # 7x7 -> 7x7
            act(),
        )
        
        # Calculate flattened feature dimension (12 * 7 * 7 = 588)
        self.feature_dim = 12 * 7 * 7
        
        # Fully connected layer
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Weight initialization - use same initialization as original code
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """Weight initialization - consistent with original iDLG code"""
        try:
            if hasattr(m, "weight"):
                m.weight.data.uniform_(-0.5, 0.5)
        except Exception:
            print(f'Warning: failed in weights_init for {m._get_name()}.weight')
        try:
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.uniform_(-0.5, 0.5)
        except Exception:
            print(f'Warning: failed in weights_init for {m._get_name()}.bias')
    
    def forward(self, x):
        # Convolutional feature extraction
        out = self.conv_layers(x)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # Fully connected classification
        out = self.fc(out)
        
        return out
    
    def get_feature_dim(self):
        """Get feature dimension"""
        return self.feature_dim


class LeNetMNIST(nn.Module):
    """
    LeNet architecture MNIST model - directly adapted from original iDLG code
    """
    def __init__(self, channel=1, hidden=588, num_classes=10):
        super(LeNetMNIST, self).__init__()
        
        act = nn.Sigmoid
        
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )
        
        # Apply weight initialization
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """Weight initialization"""
        try:
            if hasattr(m, "weight"):
                m.weight.data.uniform_(-0.5, 0.5)
        except Exception:
            print(f'Warning: failed in weights_init for {m._get_name()}.weight')
        try:
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.uniform_(-0.5, 0.5)
        except Exception:
            print(f'Warning: failed in weights_init for {m._get_name()}.bias')
    
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out