import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedMNISTCNN(nn.Module):
    """
    優化的MNIST CNN模型 - 適配iDLG攻擊
    """
    def __init__(self, num_classes=10):
        super(OptimizedMNISTCNN, self).__init__()
        
        # 使用Sigmoid激活函數，更適合梯度反轉攻擊
        act = nn.Sigmoid
        
        # 卷積層 - 類似LeNet架構
        self.conv_layers = nn.Sequential(
            # 第一層卷積
            nn.Conv2d(1, 12, kernel_size=5, padding=2, stride=2),  # 28x28 -> 14x14
            act(),
            
            # 第二層卷積  
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),  # 14x14 -> 7x7
            act(),
            
            # 第三層卷積
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1),  # 7x7 -> 7x7
            act(),
        )
        
        # 計算展平後的特徵維度 (12 * 7 * 7 = 588)
        self.feature_dim = 12 * 7 * 7
        
        # 全連接層
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # 初始化權重 - 使用與原代碼相同的初始化方式
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """權重初始化 - 與原iDLG代碼保持一致"""
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
        # 卷積特徵提取
        out = self.conv_layers(x)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全連接分類
        out = self.fc(out)
        
        return out
    
    def get_feature_dim(self):
        """獲取特徵維度"""
        return self.feature_dim


class LeNetMNIST(nn.Module):
    """
    LeNet架構的MNIST模型 - 直接適配原iDLG代碼
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
        
        # 應用權重初始化
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """權重初始化"""
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