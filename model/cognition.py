import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)   # 输入3通道(RGB)，输出16通道
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)               # 下采样
        self.fc1   = nn.Linear(32 * 16 * 16, 128)     # 64x64经过两次池化 → 16x16
        self.fc2   = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [3,64,64] -> [16,32,32]
        x = self.pool(F.relu(self.conv2(x)))  # [16,32,32] -> [32,16,16]
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x