import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.input_dim = 3
        self.hidden_dim = 8
        self.output_dim = 128
        self.conv2d_1 = nn.Conv2d(in_channels=self.input_dim , out_channels= self.hidden_dim, kernel_size=2 , stride=2)
        self.conv2d_2 = nn.Conv2d(in_channels=self.hidden_dim , out_channels= 8* self.hidden_dim, kernel_size=2 , stride=2 , padding=1)
        self.conv2d_3 = nn.Conv2d(in_channels=8 * self.hidden_dim, out_channels=self.output_dim, kernel_size=2 , stride=2 , padding=1)
        self.pool = nn.MaxPool2d(2 , 2)
        # self.pool  = nn.MaxPool2d(2, 2)               # 下采样
        self.fx = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128 ),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128 , 16) ,
            nn.ReLU(),
            nn.Linear(16 , num_classes)

        )
    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))  # [3,64,64] -> [16,32,32]
        # x = self.pool(F.relu(self.conv2(x)))  # [16,32,32] -> [32,16,16]
        
        x = self.pool(F.relu(self.conv2d_1(x)))
        x = self.pool(F.relu(self.conv2d_2(x)))
        x = self.pool(F.relu(self.conv2d_3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fx(x)
        return x