from model.cnn_block import DoubleConv
import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self, num_classes , input_dim , hidden_dim):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool1(c1))
        bn = self.bottleneck(self.pool2(c2))
        u2 = self.up2(bn)
        c2 = torch.cat([u2, c2], dim=1)
        c2 = self.conv2(c2)
        u1 = self.up1(c2)
        c1 = torch.cat([u1, c1], dim=1)
        c1 = self.conv1(c1)
        return self.outc(c1)