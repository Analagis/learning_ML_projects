import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, in_channels=3, use_dropout=False, dropout_p=0.5, num_classes=1):
        super().__init__()
        # Блок 1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0
        )   # 64x64 -> 60x60 (без padding)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 60x60 -> 30x30

        # Блок 2: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )   # 30x30 -> 26x26
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 26x26 -> 13x13

        # После второго пула размер feature map:
        # channels = 16, height = 13, width = 13 => 16 * 13 * 13 = 2704
        self.flatten_dim = 16 * 13 * 13

        # Полносвязные слои
        self.fc1 = nn.Linear(self.flatten_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)  # выход по размеру классификации

        # Dropout
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # x: (batch_size, in_channels, 64, 64)
        x = self.pool1(F.relu(self.conv1(x)))  # -> (batch, 6, 30, 30)
        x = self.pool2(F.relu(self.conv2(x)))  # -> (batch, 16, 13, 13)

        # расплющиваем
        x = x.view(x.size(0), -1)              # -> (batch, 2704)

        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)

        x = self.out(x)                        # -> (batch, num_classes)
        return x
