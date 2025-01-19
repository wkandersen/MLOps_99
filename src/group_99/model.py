import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import models
import torch.nn.functional as F



import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=101, pretrained=True, dropout_rate=0.5):
        """
        Custom ResNet-18 model with a dropout and final fully connected layer.

        Args:
            num_classes (int): Number of output classes (default: 101).
            pretrained (bool): Whether to load pretrained weights (default: True).
            dropout_rate (float): Dropout probability (default: 0.5).
        """
        super(CustomResNet18, self).__init__()

        # Load ResNet-18 with or without pretrained weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = resnet18(weights=weights)

        # Modify the fully connected layer (fc) to add Dropout and then a new Linear layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # Apply dropout
            nn.Linear(self.resnet.fc.in_features, num_classes)  # Output layer with specified number of classes
        )

    def forward(self, x):
        return self.resnet(x)  # Forward pass through ResNet-18 with modified fc layer



class ResNet50Simple():
    def __init__(self, num_classes, x_dim):
        super(ResNet50Simple, self).__init__()
        
        # Load the pretrained ResNet50 model
        self.resnet = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes))



import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes, x_dim, dropout_rate):
        super(CNNModel, self).__init__()
        
        # x_dim represents flattened input size: channels * height * width
        # We can infer the input dimensions based on x_dim if needed.
        self.x_dim = x_dim
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # Define the convolutional layers with increasing depth
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Input: 3 channels (RGB), Output: 64 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Input: 64 channels, Output: 128 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Input: 128 channels, Output: 256 channels

        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Pooling with 2x2 kernel

        # Calculate the output size after convolutional and pooling layers
        # We assume the input size is square and that the input dimensions are in x_dim
        # The size after 3 max-pooling layers (each halving the dimension) will be:
        self.fc_input_dim = x_dim

        # Fully connected layers after the convolutions
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)  # Flattened image size
        self.fc2 = nn.Linear(1024, num_classes)  # Final layer for classification

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        
        # Final output layer (logits)
        x = self.fc2(x)
        return x

