import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import models


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



class SimpleCNN(nn.Module):
    def __init__(self, num_classes, x_dim):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(x_dim, 128)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(128, num_classes)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply convolutional layers with ReLU activations
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten the tensor
        x = torch.flatten(x, 1)
        
        # Pass through fully connected layers
        x = torch.relu(self.fc(x))
        x = self.fc2(x)  # Final output layer (no activation here for classification)
        
        return x




