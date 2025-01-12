import torch
import torch.nn as nn
from torchvision.models import resnet50


class CustomResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CustomResNet50, self).__init__()
        # Load the pretrained ResNet50 model
        self.resnet = resnet50(pretrained=pretrained)
        
        # layers go from 2048 to 1024 to 512 to num_classes
        # Change the output layer to num_classes

        self.resnet.fc = nn.Linear(2048, num_classes)

    
    def forward(self, x):
            # Pass through ResNet50 layers (excluding the fully connected part)
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            
            # Pass through the ResNet50's residual blocks
            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
            
            # Global average pooling
            x = self.resnet.avgpool(x)
            
            # Flatten the output from the ResNet50
            x = torch.flatten(x, 1)  # Flatten all dimensions except batch size
            
            # Pass through custom fully connected layers
            x = self.resnet.fc(x)

            
            return x
        

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
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Final output layer (no activation here for classification)
        
        return x
