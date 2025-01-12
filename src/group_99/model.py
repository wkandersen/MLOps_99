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
        


        
