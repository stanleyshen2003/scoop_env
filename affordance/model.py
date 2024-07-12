import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, input_channels=4, num_classes=5):
        super(ResNet50, self).__init__()
        # Load pre-trained ResNet50 model
        resnet50 = models.resnet50(pretrained=True)
        
        # Modify the first layer to accept 4 input channels
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the first layer of ResNet50 with our modified layer
        self.conv1.weight.data[:, :3, :, :] = resnet50.conv1.weight.data
        self.conv1.weight.data[:, 3, :, :] = resnet50.conv1.weight.data[:, 0, :, :]  # Copy the weights of the first channel for the 4th channel
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        
        # Use the rest of the ResNet50 layers
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        
        # Modify the fully connected layer to have 5 output classes
        self.avgpool = resnet50.avgpool
        self.fc = nn.Linear(resnet50.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
