"""
ResNet backbone with 3d convolutional layers for processing multi/hyperpectral images.

Author: Peter Thomas
Date: 28 October 2025
"""
import torch


class Resnet3DConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride, padding):
        super(Resnet3DConvBlock, self).__init__()
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = torch.nn.BatchNorm3d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Residual3DConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Residual3DConvBlock, self).__init__()
        self.conv1 = Resnet3DConvBlock(in_channels, in_channels, kernel_size=1, groups=1, stride=stride, padding=1)
        self.conv2 = Resnet3DConvBlock(in_channels, in_channels, kernel_size=3, groups=1, stride=1, padding=1)
        self.conv3 = Resnet3DConvBlock(in_channels, out_channels, kernel_size=1, groups=1, stride=1, padding=0)
        self.downsample = downsample
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Resnet50With3dConv(torch.nn.Module):
    def __init__(self, classification_head=False, num_classes=100):
        super(Resnet50With3dConv, self).__init__()
        self.num_classes = num_classes
        self.classification_head = classification_head

        # Initialize 3D ResNet50 model here
        self.conv1 = Resnet3DConvBlock(3, 64, kernel_size=7, groups=1, stride=2, padding=3)
        self.maxpool = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.resblock1 = self._make_layer(Residual3DConvBlock, 64, 3, stride=1)
        self.resblock2 = self._make_layer(Residual3DConvBlock, 128, 4, stride=2)
        self.resblock3 = self._make_layer(Residual3DConvBlock, 256, 6, stride=2)
        self.resblock4 = self._make_layer(Residual3DConvBlock, 512, 3, stride=2)

        # Global average pooling and fully connected layer
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

        if classification_head:
            self.fc = torch.nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # Define forward pass for 3D ResNet50 here
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.avgpool(x)
        if self.classification_head:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x