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


class Resnet50With3dConv(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet50With3dConv, self).__init__()


        # Initialize 3D ResNet50 model here
        pass

    def forward(self, x):
        # Define forward pass for 3D ResNet50 here
        pass