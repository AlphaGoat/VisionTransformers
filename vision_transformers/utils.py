"""
Utility functions specific for vision transformers.

Author: Peter Thomas
Date: 28 October 2025
"""
import torch


def get_x_positions(num_pixels_feature_map, start_idx=0):
    """
    Generate x-coordinate positions for a feature map.

    Taken from s-chh's implementation here:
    https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer/blob/main/utils.py

    Args:
        num_pixels_feature_map (int): Number of pixels in the feature map along one dimension.
        start_idx (int): Starting index for position generation.

    Returns:
        torch.Tensor: A tensor containing the x-coordinate positions.
    """
    num_pixels_ = int(num_pixels_feature_map** 0.5)

    x_positions = torch.arange(start_idx, start_idx + num_pixels_, dtype=torch.float32)
    x_positions = x_positions.unsqueeze(0)
    x_positions = torch.repeat_interleave(x_positions, num_pixels_, dim=0)
    x_positions = x_positions.flatten()
    return x_positions


def get_y_positions(num_pixels_feature_map, start_idx=0):
    """
    Generate y-coordinate positions for a feature map.

    Taken from s-chh's implementation here:
    https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer/blob/main/utils.py

    Args:
        num_pixels_feature_map (int): Number of pixels in the feature map along one dimension.
        start_idx (int): Starting index for position generation.

    Returns:
        torch.Tensor: A tensor containing the y-coordinate positions.
    """
    num_pixels_ = int(num_pixels_feature_map** 0.5)

    y_positions = torch.arange(start_idx, start_idx + num_pixels_, dtype=torch.float32)
    y_positions = torch.repeat_interleave(y_positions, num_pixels_, dim=0)
    return y_positions


def get_num_output_channels(backbone, input_shape=(3, 256, 256)):
    """ Utility function to get the number of output channels from the backbone model. """
    dummy_input = torch.randn(1, *input_shape)
    with torch.no_grad():
        features = backbone(dummy_input)
    return features.size(1)  # Assuming features shape is (1, C, H, W)


def get_output_shape(backbone, input_shape=(3, 256, 256)):
    """ Utility function to get the number of output channels from the backbone model. """
    dummy_input = torch.randn(1, *input_shape)
    with torch.no_grad():
        features = backbone(dummy_input)["feature_map"]
    return features.shape  # Assuming features shape is (1, C, H, W)

def init_weights(module):
    """ Initialize weights of linear and convolutional layers using Xavier uniform initialization. """
    def _init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    for m in module.children():
        m.apply(_init_weights)
