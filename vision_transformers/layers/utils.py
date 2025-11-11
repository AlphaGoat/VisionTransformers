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

