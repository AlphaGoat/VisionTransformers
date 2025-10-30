# vision_transformers/torch-ext/bilinear_interpolate/__init__.py
import torch
from ._ops import ops


def bilinear_interpolate(input: torch.Tensor, output: torch.Tensor) -> None:
    """
    Performs bilinear interpolation on the input tensor and writes the result to the output tensor.

    Args:
        input (torch.Tensor): The input tensor of shape (H, W).
        height (int): The height of the output tensor.
        width (int): The width of the output tensor.
        output (torch.Tensor): The output tensor to write the interpolated values to, of shape (height, width).
    """
    height, width = input.shape[2:]
    output = torch.empty(height, width, dtype=input.dtype, device=input.device)
    ops.bilinear_interpolate(input, height, width, output)

    return output