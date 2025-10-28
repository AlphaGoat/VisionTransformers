"""
Module for plotting utilities.

Author: Peter Thomas
Date: 28 October 2025
"""
import torch
import matplotlib.pyplot as plt
from vision_transformers.layers import SinusoidalPositionalEncoding


def plot_attention(layer_name, feature_map, queries, keys, values):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"{layer_name} - Queries")
    plt.imshow(queries, aspect='auto', cmap='viridis')
    plt.colorbar()


def plot_positional_encodings(input_sequence, pos_encodings="sinusoidal"):

    if pos_encodings == "sinusoidal":
        pos_encoder = SinusoidalPositionalEncoding(input_sequence.size(-1), embed_dim=input_sequence.size(-1))
        pos_encodings = pos_encoder(input_sequence).squeeze(0).cpu().numpy()
    else:
        raise ValueError("Unsupported positional encoding type.")

    fig = plt.figure(figsize=(10, 5))
    plt.title("Positional Encodings")
    plt.imshow(pos_encodings, aspect='auto', cmap='plasma')
    plt.colorbar()
    return fig


def plot_bboxes(image, bboxes, labels=None):
    """
    Plots bounding boxes on the image.

    Args:
        image: The input image as a numpy array.
        bboxes: List of bounding boxes in (x_min, y_min, x_max, y_max) format.
        labels: Optional list of labels for each bounding box.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if labels:
            plt.text(x_min, y_min - 10, labels[i], color='white',
                     bbox=dict(facecolor='red', alpha=0.5))

    plt.show()


if __name__ == "__main__":
    # Example usage
    dummy_input = torch.randn(1, 3, 126 * 126)  # (batch_size, num_channels, width * height)
    fig = plot_positional_encodings(dummy_input, pos_encodings="sinusoidal")
    plt.show()