import torch
from .utils import get_x_positions, get_y_positions


class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, n_pixels, embed_dim):
        super().__init__()
        self.n_pixels = n_pixels
        self.embed_dim = embed_dim // 2 # Since we have x and y positions

        # get axis specific positions
        self.x_positions = get_x_positions(n_pixels)
        self.y_positions = get_y_positions(n_pixels)

        # Generate positional encodings
        x_pos_embedding = self.generate_sinusoidal_1d(self.x_positions.reshape(-1, 1))
        y_pos_embedding = self.generate_sinusoidal_1d(self.y_positions.reshape(-1, 1))

        # Combine x-axis and y-axis positional encodings
        pe = torch.cat((x_pos_embedding, y_pos_embedding), dim=-1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

    def generate_sinusoidal_1d(self, sequence):
        # Denominator
        denominator = torch.pow(10000, torch.arange(0, self.embed_dim, 2).float() / self.embed_dim)

        # Create empty tensor for positional encodings
        pos_encodings = torch.zeros((1, sequence.size(0), self.embed_dim))
        denominator = sequence / denominator
        pos_encodings[:, :, 0::2] = torch.sin(denominator)
        pos_encodings[:, :, 1::2] = torch.cos(denominator)
        return pos_encodings


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pos_encoding = SinusoidalPositionalEncoding(n_pixels=16, embed_dim=64)
    pe = pos_encoding.pe.squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.imshow(pe, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Sinusoidal Positional Encoding")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.show()