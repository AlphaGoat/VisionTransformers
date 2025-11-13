import torch
from .utils import get_x_positions, get_y_positions


class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, n_pixels, embed_dim, *args, **kwargs):
        super().__init__()
        self.position_embeddings = torch.nn.Embedding(n_pixels, embed_dim)
        self.n_pixels = n_pixels
        self.embed_dim = embed_dim

        # Initialize the positional embeddings
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, x):
        positions = torch.arange(0, self.n_pixels, device=x.device).unsqueeze(0)
        pos_embeddings = self.position_embeddings(positions)
        x = x + pos_embeddings
        return x


class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, n_pixels, embed_dim, *args, **kwargs):
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
#        x = x + self.pe[:x.size(0), :]
        x = x + self.pe
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


class SinusoidalPositionalEncoding2D(torch.nn.Module):
    def __init__(self, height, width, embed_dim):
        super().__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim 

        # Generate positional encodings
        pe = self.generate_sinusoidal_2d()
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

    def generate_sinusoidal_2d(self):
        d_model = self.embed_dim // 2
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_w = torch.arange(0., self.width).unsqueeze(1)
        pos_h = torch.arange(0., self.height).unsqueeze(1)
        pe = torch.zeros(self.embed_dim, self.width, self.height)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)
        pe = pe.view(self.embed_dim, self.height * self.width).transpose(1, 0).unsqueeze(0)
        return pe

#    def generate_sinusoidal_2d(self):
#        # Create position grids
#        y_positions = torch.arange(0, self.height, dtype=torch.float32).unsqueeze(1).repeat(1, self.width).flatten()
#        x_positions = torch.arange(0, self.width, dtype=torch.float32).unsqueeze(0).repeat(self.height, 1).flatten()
#
#        # Denominator
#        denominator = torch.pow(10000, torch.arange(0, self.embed_dim, 2).float() / self.embed_dim)
#
#        # Create empty tensor for positional encodings
#        pos_encodings = torch.zeros((1, self.height * self.width, self.embed_dim * 2))
#
#        # X-axis positional encodings
#        x_denominator = x_positions.unsqueeze(1) / denominator
#        pos_encodings[:, :, 0::4] = torch.sin(x_denominator)
#        pos_encodings[:, :, 1::4] = torch.cos(x_denominator)
#
#        # Y-axis positional encodings
#        y_denominator = y_positions.unsqueeze(1) / denominator
#        pos_encodings[:, :, 2::4] = torch.sin(y_denominator)
#        pos_encodings[:, :, 3::4] = torch.cos(y_denominator)
#
#        return pos_encodings


def get_positional_encoding(encoding_type, *args, **kwargs):
    if encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(*args, **kwargs)
    elif encoding_type == "learned":
        return LearnedPositionalEncoding(*args, **kwargs)
    else:
        raise NotImplementedError(f"{encoding_type} positional encoding not implemented.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
#    pos_encoding = SinusoidalPositionalEncoding2D(height=15, width=15, embed_dim=64)
    pos_encoding = SinusoidalPositionalEncoding(n_pixels=15 * 15, embed_dim=64)
    pe = pos_encoding.pe.squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.imshow(pe, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Sinusoidal Positional Encoding")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.show()