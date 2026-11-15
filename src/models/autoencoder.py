
import torch
import torch.nn as nn


class AstronautHealthAutoencoder(nn.Module):
    """
    PyTorch Autoencoder for astronaut health anomaly monitoring.

    Architecture:
        Input (18) -> 12 -> 6 -> Latent (3)
        Latent (3) -> 6 -> 12 -> Output (18)
    """

    def __init__(self, input_dim=18, latent_dim=3):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: compress input features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),

            nn.Linear(12, 6),
            nn.ReLU(),

            nn.Linear(6, latent_dim)
        )

        # Decoder: reconstruct original features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 6),
            nn.ReLU(),

            nn.Linear(6, 12),
            nn.ReLU(),

            nn.Linear(12, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        return reconstructed

    def get_reconstruction_error(self, x):
        """
        Calculate mean squared reconstruction error
        for each input sample.
        """

        reconstructed = self.forward(x)

        errors = torch.mean(
            (x - reconstructed) ** 2,
            dim=1
        )

        return errors
