
import torch

from src.models.autoencoder import AstronautHealthAutoencoder


def test_autoencoder_output_shape():

    model = AstronautHealthAutoencoder(
        input_dim=18,
        latent_dim=3
    )

    input_data = torch.randn(4, 18)

    output = model(input_data)

    assert output.shape == (4, 18)


def test_reconstruction_error_shape():

    model = AstronautHealthAutoencoder(
        input_dim=18,
        latent_dim=3
    )

    input_data = torch.randn(5, 18)

    errors = model.get_reconstruction_error(input_data)

    assert errors.shape == (5,)


def test_reconstruction_error_non_negative():

    model = AstronautHealthAutoencoder()

    input_data = torch.randn(4, 18)

    errors = model.get_reconstruction_error(input_data)

    assert torch.all(errors >= 0)


def test_autoencoder_parameter_count():

    model = AstronautHealthAutoencoder()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameter_count == 669
