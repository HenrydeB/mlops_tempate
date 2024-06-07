import pytest
import torch

from models.MNIST_working import Decoder, Encoder, Model


@pytest.fixture
def model():
    input_dim = 784
    hidden_dim = 400
    latent_dim = 20
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)
    return Model(encoder, decoder)


def test_model_output_shape(model):
    batch_size = 64
    input_shape = (batch_size, 784)
    x = torch.randn(input_shape)

    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)

    expected_output_shape = input_shape
    assert x_hat.shape == expected_output_shape

    """
        this test can be used to test different input and output shapes
        to determine we are on the right track in our development
    """
