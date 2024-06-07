import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from models.MNIST_working import Decoder, Encoder, Model, loss_function


# Define a fixture to create a small dataset and dataloader
@pytest.fixture
def data():
    # Create a small dataset and dataloader
    batch_size = 2
    x_dim = 784
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = MNIST("datasets", transform=mnist_transform, train=False, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    x, _ = next(iter(test_loader))
    return x


# Test the forward pass of the Encoder to ensure output tensors have the right shape
def test_encoder_forward(data):
    hidden_dim = 400
    latent_dim = 20
    encoder = Encoder(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim)
    x = data.view(data.size(0), -1)  # Flatten the input tensor
    z, mean, log_var = encoder(x)
    assert z.shape == (data.shape[0], latent_dim)
    assert mean.shape == (data.shape[0], latent_dim)
    assert log_var.shape == (data.shape[0], latent_dim)


# Test the forward pass of the Decoder to ensure output tensors have the right shape
def test_decoder_forward(data):
    hidden_dim = 400
    latent_dim = 20
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=784)
    z = torch.randn(data.shape[0], latent_dim)
    x_hat = decoder(z)
    assert x_hat.shape == (data.shape[0], 784)


# Test the loss function to ensure the loss value is non-negative
def test_loss_function(data):
    hidden_dim = 400
    latent_dim = 20
    encoder = Encoder(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=784)
    model = Model(encoder=encoder, decoder=decoder)
    x = data.view(data.size(0), -1)  # Flatten the input tensor
    x_hat, mean, log_var = model(x)
    loss = loss_function(x, x_hat, mean, log_var)
    assert loss.item() >= 0
