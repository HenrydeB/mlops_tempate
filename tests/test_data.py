import pytest
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

from tests import _PATH_DATA

dataset_path = "../datasets"

# expected number of samples for each
N_train = 60000
N_test = 10000

mnist_transform = transforms.Compose([transforms.ToTensor()])


@pytest.fixture
def train_dataset():
    return MNIST(dataset_path, transform=mnist_transform, train=True, download=True)


@pytest.fixture
def test_dataset():
    return MNIST(dataset_path, transform=mnist_transform, train=False, download=True)


def test_train_dataset_length(train_dataset):
    assert len(train_dataset) == N_train


def test_test_dataset_length(test_dataset):
    assert len(test_dataset) == N_test


# iterates over both train and test data and checks shape of data
def test_data_shape(train_dataset, test_dataset):
    for dataset in [train_dataset, test_dataset]:
        for data, _ in dataset:
            assert data.shape == (1, 28, 28) or data.shape == (784,)


# collects all unique labels in both datasets
# assserts there are 10 unique labels
def test_data_representation(train_dataset, test_dataset):
    labels = set()
    for dataset in [train_dataset, test_dataset]:
        for _, label in dataset:
            labels.add(label)
    assert len(labels) == 10
