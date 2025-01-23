import pytest
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from src.group_99.data import CustomDataset
from unittest.mock import patch


@pytest.fixture
def mock_dataset():
    """
    Fixture to provide a mock dataset for testing.
    """
    data = pd.DataFrame({
        'path': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'],
        'class': ['class1', 'class2', 'class1', 'class2'],
        'label': [0, 1, 0, 1]
    })
    return data


@pytest.fixture
def mock_transform():
    """
    Fixture to provide a mock transform.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


def test_custom_dataset(mock_dataset, mock_transform):
    """
    Test the CustomDataset class to ensure it loads data and applies transformations correctly.
    """
    # Mock PIL.Image.open
    with patch("PIL.Image.open", return_value=Image.new('RGB', (224, 224))) as mock_open:
        dataset = CustomDataset(mock_dataset, transform=mock_transform)
        
        # Test length
        assert len(dataset) == 4, "Dataset length should match input data"

        # Test item retrieval
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor), "Image should be a tensor"
        assert img.shape[1:] == (224, 224), "Image dimensions should be 224x224"
        assert label == 0, "Label should match the input data"

        # Ensure PIL.Image.open is called correctly
        mock_open.assert_called_with('image1.jpg')
