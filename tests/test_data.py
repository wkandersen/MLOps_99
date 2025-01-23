import os
import pytest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from src.group_99.data import load_data, CustomDataset, ImageDataModule


@pytest.fixture
def mock_dataset():
    """
    Fixture to provide a mock dataset for testing.
    """
<<<<<<< Updated upstream
    data = pd.DataFrame({
        'path': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'],
        'class': ['class1', 'class2', 'class1', 'class2'],
        'label': [0, 1, 0, 1]
    })
    return data
=======
    fake_dataset_path = os.path.join(tmp_path, "fake_dataset")
    os.makedirs(os.path.join(fake_dataset_path, "class1"), exist_ok=True)
    os.makedirs(os.path.join(fake_dataset_path, "class2"), exist_ok=True)

    with open(os.path.join(fake_dataset_path, "class1", "1.jpg"), "w") as f:
        f.write("fake_image_data")
    with open(os.path.join(fake_dataset_path, "class2", "2.jpg"), "w") as f:
        f.write("fake_image_data")

    # Ensure that kagglehub is mocked to return the fake dataset path
    with patch("kagglehub.dataset_download", return_value=fake_dataset_path):
        yield fake_dataset_path
>>>>>>> Stashed changes


@pytest.fixture
def mock_transform():
    """
    Fixture to provide a mock transform.
    """
<<<<<<< Updated upstream
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
=======
    # Load the data, and ensure the mock_kagglehub fixture is used to provide the dataset path
    data, transform, class_names, dataset_path = load_data()
>>>>>>> Stashed changes


def test_custom_dataset(mock_dataset, mock_transform):
    """
    Test the CustomDataset class to ensure it loads data and applies transformations correctly.
    """
<<<<<<< Updated upstream
    # Mock PIL.Image.open
    with patch("PIL.Image.open", return_value=Image.new('RGB', (224, 224))) as mock_open:
        dataset = CustomDataset(mock_dataset, transform=mock_transform)
        
        # Test length
        assert len(dataset) == 4, "Dataset length should match input data"
=======
    data, transform, class_names, dataset_path = load_data()
>>>>>>> Stashed changes

        # Test item retrieval
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor), "Image should be a tensor"
        assert img.shape[1:] == (224, 224), "Image dimensions should be 224x224"
        assert label == 0, "Label should match the input data"

<<<<<<< Updated upstream
        # Ensure PIL.Image.open is called correctly
        mock_open.assert_called_with('image1.jpg')
=======
    dataset_size = len(dataset)
    train_size = int(0.6 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_subset_new = torch.utils.data.Subset(train_dataset, range(2))

    subset_size = len(train_subset_new)
    assert subset_size == 2, f"Expected subset size of 2, got {subset_size}"
    assert subset_size <= train_size, "Subset size should not exceed train dataset size"


@patch("os.path.exists", return_value=False)
@patch("kagglehub.dataset_download", side_effect=Exception("Download failed"))
def test_data_load_exceptions(mock_path_exists, mock_kagglehub_download):
    """
    Test that the dataset load fails gracefully when the dataset is unavailable.
    """
    with pytest.raises(RuntimeError, match="Dataset download failed"):
        load_data()


# Run the tests 1 and 2
if __name__ == "__main__":
    print(get_dataset_path_from_config())

>>>>>>> Stashed changes
