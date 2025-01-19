import os
import pytest
import torch
from unittest.mock import patch, MagicMock, mock_open
from torchvision import datasets
from torch.utils.data import DataLoader

from src.group_99.data import load_data, CustomDataset, ImageDataset


@pytest.fixture
def mock_kagglehub(tmp_path):
    """
    Mock the kagglehub.dataset_download function and create a fake directory structure.
    """
    # Mock dataset download path
    fake_dataset_path = str(tmp_path / "sea_animals")

    # Create directory structure
    os.makedirs(os.path.join(fake_dataset_path, "class1"), exist_ok=True)
    os.makedirs(os.path.join(fake_dataset_path, "class2"), exist_ok=True)

    # Create dummy image files
    with open(os.path.join(fake_dataset_path, "class1", "1.jpg"), "w") as f:
        f.write("fake_image_data")
    with open(os.path.join(fake_dataset_path, "class2", "2.jpg"), "w") as f:
        f.write("fake_image_data")

    # Mock kagglehub download
    with patch("kagglehub.dataset_download", return_value=fake_dataset_path):
        yield fake_dataset_path


@patch("shutil.move")
def test_load_data_batch_shapes(mock_shutil_move, mock_kagglehub):
    """
    Test that the shapes of the data batches returned by train_loader
    (and others) are as expected, e.g. (batch_size, 3, 224, 224).
    """
    data, transform, class_names = load_data()

    # Convert the path-label pairs into the dataset
    path_label = [(row['path'], row['label']) for _, row in data.iterrows()]
    dataset = CustomDataset(path_label, transform)

    # Split into training and validation
    dataset_size = len(dataset)
    train_size = int(0.6 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Get one batch from the training loader
    batch = next(iter(train_loader))
    inputs, labels = batch

    # Assert inputs is a 4D tensor
    assert len(inputs.shape) == 4, "Inputs should be a 4D tensor"
    # Typically we expect (batch_size, 3, 224, 224) for an RGB image
    assert inputs.shape[1] == 3, "Channel dimension should be 3 for RGB images"
    assert inputs.shape[2] == 224 and inputs.shape[3] == 224, (
        "Images should be resized to 224x224"
    )
    
    # Check if the label shape matches the batch size
    assert labels.shape[0] == inputs.shape[0], "Labels batch size must match image batch size"


@patch("shutil.move")
def test_load_data_subset(mock_shutil_move, mock_kagglehub):
    """
    Test that the subset data loader (train_subset_new) has the correct length
    and is indeed smaller than the full training dataset.
    """
    data, transform, class_names = load_data()

    # Convert the path-label pairs into the dataset
    path_label = [(row['path'], row['label']) for _, row in data.iterrows()]
    dataset = CustomDataset(path_label, transform)

    # Split into training and validation
    dataset_size = len(dataset)
    train_size = int(0.6 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Get subset (first 2 samples for test)
    train_subset_new = torch.utils.data.Subset(train_dataset, range(2))

    # Check subset length
    subset_size = len(train_subset_new)
    assert subset_size == 2, f"Expected subset size of 2, got {subset_size}"

    # Also confirm that it's smaller than or equal to the original train size
    assert subset_size <= train_size, "Subset size should not exceed train dataset size"


@patch("os.path.exists", return_value=False)
@patch("kagglehub.dataset_download", side_effect=Exception("Download failed"))
def test_data_load_exceptions(mock_path_exists, mock_kagglehub_download):
    """
    Test that the dataset load fails gracefully when the dataset is unavailable.
    """
    with pytest.raises(RuntimeError, match="Dataset download failed"):
        load_data()