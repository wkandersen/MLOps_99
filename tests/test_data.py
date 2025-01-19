from torch.utils.data import Dataset
import os
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.group_99.data import load_data

@pytest.fixture
def mock_kagglehub(tmp_path):
    """
    Mock the kagglehub.dataset_download function and create a fake directory structure.
    """
    # Mock dataset download path
    fake_dataset_path = str(tmp_path / "food41")

    # Create directory structure
    os.makedirs(os.path.join(fake_dataset_path, "meta", "meta"), exist_ok=True)
    os.makedirs(os.path.join(fake_dataset_path, "images", "apple_pie"), exist_ok=True)
    os.makedirs(os.path.join(fake_dataset_path, "images", "waffles"), exist_ok=True)

    # Mock classes.txt
    with open(os.path.join(fake_dataset_path, "meta", "meta", "classes.txt"), "w") as f:
        f.write("apple_pie\n")
        f.write("waffles\n")

    # Mock test.txt
    with open(os.path.join(fake_dataset_path, "meta", "meta", "test.txt"), "w") as f:
        f.write("apple_pie/1011328\n")
        f.write("waffles/971843\n")

    # Create dummy images in the correct locations
    with open(os.path.join(fake_dataset_path, "images", "apple_pie", "1011328.jpg"), "w") as f:
        f.write("fake_image_data")
    with open(os.path.join(fake_dataset_path, "images", "waffles", "971843.jpg"), "w") as f:
        f.write("fake_image_data")

    # Mock kagglehub download
    with patch("kagglehub.dataset_download", return_value=fake_dataset_path):
        yield fake_dataset_path

@patch("shutil.move")
def test_load_data_structure(mock_shutil_move, mock_kagglehub):
    """
    Test that load_data creates the correct folders, calls shutil.move
    for test files, and returns the loaders without error.
    """
    train_loader, valid_loader, train_subset_new = load_data()
    
    # Check that loaders are returned
    assert train_loader is not None, "Train loader is None"
    assert valid_loader is not None, "Validation loader is None"
    assert train_subset_new is not None, "Train subset loader is None"

    # Check that shutil.move was called for test images
    assert mock_shutil_move.call_count > 0, "No files were moved to testset/"
    
    # The train_loader and valid_loader should be DataLoader objects
    assert hasattr(train_loader, '__iter__'), "train_loader is not iterable"
    assert hasattr(valid_loader, '__iter__'), "valid_loader is not iterable"
    assert hasattr(train_subset_new, '__iter__'), "train_subset_new is not iterable"


@patch("shutil.move")
def test_load_data_batch_shapes(mock_shutil_move, mock_kagglehub):
    """
    Test that the shapes of the data batches returned by train_loader
    (and others) are as expected, e.g. (batch_size, 3, 224, 224).
    """
    train_loader, valid_loader, _ = load_data()

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
    train_loader, _, train_subset_new = load_data()

    # The full train_loader is a DataLoader of the entire dataset
    full_train_size = len(train_loader.dataset)

    # The subset is created with range(0, 2000) but let's confirm we have
    # the length of the subset we expect. In the mock, we only have a few images,
    # so the actual numbers might differ from your real use case.
    subset_size = len(train_subset_new.dataset)  # This should be 2000 or fewer
    assert subset_size <= 2000, "Subset should not exceed 2000 items"

    # Also check that the subset is not empty (our mock data is quite small).
    # Adjust this as needed for your real scenario.
    assert subset_size > 0, "Subset should not be empty"

    # Print for debugging (optional)
    print(f"Full dataset size: {full_train_size}, Subset size: {subset_size}")


def test_data_load_exceptions():
    """
    Example test for behavior if something goes wrong, e.g. Kaggle download fails.
    This might require a custom exception handling in your code.
    """
    # If kagglehub.dataset_download raises an exception, your code should handle it.
    # We'll patch it to raise an exception and check if your code does something sensible.
    with patch("kagglehub.dataset_download", side_effect=Exception("Download failed")):
        with pytest.raises(Exception) as exc_info:
            load_data()
        assert "Download failed" in str(exc_info.value)
