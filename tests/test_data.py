import os
import pytest
import torch
from unittest.mock import patch
from torch.utils.data import DataLoader
from src.group_99.data import load_data, CustomDataset

@pytest.fixture
def mock_kagglehub(tmp_path):
    """
    Mock the kagglehub.dataset_download function and create a fake directory structure.
    """
    fake_dataset_path = str(tmp_path / "sea_animals")
    os.makedirs(os.path.join(fake_dataset_path, "class1"), exist_ok=True)
    os.makedirs(os.path.join(fake_dataset_path, "class2"), exist_ok=True)

    with open(os.path.join(fake_dataset_path, "class1", "1.jpg"), "w") as f:
        f.write("fake_image_data")
    with open(os.path.join(fake_dataset_path, "class2", "2.jpg"), "w") as f:
        f.write("fake_image_data")

    # Ensure that kagglehub is mocked to return the fake dataset path
    with patch("kagglehub.dataset_download", return_value=fake_dataset_path):
        yield fake_dataset_path


@patch("shutil.move")
def test_load_data_batch_shapes(mock_shutil_move, mock_kagglehub):
    """
    Test that the shapes of the data batches returned by train_loader
    (and others) are as expected, e.g. (batch_size, 3, 224, 224).
    """
    # Load the data, and ensure the mock_kagglehub fixture is used to provide the dataset path
    data, transform, class_names, dataset_path = load_data()

    # Create the CustomDataset instance
    dataset = CustomDataset(data, transform)

    # Split the dataset into train and validation sets
    dataset_size = len(dataset)
    train_size = int(0.6 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Get a batch of data
    batch = next(iter(train_loader))
    inputs, labels = batch

    # Assertions to verify the shape of the input batch
    assert len(inputs.shape) == 4, "Inputs should be a 4D tensor (batch_size, channels, height, width)"
    assert inputs.shape[1] == 3, "Channel dimension should be 3 for RGB images"
    assert inputs.shape[2] == 224 and inputs.shape[3] == 224, (
        "Images should be resized to 224x224"
    )
    assert labels.shape[0] == inputs.shape[0], "Labels batch size must match image batch size"

@patch("shutil.move")
def test_load_data_subset(mock_shutil_move, mock_kagglehub):
    """
    Test that the subset data loader (train_subset_new) has the correct length
    and is indeed smaller than the full training dataset.
    """
    data, transform, class_names, dataset_path = load_data()

    dataset = CustomDataset(data, transform)

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
    pytest.main(["-v", __file__])
    

