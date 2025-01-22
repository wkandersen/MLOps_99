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
    data, transform, class_names, dataset_path = load_data()

    dataset = CustomDataset(data, transform)

    dataset_size = len(dataset)
    train_size = int(0.6 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch = next(iter(train_loader))
    inputs, labels = batch

    assert len(inputs.shape) == 3, "Inputs should be a 4D tensor"
    assert inputs.shape[1] == 3, "Channel dimension should be 3 for RGB images"
    assert inputs.shape[2] == 224 and inputs.shape[3] == 224, (
        "Images should be resized to 224x224"
    )
    assert labels.shape[0] == inputs.shape[0], "Labels batch size must match image batch size"
