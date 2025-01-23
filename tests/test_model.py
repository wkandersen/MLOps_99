import pytest
import torch
from src.group_99.model import TimmModel
from torch.utils.data import DataLoader, TensorDataset


def create_dummy_data(num_samples=100, num_classes=10, input_size=(3, 224, 224)):
    X = torch.rand(num_samples, *input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return dataset

@pytest.fixture
def dummy_dataloader():
    dataset = create_dummy_data(num_samples=50, num_classes=5)
    dataloader = DataLoader(dataset, batch_size=10)
    return dataloader

@pytest.fixture
def timm_model():
    model = TimmModel(class_names=5, model_name="resnet18", lr=0.001, dropout=0.2)
    
    # Ensure model is properly initialized by checking classifier structure
    assert isinstance(model.classifier, torch.nn.Sequential), "Classifier is not a Sequential model"
    assert hasattr(model.classifier[1], 'in_features'), "Classifier's Linear layer doesn't have 'in_features'"
    
    return model

def test_forward_pass(timm_model):
    X = torch.rand(4, 3, 224, 224)  # Dummy input data
    output = timm_model(X)
    assert output.shape == (4, 5), "Output shape mismatch"
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"

def test_training_step(timm_model, dummy_dataloader):
    batch = next(iter(dummy_dataloader))
    loss = timm_model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor), "Training step loss is not a tensor"
    assert loss > 0, "Loss should be greater than zero"

def test_validation_step(timm_model, dummy_dataloader):
    batch = next(iter(dummy_dataloader))
    timm_model.validation_step(batch, batch_idx=0)
    # No assertion needed as validation logs are handled by PyTorch Lightning

def test_test_step(timm_model, dummy_dataloader):
    batch = next(iter(dummy_dataloader))
    timm_model.test_step(batch, batch_idx=0)
    # No assertion needed as test logs are handled by PyTorch Lightning

def test_optimizer_configuration(timm_model):
    optimizer = timm_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer), "Optimizer is not configured correctly"

if __name__ == "__main__":
    pytest.main()
