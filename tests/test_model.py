from omegaconf import OmegaConf
import pytest
import torch
from src.group_99.model import ConvolutionalNetwork

@pytest.fixture
def config():
    config_path = "src/group_99/config/config.yaml"
    return OmegaConf.load(config_path)

def test_model_output_shape(config):
    hparams = config.hyperparameters
    torch.manual_seed(hparams['num_classes'])
    torch.manual_seed(hparams['batch_size'])
    
    num_classes = hparams['num_classes']
    batch_size = hparams['batch_size']
    
    input_height = 224
    input_width = 224
    
    # Directly pass num_classes, not a list of class names
    model = ConvolutionalNetwork(num_classes)
    device = torch.device('cpu')
    model.to(device)
    
    input_tensor = torch.randn(batch_size, 3, input_height, input_width, device=device)
    
    output = model(input_tensor)
    
    assert output.shape == (batch_size, num_classes), f"Expected output shape ({batch_size}, {num_classes}), but got {output.shape}"

def test_model_forward_pass(config):
    hparams = config.hyperparameters
    torch.manual_seed(hparams['num_classes'])
    torch.manual_seed(hparams['batch_size'])
    
    num_classes = hparams['num_classes']
    batch_size = hparams['batch_size']
    
    input_height = 224
    input_width = 224
    
    # Directly pass num_classes, not a list of class names
    model = ConvolutionalNetwork(num_classes)
    device = torch.device('cpu')
    model.to(device)
    
    input_tensor = torch.randn(batch_size, 3, input_height, input_width, device=device) 
    
    try:
        output = model(input_tensor)
    except Exception as e:
        pytest.fail(f"Forward pass raised an exception: {e}")