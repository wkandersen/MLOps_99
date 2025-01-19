import torch
import pytest
from src.group_99.model import ConvolutionalNetwork

def test_model_output_shape():
    # Define the number of classes based on your dataset
    num_classes = 23
    
    # Fixed input size
    input_height = 224
    input_width = 224
    
    # Assume the class names correspond to the number of classes (for example 10 classes)
    class_names = [f'class_{i}' for i in range(num_classes)]
    
    # Instantiate the model
    model = ConvolutionalNetwork(class_names)
    
    # Create a random input tensor with the shape (batch_size, channels, height, width)
    input_tensor = torch.randn(8, 3, input_height, input_width)  # Batch size of 8, 3 channels (RGB), 224x224 images
    
    # Get the output of the model
    output = model(input_tensor)
    
    # Check if the output shape is as expected (batch_size, num_classes)
    assert output.shape == (8, num_classes), f"Expected output shape (8, {num_classes}), but got {output.shape}"

def test_model_forward_pass():
    # Define the number of classes based on your dataset
    num_classes = 10  # Example for a classification problem with 10 classes
    
    # Assume a fixed input size (224x224, as given in the test description)
    input_height = 224
    input_width = 224
    
    # Assume the class names correspond to the number of classes (for example 10 classes)
    class_names = [f'class_{i}' for i in range(num_classes)]
    
    # Instantiate the model
    model = ConvolutionalNetwork(class_names)
    
    # Create a random input tensor with the shape (batch_size, channels, height, width)
    input_tensor = torch.randn(8, 3, input_height, input_width)  # Batch size of 8, 3 channels (RGB), 224x224 images
    
    try:
        # Perform a forward pass
        output = model(input_tensor)
    except Exception as e:
        pytest.fail(f"Forward pass raised an exception: {e}")