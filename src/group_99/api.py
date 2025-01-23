from fastapi import FastAPI, File, UploadFile
from PIL import Image
from contextlib import asynccontextmanager
import torch
from src.group_99.data import load_data
from model import ConvolutionalNetwork

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device, transform, class_names
    print("Loading model")
    # Define checkpoint path
    checkpoint_path = "models/bestmodel.ckpt"

    # Load data and initialize the datamodule
    data, transform, class_names, path = load_data()

    # Load the classification model from checkpoint
    model = ConvolutionalNetwork.load_from_checkpoint(
        checkpoint_path, class_names=class_names, lr=0.001
    )

    # Force use of CPU
    device = torch.device("cpu")
    model.to(device)  # Ensure the model is on CPU
    model.eval()  # Set the model to evaluation mode

    yield

    print("Cleaning up")
    del model, device, transform, class_names


app = FastAPI(lifespan=lifespan)


@app.post("/classify/")
async def classify(data: UploadFile = File(...)):
    """Classify an image."""
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    # Preprocess the image using the transform defined in load_data
    image_tensor = transform(i_image).unsqueeze(0).to(device)  # Ensure tensor is on CPU

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class_idx = probabilities.argmax().item()
        predicted_class = class_names[predicted_class_idx]

    return {"predicted_class": predicted_class, "probabilities": probabilities.tolist()}
