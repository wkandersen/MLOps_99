# from contextlib import asynccontextmanager
# import torch
# from fastapi import FastAPI, File, UploadFile
# from PIL import Image
# from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Load and clean up classification model on startup and shutdown."""
#     global model, feature_extractor, device
#     print("Loading classification model")
    
#     # Load your classification model and feature extractor
#     model = AutoModelForImageClassification.from_pretrained("your-model-name")  # Replace with your model's name/path
#     feature_extractor = AutoFeatureExtractor.from_pretrained("your-model-name")  # Replace with your feature extractor's path
    
#     # Set device (GPU or CPU)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     yield  # Keep the app running while the model is loaded

#     print("Cleaning up")
#     del model, feature_extractor, device


# app = FastAPI(lifespan=lifespan)


# @app.post("/classify/")
# async def classify(data: UploadFile = File(...)):
#     """Classify an image into categories."""
#     # Load and preprocess the image
#     i_image = Image.open(data.file)
#     if i_image.mode != "RGB":
#         i_image = i_image.convert(mode="RGB")

#     # Convert the image into input tensors
#     inputs = feature_extractor(images=[i_image], return_tensors="pt")
#     inputs = {key: value.to(device) for key, value in inputs.items()}

#     # Perform inference
#     outputs = model(**inputs)
#     probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     predicted_label_idx = torch.argmax(probabilities, dim=-1).item()

#     # Get label names (if available in your model's configuration)
#     labels = model.config.id2label  # `id2label` is a dictionary mapping indices to class names
#     predicted_label = labels[predicted_label_idx]

#     return {"label": predicted_label, "confidence": probabilities[0][predicted_label_idx].item()}

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, feature_extractor, tokenizer, device, gen_kwargs
    print("Loading model")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up")
    del model, feature_extractor, tokenizer, device, gen_kwargs


app = FastAPI(lifespan=lifespan)


@app.post("/caption/")
async def caption(data: UploadFile = File(...)):
    """Generate a caption for an image."""
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [pred.strip() for pred in preds]

