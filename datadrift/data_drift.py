import torch
from sklearn.metrics import accuracy_score, classification_report
from src.group_99.model import TimmModel
from src.group_99.data import load_data
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Load the model checkpoint
checkpoint_path = "models/best-model-epoch=04-val_loss=0.77.ckpt"




def load_model(checkpoint_path, num_classes):
    """Load the trained model from a checkpoint."""
    model = TimmModel.load_from_checkpoint(
        checkpoint_path, class_names=num_classes
    )
    model.eval()  # Set to evaluation mode
    return model


def simulate_drift(image):
    """Apply random transformations to simulate data drift."""
    drift_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(5, 5)),
    ])
    return drift_transform(image)


def evaluate_model_on_drift(model, data, transform):
    """Evaluate the model on original and drifted data."""
    original_preds = []
    drifted_preds = []
    labels = []

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating"):
        image_path, label = row['path'], row['label']
        labels.append(label)

        # Load and preprocess the original image
        image = Image.open(image_path).convert("RGB")
        original_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            original_pred = torch.argmax(model(original_tensor), dim=1).item()
            original_preds.append(original_pred)

        # Simulate drift and preprocess
        drifted_image = simulate_drift(image)
        drifted_tensor = transform(drifted_image).unsqueeze(0)

        with torch.no_grad():
            drifted_pred = torch.argmax(model(drifted_tensor), dim=1).item()
            drifted_preds.append(drifted_pred)

    # Calculate metrics
    original_acc = accuracy_score(labels, original_preds)
    drifted_acc = accuracy_score(labels, drifted_preds)

    print("\nClassification Report on Original Data:")
    print(classification_report(labels, original_preds))

    print("\nClassification Report on Drifted Data:")
    print(classification_report(labels, drifted_preds))

    return original_acc, drifted_acc


if __name__ == "__main__":
    # Load the data
    data, transform, class_names, dataset_path = load_data()

    # Load the model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, len(class_names))

    # Evaluate the model's robustness to data drift
    print("Evaluating model robustness to data drift...")
    original_acc, drifted_acc = evaluate_model_on_drift(model, data, transform)

    print(f"Accuracy on original data: {original_acc * 100:.2f}%")
    print(f"Accuracy on drifted data: {drifted_acc * 100:.2f}%")
