import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.group_99.data import load_data, ImageDataModule
from src.group_99.model import TimmModel
from typing import List
from tqdm import tqdm

seed = 69420
torch.manual_seed(seed)

# Define checkpoint path
checkpoint_path = "models/best-model-epoch=04-val_loss=0.77.ckpt"

# Load data and initialize the datamodule
data, transform, class_names, path = load_data()
datamodule = ImageDataModule(data, transform, batch_size=128)

datamodule.setup()

# Load the model from checkpoint
model = TimmModel.load_from_checkpoint(checkpoint_path, class_names=23, num_features=3)

def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, class_names: List[str], device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    """
    Evaluate the model on a given dataloader and print performance metrics.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader containing evaluation data.
        class_names: List[str]: List of class names for the dataset.
        device: Device to run the evaluation on ('cuda' or 'cpu').
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get the predicted class

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    #save as txt
    with open('reports/figures/classification_report.txt', 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

def plot_confusion_matrix(cm: List[List[int]], class_names: List[str]) -> None:
    """
    Plot the confusion matrix using matplotlib and seaborn.

    Args:
        cm (List[List[int]]): Confusion matrix as a NumPy array.
        class_names: List of class names for the dataset.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("reports/figures/confusion_matrix.png")
    plt.show()

# Evaluate the model using the validation dataloader

if __name__ == "__main__":

    evaluate_model(model, datamodule.val_dataloader(), class_names)
