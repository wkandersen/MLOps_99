from src.group_99.data import load_data, ImageDataModule
from src.group_99.model import TimmModel
import torch
import pytorch_lightning as pl
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
import wandb


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def train(config: DictConfig) -> None:
    # Check if CUDA is available
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.login()

    hparams: DictConfig = config.hyperparameters

    # Unpack the hyperparameters
    seed: int = hparams["seed"]
    batch_size: int = hparams["batch_size"]
    num_classes: int = hparams["num_classes"]
    lr: float = hparams["lr"]
    epochs: int = hparams["epochs"]

    # Set the random seed
    torch.manual_seed(seed)

    # Load the data
    data, transform, class_names, path = load_data()

    # Initialize the data module
    datamodule: ImageDataModule = ImageDataModule(
        data=data,
        transform=transform,
        batch_size=batch_size  # Pass the batch size from sweep configuration
    )

    # Define the model
    model: TimmModel = TimmModel(
        class_names=num_classes,
        lr=lr,  # Use learning rate from sweep configuration
        num_features=data.shape[1]  # Use the appropriate dimension from data.shape
    ).to(device)

    # Set up the WandB logger
    wandb_logger: WandbLogger = WandbLogger(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)

    # Set up the ModelCheckpoint callback
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        monitor="val_loss",                # Metric to monitor
        dirpath="models/",                 # Directory to save checkpoints
        filename="best-model-{epoch:02d}-{val_loss:.2f}",  # Filename format
        save_top_k=1,                      # Save only the best model
        mode="min",                        # "min" because we want the lowest loss
        verbose=True                       # Print information about checkpointing
    )

    # Set up the PyTorch Lightning trainer
    trainer: pl.Trainer = pl.Trainer(
        max_epochs=epochs,  # Use the epochs from sweep configuration
        callbacks=[checkpoint_callback],   # Add the ModelCheckpoint callback
        logger=wandb_logger,               # Use WandB logger
        accelerator="gpu" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    )

    print(f"Training model for {epochs} epochs...")

    # Train the model
    trainer.fit(model, datamodule)

    # Test the model using the validation set
    print("Testing the model...")
    trainer.test(model, dataloaders=datamodule.val_dataloader())

    # Print the path of the best model
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train()
