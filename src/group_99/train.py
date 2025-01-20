from data import load_data, ImageDataModule
from model import ConvolutionalNetwork
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def train(config: DictConfig):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters from config
    hparams = config.hyperparameters
    torch.manual_seed(hparams['seed'])

    # Load the data
    data, transform, class_names, path = load_data()

    # Initialize the data module
    datamodule = ImageDataModule(
        data=data,
        transform=transform,
        batch_size=hparams['batch_size']
    )

    # Define the model
    model = ConvolutionalNetwork(
        class_names=hparams['num_classes'],
        lr=hparams['lr']
    ).to(device)

    # wandb.init(project=config.sweep.project, entity=config.sweep.entity, config=hparams)
    wandb_logger = WandbLogger(project=config.sweep.project, entity=config.sweep.entity, config=hparams)
    # Set up the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",                # Metric to monitor
        dirpath="models/",                 # Directory to save checkpoints
        filename="best-model-{epoch:02d}-{val_loss:.2f}",  # Filename format
        save_top_k=1,                      # Save only the best model
        mode="min",                        # "min" because we want the lowest loss
        verbose=True                       # Print information about checkpointing
    )

    # Set up the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=hparams['epochs'],
        callbacks=[checkpoint_callback],   # Add the ModelCheckpoint callback
        logger=wandb_logger,               # Use WandB logger
        accelerator="gpu" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    )

    print(f"Training model for {hparams['epochs']} epochs...")

    # Train the model
    trainer.fit(model, datamodule)

    # Test the model using the validation set
    print("Testing the model...")
    trainer.test(model, dataloaders=datamodule.val_dataloader())

    # Print the path of the best model
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
