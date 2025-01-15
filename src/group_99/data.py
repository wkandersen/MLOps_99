import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import resnet50
import shutil
import hydra
import kagglehub


# @hydra.main(config_path="config", config_name="config.yaml", version_base="1.1")


def load_data():

    # hparams = config['hyperparameters']   
    print("Starting download")
    # Download latest version
    path = kagglehub.dataset_download("kmader/food41")
    print("Downloaded to", path)
    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    fl = open(path + '/meta/meta/classes.txt')
    cls = fl.readline().strip()

    while(cls):
        os.makedirs(path + f'/testset/{cls}', exist_ok=True)
        cls = fl.readline().strip()
    # Moving test files to testset/, train files will be left.
    testfile = open(path + '/meta/meta/test.txt')
    img = testfile.readline().strip()
    batch_size = 100
    batch = []

    while(img):
        cls = img.split('/')[0]
        path1 = os.path.join(path, f'images/', (img + '.jpg'))
        path2 = os.path.join(path, f"testset/", (img + '.jpg'))
        if (os.path.exists(path1)):
            batch.append((path1, path2))
            if len(batch) >= batch_size:
                for src, dst in batch:
                    shutil.move(src, dst)
                batch = []
                print(f'Moved {len(batch)} files')
        img = testfile.readline().strip()

    # Move remaining files in the batch
    if batch:
        for src, dst in batch:
            shutil.move(src, dst)
        print(f'\rMoved remaining {len(batch)} files', end='')


    train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1),
            torchvision.transforms.RandomAffine(15),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    valid_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


    train_dataset = torchvision.datasets.ImageFolder(path + '/images/', transform=train_transforms)
    valid_dataset = torchvision.datasets.ImageFolder(path + '/testset/', transform=valid_transforms)

    train_subset = torch.utils.data.Subset(train_dataset, range(0, 2000))
    train_subset_new = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True,num_workers=4,pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size,shuffle=False,num_workers=4,pin_memory=True)
    # Get the first batch from the DataLoader
    inputs, labels = next(iter(train_loader))

    # Print the shape of the inputs (i.e., the image tensor)
    print("Input size (batch_size, channels, height, width):", inputs.shape)

    batch_size, channels, height, width = inputs.shape
    flattened_inputs = inputs.view(batch_size, -1)  # Flatten everything except the batch dimension

    # Print the shape of the flattened input
    print("Flattened input shape:", flattened_inputs.shape)

    return train_loader, valid_loader, train_subset_new
    
if __name__ == "__main__":
    load_data()