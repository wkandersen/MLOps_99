import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
import kagglehub


# @hydra.main(config_path="config", config_name="config.yaml", version_base="1.1")

def load_data():
    # Define the home directory
    home_dir = os.path.expanduser("~")

    # Define the dataset path based on the home directory
    dataset_path = os.path.join(home_dir, ".cache/kagglehub/datasets/vencerlanz09/sea-animals-image-dataste")

    # Check if the dataset already exists at the specified path
    if os.path.exists(dataset_path):
        print(f"Dataset found at {dataset_path}, using the existing dataset.")
        path = dataset_path
    else:
        print("Starting download")
        try:
            path = kagglehub.dataset_download("vencerlanz09/sea-animals-image-dataste")
            print(f"Dataset downloaded at {path}")
        except Exception as e:
            raise RuntimeError("Dataset download failed") from e

    classes=[]
    paths=[]
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.png') or filename.endswith('.jpg') :
                classes+=[dirname.split('/')[-1]]
                paths+=[(os.path.join(dirname, filename))]
    print(len(paths))


    N = list(range(len(classes)))
    class_names=sorted(set(classes))
    print(class_names)
    normal_mapping=dict(zip(class_names,N)) 
    reverse_mapping=dict(zip(N,class_names))       

    data=pd.DataFrame(columns=['path','class','label'])
    data['path']=paths
    data['class']=classes
    data['label']=data['class'].map(normal_mapping)
    m=len(data)
    M=list(range(m))
    random.shuffle(M)
    data=data.iloc[M]


    transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                             ])

    return data, transform, class_names


def create_path_label_list(df):
    path_label_list = []
    for _, row in df.iterrows():
        path = row['path']
        label = row['label']
        path_label_list.append((path, label))
    return path_label_list


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_label, transform=None):
        self.path_label = path_label
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    

class ImageDataset(pl.LightningDataModule):
    def __init__(self, path_label, batch_size=32):
        super().__init__()
        self.path_label = path_label
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(224),             # resize shortest side to 224 pixels
            transforms.CenterCrop(224),         # crop longest side to 224 pixels at center            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        dataset = CustomDataset(self.path_label, self.transform)
        dataset_size = len(dataset)
        train_size = int(0.6 * dataset_size) 
        val_size = dataset_size - train_size
        print(train_size,val_size)

        self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))

    def __len__(self):
        if self.train_dataset is not None:
            return len(self.train_dataset)
        elif self.val_dataset is not None:
            return len(self.val_dataset)
        else:
            return 0        

    def __getitem__(self, index):
        if self.train_dataset is not None:
            return self.train_dataset[index]
        elif self.test_dataset is not None:
            return self.test_dataset[index]
        else:
            raise IndexError("Index out of range. The dataset is empty.")

    def train_dataset(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataset(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    load_data()
