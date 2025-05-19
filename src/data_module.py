import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision.transforms as T
from PIL import Image
import numpy as np

class LatexDataset(Dataset):
    def __init__(self, image_dir, annotations_file, transform=None, max_seq_len=512):
        self.image_dir = image_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        # Parse annotations
        self.data = []
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
                image_name = parts[0]
                formula = parts[1]
                image_id = int(image_name.replace('image', ''))
                image_path = os.path.join(image_dir, f"{image_name}.jpg")
                self.data.append((image_path, formula, image_id))
        
        # Sort by image ID for reproducibility
        self.data.sort(key=lambda x: x[2])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, formula, _ = self.data[idx]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Return image and formula
        return {
            'image': image,
            'formula': formula,
            'image_path': image_path,
        }

class LatexOCRDataModule(LightningDataModule):
    def __init__(self, data_dir="data", batch_size=8, num_workers=4, 
                 train_val_test_split=[0.8, 0.1, 0.1], img_size=224,
                 max_seq_len=512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.max_seq_len = max_seq_len
        
        # Define transforms
        self.train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomRotation(2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage=None):
        annotations_file = os.path.join(self.data_dir, "annotations.txt")
        image_dir = os.path.join(self.data_dir, "images")
        
        # Create dataset
        full_dataset = LatexDataset(
            image_dir=image_dir,
            annotations_file=annotations_file,
            transform=self.train_transform,
            max_seq_len=self.max_seq_len
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(self.train_val_test_split[0] * total_size)
        val_size = int(self.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update transforms for validation and test sets
        self.val_dataset.dataset = LatexDataset(
            image_dir=image_dir,
            annotations_file=annotations_file,
            transform=self.val_transform,
            max_seq_len=self.max_seq_len
        )
        
        self.test_dataset.dataset = LatexDataset(
            image_dir=image_dir,
            annotations_file=annotations_file,
            transform=self.val_transform,
            max_seq_len=self.max_seq_len
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
