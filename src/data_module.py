# src/data_module.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision.transforms as T
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple
import numpy as np

def build_vocab(annotations_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary from annotations file."""
    with open(annotations_file, "r") as f:
        formulas = [line.strip() for line in f if line.strip()]
    
    # Collect all unique characters
    chars = sorted(set("".join(formulas)))
    
    # Special tokens: 0-pad, 1-sos, 2-eos
    idx2char = {i+3: c for i, c in enumerate(chars)}
    char2idx = {c: i+3 for i, c in enumerate(chars)}
    
    return char2idx, idx2char

class LatexDataset(Dataset):
    def __init__(
        self, 
        image_dir: str, 
        annotations_file: str, 
        char2idx: Dict[str, int], 
        transform=None, 
        max_seq_len: int = 512
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.char2idx = char2idx

        # Parse annotations and pair with images
        self.data = []
        with open(annotations_file, 'r') as f:
            for idx, line in enumerate(f):
                formula = line.strip()
                if not formula:
                    continue
                
                # Формат файла: 0000000.png, 0000001.png, ...
                image_path = os.path.join(image_dir, f"{idx:07d}.png")  # 7 цифр с ведущими нулями
                if not os.path.exists(image_path):
                    continue
                    
                self.data.append((image_path, formula))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        image_path, formula = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Tokenize formula
        tokens = [self.char2idx.get(c, 0) for c in formula]

        if len(tokens) > self.max_seq_len - 2:  # -2 для <sos> и <eos>
            raise ValueError(f"Формула слишком длинная: {formula}")
        
        tokens = [self.char2idx['<sos>']] + tokens + [self.char2idx['<eos>']]
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return {
            'image': image,
            'formula': tokens,
            'image_path': image_path,
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader to handle variable-length sequences."""
    images = torch.stack([item['image'] for item in batch])
    formulas = [item['formula'] for item in batch]
    formulas_padded = pad_sequence(formulas, batch_first=True, padding_value=0)
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'image': images,
        'formula': formulas_padded,
        'image_path': image_paths,
    }

class LatexOCRDataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir: str = "data", 
        batch_size: int = 32, 
        num_workers: int = 4, 
        train_val_test_split: List[float] = [0.8, 0.1, 0.1],
        img_size: int = 224,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.max_seq_len = max_seq_len

        # Build vocabulary
        annotations_file = os.path.join(self.data_dir, "annotations.txt")
        self.char2idx, self.idx2char = build_vocab(annotations_file)
        
        # Add special tokens to vocab
        self.char2idx['<pad>'] = 0
        self.char2idx['<sos>'] = 1
        self.char2idx['<eos>'] = 2
        self.idx2char[0] = '<pad>'
        self.idx2char[1] = '<sos>'
        self.idx2char[2] = '<eos>'

        # Image transforms
        self.train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomRotation(2),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage: str = None):
        """Setup datasets for each stage."""
        annotations_file = os.path.join(self.data_dir, "annotations.txt")
        image_dir = os.path.join(self.data_dir, "images")
        
        full_dataset = LatexDataset(
            image_dir=image_dir,
            annotations_file=annotations_file,
            char2idx=self.char2idx,
            transform=self.train_transform,
            max_seq_len=self.max_seq_len
        )
        
        # Split dataset
        n = len(full_dataset)
        train_size = int(n * self.train_val_test_split[0])
        val_size = int(n * self.train_val_test_split[1])
        test_size = n - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Apply different transforms for validation/test
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.val_transform

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )