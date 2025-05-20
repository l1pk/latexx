import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision.transforms as T
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

def build_vocab(annotations_file):
    with open(annotations_file, "r") as f:
        formulas = [line.strip().split(' ', 1)[1] for line in f if len(line.strip().split(' ', 1)) == 2]
    chars = sorted(set("".join(formulas)))
    idx2char = {i+3: c for i, c in enumerate(chars)}  # 0-pad, 1-sos, 2-eos
    char2idx = {c: i+3 for i, c in enumerate(chars)}
    return char2idx, idx2char

class LatexDataset(Dataset):
    def __init__(self, image_dir, annotations_file, char2idx, transform=None, max_seq_len=512):
        self.image_dir = image_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.char2idx = char2idx

        # Parse annotations
        self.data = self.data[:1]
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
        self.data.sort(key=lambda x: x[2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, formula, _ = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Токенизация формулы
        tokens = [1]  # sos_token_id
        tokens += [self.char2idx.get(c, 0) for c in formula]
        tokens.append(2)  # eos_token_id
        tokens = torch.tensor(tokens, dtype=torch.long)
        return {
            'image': image,
            'formula': tokens,
            'image_path': image_path,
        }

def collate_fn(batch):
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
    def __init__(self, data_dir="data", batch_size=8, num_workers=4, 
                 train_val_test_split=[0.8, 0.1, 0.1], img_size=224,
                 max_seq_len=512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.max_seq_len = max_seq_len

        # Словарь токенов
        annotations_file = os.path.join(self.data_dir, "annotations.txt")
        self.char2idx, self.idx2char = build_vocab(annotations_file)

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
        full_dataset = LatexDataset(
            image_dir=image_dir,
            annotations_file=annotations_file,
            char2idx=self.char2idx,
            transform=self.train_transform,
            max_seq_len=self.max_seq_len
        )
        total_size = len(full_dataset)
        train_size = int(self.train_val_test_split[0] * total_size)
        val_size = int(self.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        # Обновим transform для валидации и теста
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.val_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
