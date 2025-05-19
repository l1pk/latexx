import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchmetrics.text import BLEUScore, WordErrorRate
from typing import List, Tuple, Dict, Union

def preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """Preprocess an image for inference."""
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def compute_bleu(predictions: List[str], references: List[List[str]]) -> float:
    """Compute BLEU score between predictions and references."""
    bleu_metric = BLEUScore()
    return bleu_metric(predictions, references).item()

def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Compute Word Error Rate between predictions and references."""
    wer_metric = WordErrorRate()
    return wer_metric(predictions, references).item()

def tokenize_latex(formula: str, tokenizer=None) -> List[int]:
    """Tokenize LaTeX formula to token IDs."""
    # This is a placeholder - in real implementation would use tokenizer
    if tokenizer is None:
        # Return placeholder tokenization
        return [1] + [i % 100 for i in range(len(formula))] + [2]
    else:
        return tokenizer.encode(formula)

def detokenize_latex(token_ids: List[int], tokenizer=None) -> str:
    """Convert token IDs back to LaTeX formula."""
    # This is a placeholder - in real implementation would use tokenizer
    if tokenizer is None:
        # Return placeholder detokenization
        return "placeholder_formula"
    else:
        return tokenizer.decode(token_ids)

def visualize_attention(image: torch.Tensor, attention_weights: torch.Tensor, save_path: str = None):
    """Visualize attention weights on the image."""
    import matplotlib.pyplot as plt
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image.clone()
    img = img * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    # Resize attention to match image size
    h, w = img.shape[:2]
    attention = attention_weights.cpu().numpy()
    attention = np.mean(attention, axis=0)  # Average attention across heads
    
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Plot attention overlay
    ax[1].imshow(img)
    ax[1].imshow(attention, alpha=0.5, cmap='jet')
    ax[1].set_title("Attention Overlay")
    ax[1].axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig
