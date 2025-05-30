# src/utils.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchmetrics.text import BLEUScore, CharErrorRate
from typing import List, Tuple, Dict, Union, Optional

def preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """Preprocess image for inference."""
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU and CER metrics."""
    bleu = BLEUScore()(predictions, [[ref] for ref in references])
    cer = CharErrorRate()(predictions, references)
    return {"bleu": bleu.item(), "cer": cer.item()}

def visualize_attention(
    image: torch.Tensor, 
    attention: torch.Tensor, 
    save_path: Optional[str] = None
):
    """Visualize attention weights on image."""
    import matplotlib.pyplot as plt
    
    # Denormalize image
    img = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")
    
    ax[1].imshow(img)
    ax[1].imshow(attention.cpu().numpy(), alpha=0.5, cmap="hot")
    ax[1].set_title("Attention")
    ax[1].axis("off")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig