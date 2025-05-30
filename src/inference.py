# src/inference.py
import torch
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import hydra
from omegaconf import DictConfig
from typing import List, Union, Optional

from model import LatexOCRModel
from data_module import LatexOCRDataModule

class LatexOCRPredictor:
    def __init__(
        self, 
        model_path: str, 
        config: Optional[DictConfig] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Load model
        self.model = LatexOCRModel.load_from_checkpoint(model_path).to(self.device)
        self.model.eval()
        
        # Set up transforms
        self.transform = T.Compose([
            T.Resize((self.config.data.img_size, self.config.data.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def predict(self, image: Union[str, Image.Image]) -> str:
        """Predict LaTeX from single image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        tokens = self.model.generate(self.model.encoder(image_tensor))
        return self.model._tokens_to_text(tokens)[0]
    
    @torch.no_grad()
    def predict_batch(self, images: List[Union[str, Image.Image]]) -> List[str]:
        """Predict LaTeX from batch of images."""
        image_tensors = []
        
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            image_tensors.append(self.transform(img))
        
        image_tensor = torch.stack(image_tensors).to(self.device)
        tokens = self.model.generate(self.model.encoder(image_tensor))
        return self.model._tokens_to_text(tokens)

@hydra.main(config_path="../", config_name="config")
def run_inference(config: DictConfig):
    predictor = LatexOCRPredictor(
        model_path=config.inference.model_path,
        config=config
    )
    
    image_path = Path(config.inference.image_path)
    output_dir = Path(config.inference.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if image_path.is_dir():
        # Process all images in directory
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
        for img_file in image_files:
            latex = predictor.predict(str(img_file))
            output_file = output_dir / f"{img_file.stem}.tex"
            output_file.write_text(latex)
    else:
        # Process single image
        latex = predictor.predict(str(image_path))
        output_file = output_dir / f"{image_path.stem}.tex"
        output_file.write_text(latex)

if __name__ == "__main__":
    run_inference()