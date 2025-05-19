import torch
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import hydra
from omegaconf import DictConfig
from typing import List, Union, Tuple

from .model import LatexOCRModel
from .utils import preprocess_image

class LatexOCRPredictor:
    def __init__(self, model_path, device=None, config=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Load model
        self.model = LatexOCRModel.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Set up image transforms
        self.transform = T.Compose([
            T.Resize((self.config.data.img_size, self.config.data.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def predict(self, image: Union[str, Image.Image]) -> str:
        # Load and preprocess image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = image
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Generate prediction
        output_tokens = self.model(img_tensor)
        
        # Convert tokens to LaTeX string
        latex_code = self._tokens_to_latex(output_tokens[0])
        
        return latex_code
    
    @torch.no_grad()
    def predict_batch(self, images: List[Union[str, Image.Image]]) -> List[str]:
        batch_tensors = []
        
        # Preprocess all images
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
                
            img_tensor = self.transform(img)
            batch_tensors.append(img_tensor)
        
        # Stack to batch
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        # Generate predictions
        output_tokens = self.model(batch_tensor)
        
        # Convert tokens to LaTeX strings
        latex_codes = [self._tokens_to_latex(tokens) for tokens in output_tokens]
        
        return latex_codes
    
    def _tokens_to_latex(self, tokens):
        # This is a placeholder - in real implementation would use tokenizer
        # to convert token IDs back to text
        return "placeholder_latex"

@hydra.main(config_path="../", config_name="config")
def run_inference(config: DictConfig):
    model_path = config.inference.model_path
    image_path = config.inference.image_path
    output_dir = config.inference.output_dir
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = LatexOCRPredictor(model_path, config=config)
    
    # Check if image_path is a directory or a file
    image_path = Path(image_path)
    if image_path.is_dir():
        # Process all images in directory
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
        for img_file in image_files:
            print(f"Processing {img_file}")
            latex_code = predictor.predict(str(img_file))
            
            # Save output
            output_file = Path(output_dir) / f"{img_file.stem}.tex"
            with open(output_file, 'w') as f:
                f.write(latex_code)
    else:
        # Process single image
        latex_code = predictor.predict(str(image_path))
        
        # Save output
        output_file = Path(output_dir) / f"{image_path.stem}.tex"
        with open(output_file, 'w') as f:
            f.write(latex_code)

if __name__ == "__main__":
    run_inference()
