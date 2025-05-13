import torch
from model import FormulaModel

def load_model(checkpoint_path):
    model = FormulaModel.load_from_checkpoint(checkpoint_path)
    return model

def predict(image, model):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
    return output.argmax(dim=1)  # Return predicted class
