# src/utils.py

import os
import cv2
import numpy as np

def load_image(image_path):
    """
    Load an image from the specified path.
    
    Parameters:
    - image_path (str): The path to the image file.
    
    Returns:
    - image (ndarray): Loaded image in BGR format.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.imread(image_path)

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model input.
    
    Parameters:
    - image (ndarray): The image to preprocess.
    - target_size (tuple): Desired size of the image.
    
    Returns:
    - preprocessed_image (ndarray): Resized and normalized image.
    """
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

def save_results(results, save_directory, filename):
    """
    Save the results to a specified directory.
    
    Parameters:
    - results: The results to save (e.g., numpy array, list).
    - save_directory (str): Directory to save the results.
    - filename (str): Name of the file.
    
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    filepath = os.path.join(save_directory, filename)
    np.save(filepath, results)  # Save as numpy array
    print(f"Results saved to: {filepath}")
import os
import cv2
import numpy as np

def load_image(image_path):
    """
    Load an image from the specified path.
    
    Parameters:
    - image_path (str): The path to the image file.
    
    Returns:
    - image (ndarray): Loaded image in BGR format.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.imread(image_path)

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model input.
    
    Parameters:
    - image (ndarray): The image to preprocess.
    - target_size (tuple): Desired size of the image.
    
    Returns:
    - preprocessed_image (ndarray): Resized and normalized image.
    """
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

def save_results(results, save_directory, filename):
    """
    Save the results to a specified directory.
    
    Parameters:
    - results: The results to save (e.g., numpy array, list).
    - save_directory (str): Directory to save the results.
    - filename (str): Name of the file.
    
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    filepath = os.path.join(save_directory, filename)
    np.save(filepath, results)  # Save as numpy array
    print(f"Results saved to: {filepath}")

def validate_image(image):
    """
    Validate the image.
    
    Parameters:
    - image (ndarray): The image to validate.
    
    Returns:
    - bool: Whether the image is valid.
    """
    if image is None:
        return False
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False
    return True

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess the image.
    
    Parameters:
    - image_path (str): The path to the image file.
    - target_size (tuple): Desired size of the image.
    
    Returns:
    - preprocessed_image (ndarray): Resized and normalized image.
    """
    image = load_image(image_path)
    if not validate_image(image):
        raise ValueError(f"Invalid image: {image_path}")
    return preprocess_image(image, target_size)