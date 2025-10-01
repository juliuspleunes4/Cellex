"""
Prediction utilities for trained model
"""

import torch
from PIL import Image
from torchvision import transforms
import numpy as np


class CellexPredictor:
    """Class for making predictions on X-ray images"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        from ml_model.model import create_model
        self.model = create_model(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = ['Normal', 'Potentially Cancerous']
    
    def predict(self, image_path):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Prepare results
        pred_class = predicted_class.item()
        pred_confidence = confidence.item()
        pred_probs = probabilities.cpu().numpy()[0]
        
        results = {
            'class': self.class_names[pred_class],
            'class_id': pred_class,
            'confidence': float(pred_confidence),
            'probabilities': {
                'normal': float(pred_probs[0]),
                'cancerous': float(pred_probs[1])
            }
        }
        
        return results
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
