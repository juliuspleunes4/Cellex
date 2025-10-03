"""
Debug script to check model predictions and raw outputs
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.models.models import create_model
from src.data.data_loader import CellexTransforms
from config.config import get_config

def debug_model_prediction(model_path, image_path):
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from: {model_path}")
    print(f"Testing image: {image_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Apply transforms
    transforms = CellexTransforms(image_size=config.data.image_size)
    transform = transforms.get_test_transforms()
    augmented = transform(image=image)
    tensor_image = augmented['image'].unsqueeze(0).to(device)
    
    print(f"Input tensor shape: {tensor_image.shape}")
    
    # Get raw outputs
    with torch.no_grad():
        raw_outputs = model(tensor_image)
        probabilities = F.softmax(raw_outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    print(f"Raw outputs: {raw_outputs.cpu().numpy()[0]}")
    print(f"Softmax probabilities: {probabilities.cpu().numpy()[0]}")
    print(f"Argmax prediction: {prediction.item()}")
    print(f"Class names: ['Normal', 'Cancer']")
    print(f"Predicted class: {'Normal' if prediction.item() == 0 else 'Cancer'}")

if __name__ == "__main__":
    model_path = "models/best_model_epoch_1.pth"  # New balanced model
    
    print("🧪 TESTING NEW CLASS-BALANCED MODEL")
    print("=" * 60)
    print("Previous model: Always predicted Cancer (overfitted)")
    print("New model: Trained with class balancing and regularization")
    print("=" * 60)
    
    # Test multiple images to confirm pattern
    test_images = [
        ("data/processed/unified/test/cancer/cancer_test_00001.jpeg", "Cancer"),
        ("data/processed/unified/test/cancer/cancer_test_00005.jpeg", "Cancer"),
        ("data/processed/unified/test/healthy/healthy_test_00001.jpeg", "Healthy"),
        ("data/processed/unified/test/healthy/healthy_test_00005.jpeg", "Healthy"),
        ("data/processed/unified/test/healthy/healthy_test_00010.jpeg", "Healthy"),
    ]
    
    for image_path, true_label in test_images:
        print("="*60)
        print(f"TESTING {true_label.upper()} IMAGE: {Path(image_path).name}")
        print("="*60)
        debug_model_prediction(model_path, image_path)
        print()