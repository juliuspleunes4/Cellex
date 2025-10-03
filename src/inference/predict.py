"""
CELLEX CANCER DETECTION SYSTEM - INFERENCE ENGINE
================================================
Professional inference system for cancer detection in X-ray images.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
import json
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.models.models import create_model
from src.data.data_loader import CellexTransforms
from config.config import get_config

try:
    # Temporarily disable GradCAM to fix scipy compatibility issues
    # from pytorch_grad_cam import GradCAM
    # from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = False
    print("Info: GradCAM temporarily disabled for compatibility")
except ImportError:
    GRADCAM_AVAILABLE = False


class CellexInference:
    """
    Professional inference system for Cellex cancer detection.
    
    Features:
    - Single image prediction
    - Batch processing
    - Test Time Augmentation (TTA)
    - Attention visualization
    - Confidence scoring
    - Performance metrics
    """
    
    def __init__(self, model_path: str, config=None):
        self.config = config or get_config()
        self.logger = get_logger("CellexInference")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"[SYMBOL][INFO]  Device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize transforms
        self.transforms = CellexTransforms(
            image_size=self.config.data.image_size
        )
        
        # Class names
        self.class_names = ["Normal", "Cancer"]
        
        # Performance tracking
        self.inference_times = []
        
        self.logger.success("Cellex inference engine initialized")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        self.logger.info(f"[FILE] Loading model from: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = create_model(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Log model info
        if 'accuracy' in checkpoint:
            self.logger.metric("Model Accuracy", checkpoint['accuracy'], "")
        
        if 'epoch' in checkpoint:
            self.logger.info(f"[INFO] Trained for {checkpoint['epoch']} epochs")
        
        self.logger.success("Model loaded successfully")
        return model
    
    def predict_single(self, 
                      image_path: str, 
                      use_tta: bool = False,
                      return_attention: bool = False) -> Dict:
        """
        Predict cancer probability for a single X-ray image.
        
        Args:
            image_path: Path to the X-ray image
            use_tta: Whether to use Test Time Augmentation
            return_attention: Whether to return attention visualization
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Load and preprocess image
        image = self._load_image(image_path)
        
        if use_tta:
            prediction, probabilities = self._predict_with_tta(image)
        else:
            prediction, probabilities = self._predict_single_image(image)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Prepare results
        results = {
            'image_path': str(image_path),
            'prediction': int(prediction),
            'class_name': self.class_names[prediction],
            'probabilities': {
                'normal': float(probabilities[0]),
                'cancer': float(probabilities[1])
            },
            'confidence': float(probabilities[prediction]),
            'inference_time': inference_time,
            'timestamp': time.time()
        }
        
        # Add attention visualization if requested
        if return_attention and hasattr(self.model, 'get_attention_map'):
            try:
                attention_map = self._get_attention_visualization(image)
                results['attention_map'] = attention_map
            except Exception as e:
                self.logger.warning(f"[WARNING]  Attention visualization failed: {str(e)}")
        
        # Log prediction
        confidence_percent = results['confidence'] * 100
        confidence_color = "[SYMBOL]" if confidence_percent > 80 else "[SYMBOL]" if confidence_percent > 60 else "[SYMBOL]"
        
        self.logger.info(
            f"{confidence_color} Prediction: {results['class_name']} "
            f"({confidence_percent:.1f}% confidence) "
            f"in {inference_time*1000:.1f}ms"
        )
        
        return results
    
    def predict_batch(self, 
                     image_paths: List[str],
                     use_tta: bool = False,
                     batch_size: int = 32) -> List[Dict]:
        """
        Predict cancer probability for a batch of images.
        
        Args:
            image_paths: List of paths to X-ray images
            use_tta: Whether to use Test Time Augmentation
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        self.logger.section("BATCH PREDICTION")
        self.logger.info(f"[PROCESSING] Processing {len(image_paths):,} images")
        
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            self.logger.progress(
                i // batch_size + 1,
                (len(image_paths) + batch_size - 1) // batch_size,
                f"Processing batch {i//batch_size + 1}"
            )
            
            # Process each image in the batch
            for image_path in batch_paths:
                try:
                    result = self.predict_single(image_path, use_tta=use_tta)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"[ERROR] Failed to process {image_path}: {str(e)}")
                    # Add placeholder result for failed images
                    results.append({
                        'image_path': str(image_path),
                        'prediction': -1,
                        'class_name': 'Error',
                        'probabilities': {'normal': 0.0, 'cancer': 0.0},
                        'confidence': 0.0,
                        'error': str(e)
                    })
        
        self.logger.success(f"Batch prediction completed: {len(results):,} results")
        return results
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and validate image."""
        try:
            # Load with PIL
            image = Image.open(image_path)
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy
            image = np.array(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error loading image {image_path}: {str(e)}")
            raise
    
    def _predict_single_image(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict for a single image without TTA."""
        # Apply transforms
        transform = self.transforms.get_test_transforms()
        augmented = transform(image=image)
        tensor_image = augmented['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        return prediction.item(), probabilities.cpu().numpy()[0]
    
    def _predict_with_tta(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict using Test Time Augmentation."""
        tta_transforms = self.transforms.get_tta_transforms()
        all_predictions = []
        
        # Apply each TTA transform
        for transform in tta_transforms:
            augmented = transform(image=image)
            tensor_image = augmented['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor_image)
                probabilities = F.softmax(outputs, dim=1)
                all_predictions.append(probabilities.cpu().numpy()[0])
        
        # Average predictions
        averaged_probs = np.mean(all_predictions, axis=0)
        prediction = np.argmax(averaged_probs)
        
        return prediction, averaged_probs
    
    def _get_attention_visualization(self, image: np.ndarray) -> Optional[str]:
        """Generate attention visualization using GradCAM."""
        if not GRADCAM_AVAILABLE:
            self.logger.info("[INFO] GradCAM not available - returning None")
            return None
        
        try:
            # GradCAM is temporarily disabled for compatibility
            self.logger.info("[INFO] GradCAM temporarily disabled for scipy compatibility")
            return None
            
            # # Prepare image for GradCAM (commented out for now)
            # transform = self.transforms.get_test_transforms()
            # augmented = transform(image=image)
            # input_tensor = augmented['image'].unsqueeze(0).to(self.device)
            # 
            # # Get target layer (last convolutional layer)
            # target_layers = [self.model.backbone.features[-1]]
            # 
            # # Create GradCAM
            # cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
            # 
            # # Generate heatmap
            # grayscale_cam = cam(input_tensor=input_tensor)
            # grayscale_cam = grayscale_cam[0, :]
            # 
            # # Overlay on original image
            # rgb_img = cv2.resize(image, self.config.data.image_size) / 255.0
            # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # 
            # # Save visualization (optional)
            # # Could return base64 encoded image or save to file
            # 
            # return "attention_map_generated"  # Placeholder
            
        except Exception as e:
            self.logger.error(f"[ERROR] GradCAM visualization failed: {str(e)}")
            return None
    
    def evaluate_performance(self, 
                           test_data: List[Dict],
                           save_results: bool = True) -> Dict:
        """
        Evaluate model performance on test dataset.
        
        Args:
            test_data: List of {'image_path': str, 'label': int}
            save_results: Whether to save detailed results
            
        Returns:
            Performance metrics dictionary
        """
        self.logger.section("MODEL PERFORMANCE EVALUATION")
        
        correct_predictions = 0
        total_predictions = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        detailed_results = []
        
        # Process each test sample
        for i, sample in enumerate(test_data):
            if i % 100 == 0:
                self.logger.progress(i, len(test_data), "Evaluating")
            
            try:
                # Get prediction
                result = self.predict_single(sample['image_path'])
                prediction = result['prediction']
                true_label = sample['label']
                
                # Update counts
                total_predictions += 1
                if prediction == true_label:
                    correct_predictions += 1
                
                # Update confusion matrix
                if prediction == 1 and true_label == 1:
                    true_positives += 1
                elif prediction == 1 and true_label == 0:
                    false_positives += 1
                elif prediction == 0 and true_label == 0:
                    true_negatives += 1
                elif prediction == 0 and true_label == 1:
                    false_negatives += 1
                
                # Store detailed result
                detailed_results.append({
                    **result,
                    'true_label': true_label,
                    'correct': prediction == true_label
                })
                
            except Exception as e:
                self.logger.error(f"[ERROR] Evaluation failed for {sample['image_path']}: {str(e)}")
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Average inference time
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'total_samples': total_predictions,
            'avg_inference_time': avg_inference_time
        }
        
        # Log results
        self._log_performance_results(performance)
        
        # Save detailed results if requested
        if save_results:
            results_path = Path("results") / f"evaluation_results_{int(time.time())}.json"
            results_path.parent.mkdir(exist_ok=True)
            
            full_results = {
                'performance': performance,
                'detailed_results': detailed_results
            }
            
            with open(results_path, 'w') as f:
                json.dump(full_results, f, indent=2)
            
            self.logger.success(f"Detailed results saved to {results_path}")
        
        return performance
    
    def _log_performance_results(self, performance: Dict):
        """Log performance evaluation results."""
        self.logger.subsection("PERFORMANCE METRICS")
        
        self.logger.metric("Accuracy", performance['accuracy'], "")
        self.logger.metric("Precision", performance['precision'], "")
        self.logger.metric("Recall (Sensitivity)", performance['recall'], "")
        self.logger.metric("Specificity", performance['specificity'], "")
        self.logger.metric("F1 Score", performance['f1_score'], "")
        self.logger.metric("Average Inference Time", performance['avg_inference_time'] * 1000, "ms")
        
        self.logger.subsection("CONFUSION MATRIX")
        self.logger.info(f"True Positives:  {performance['true_positives']:,}")
        self.logger.info(f"False Positives: {performance['false_positives']:,}")
        self.logger.info(f"True Negatives:  {performance['true_negatives']:,}")
        self.logger.info(f"False Negatives: {performance['false_negatives']:,}")


def main():
    """Main inference function for testing."""
    config = get_config()
    logger = get_logger("InferenceDemo")
    
    logger.welcome()
    
    # Check if model exists
    model_path = Path("models") / "best_model.pth"
    
    if not model_path.exists():
        logger.error("[ERROR] Trained model not found!")
        logger.info("Please train the model first:")
        logger.info("python src/training/train.py")
        return
    
    # Initialize inference engine
    inference = CellexInference(str(model_path), config)
    
    # Demo prediction (if test image exists)
    test_image_path = Path("data") / "test_image.jpg"
    
    if test_image_path.exists():
        logger.section("DEMO PREDICTION")
        
        # Single prediction
        result = inference.predict_single(
            str(test_image_path),
            use_tta=True,
            return_attention=True
        )
        
        logger.success("Demo prediction completed")
        logger.info(f"[RESULTS] Results: {json.dumps(result, indent=2)}")
    else:
        logger.info("[INFO] No test image found for demo")
    
    logger.success("Inference system ready!")


if __name__ == "__main__":
    main()