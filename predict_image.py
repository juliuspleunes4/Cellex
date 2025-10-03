#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cellex Cancer Detection - Prediction Test
=========================================

Simple script to test cancer detection on individual images.
Use this after training to test your model's cancer detection capabilities.
"""

import sys
from pathlib import Path
import argparse

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def predict_image(image_path: str, model_path: str = None):
    """Predict if an image shows cancer or healthy tissue."""
    try:
        from src.inference.predict import CellexInference
        from src.utils.logger import CellexLogger
        
        logger = CellexLogger()
        
        # Use default model path if not provided
        if model_path is None:
            model_path = "models/best_model.pth"
            
        if not Path(model_path).exists():
            logger.error(f"❌ Model not found: {model_path}")
            logger.info("💡 Please train a model first:")
            logger.info("   python start_training.py")
            return
        
        logger.section("🔍 CELLEX CANCER DETECTION - PREDICTION")
        logger.info(f"🖼️  Image: {image_path}")
        logger.info(f"🧠 Model: {model_path}")
        
        # Load model and predict
        predictor = CellexInference(model_path)
        result = predictor.predict_single(
            image_path, 
            use_tta=True,  # Use test-time augmentation for better accuracy
            return_attention=True
        )
        
        # Display results
        prediction = result['class_name']
        confidence = result['confidence'] * 100
        normal_prob = result['probabilities']['normal'] * 100  
        cancer_prob = result['probabilities']['cancer'] * 100
        
        logger.success(f"🎯 Prediction: {prediction}")
        logger.info(f"📊 Confidence: {confidence:.1f}%")
        logger.info(f"💚 Healthy probability: {normal_prob:.1f}%")
        logger.info(f"🔴 Cancer probability: {cancer_prob:.1f}%")
        
        # Interpretation
        if prediction == "Cancer":
            if confidence > 80:
                logger.warning("⚠️  HIGH CONFIDENCE: Potential cancerous tissue detected")
            else:
                logger.warning("⚠️  MODERATE CONFIDENCE: Possible cancerous tissue")
                
            logger.info("💡 Recommendation: Consult with medical professional")
        else:
            if confidence > 80:
                logger.success("✅ HIGH CONFIDENCE: No cancer detected")
            else:
                logger.info("✅ MODERATE CONFIDENCE: Likely healthy tissue")
                
        logger.info(f"⏱️  Processing time: {result['inference_time']:.3f}s")
        
        return result
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install dependencies: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Test cancer detection on medical images")
    parser.add_argument("image", help="Path to medical image (JPG, PNG)")
    parser.add_argument("--model", help="Path to trained model (default: models/best_model.pth)")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
        
    predict_image(args.image, args.model)

if __name__ == "__main__":
    main()