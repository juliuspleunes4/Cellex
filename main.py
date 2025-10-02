"""
CELLEX CANCER DETECTION SYSTEM - MAIN EXECUTION SCRIPT
=====================================================
Professional entry point for the Cellex cancer detection system.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import get_logger
from config.config import get_config, CellexConfig


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Cellex Cancer Detection System - Advanced AI for Medical Imaging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and prepare data
  python main.py --mode download
  
  # Train the model
  python main.py --mode train
  
  # Run inference on a single image
  python main.py --mode predict --image path/to/xray.jpg
  
  # Evaluate model performance
  python main.py --mode evaluate --data-dir data/processed/unified
  
  # Run complete pipeline
  python main.py --mode pipeline
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['download', 'train', 'predict', 'evaluate', 'pipeline'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to X-ray image for prediction'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/unified',
        help='Directory containing processed dataset'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/best_model.pth',
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training/inference'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--use-tta',
        action='store_true',
        help='Use Test Time Augmentation for inference'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def download_data(logger):
    """Download and prepare cancer detection datasets."""
    try:
        from src.data.download_data import main as download_main
        logger.section("DATA DOWNLOAD AND PREPARATION")
        success = download_main()
        return success
    except Exception as e:
        logger.error(f"❌ Data download failed: {str(e)}")
        return False


def train_model(config, logger):
    """Train the cancer detection model."""
    try:
        from src.training.train import CellexTrainer
        logger.section("MODEL TRAINING")
        
        trainer = CellexTrainer(config)
        results = trainer.train(config.data.processed_data_dir + "/unified")
        
        logger.success("✅ Training completed successfully!")
        return results
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return None


def predict_image(image_path: str, model_path: str, config, use_tta: bool, logger):
    """Predict cancer probability for a single image."""
    try:
        from src.inference.predict import CellexInference
        
        logger.section("CANCER DETECTION INFERENCE")
        
        # Initialize inference engine
        inference = CellexInference(model_path, config)
        
        # Run prediction
        result = inference.predict_single(
            image_path,
            use_tta=use_tta,
            return_attention=True
        )
        
        # Display results
        logger.subsection("PREDICTION RESULTS")
        logger.info(f"📸 Image: {result['image_path']}")
        logger.info(f"🎯 Prediction: {result['class_name']}")
        logger.info(f"📊 Confidence: {result['confidence']:.2%}")
        logger.info(f"⏱️  Inference Time: {result['inference_time']*1000:.1f}ms")
        
        logger.subsection("PROBABILITY BREAKDOWN")
        logger.metric("Normal Probability", result['probabilities']['normal'], "")
        logger.metric("Cancer Probability", result['probabilities']['cancer'], "")
        
        if result['confidence'] > 0.8:
            logger.success("✅ High confidence prediction")
        elif result['confidence'] > 0.6:
            logger.warning("⚠️  Medium confidence prediction")
        else:
            logger.warning("🔴 Low confidence prediction - manual review recommended")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        return None


def evaluate_model(data_dir: str, model_path: str, config, logger):
    """Evaluate model performance on test dataset."""
    try:
        from src.inference.predict import CellexInference
        import json
        from pathlib import Path
        
        logger.section("MODEL PERFORMANCE EVALUATION")
        
        # Initialize inference engine
        inference = CellexInference(model_path, config)
        
        # Prepare test data
        test_dir = Path(data_dir) / "test"
        test_data = []
        
        # Collect test samples
        for class_id, class_name in enumerate(["normal", "cancer"]):
            class_dir = test_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        test_data.append({
                            'image_path': str(img_path),
                            'label': class_id
                        })
        
        if not test_data:
            logger.error("❌ No test data found!")
            return None
        
        logger.info(f"📊 Found {len(test_data):,} test samples")
        
        # Run evaluation
        performance = inference.evaluate_performance(test_data)
        
        logger.success("✅ Evaluation completed successfully!")
        return performance
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        return None


def run_complete_pipeline(config, logger):
    """Run the complete Cellex pipeline."""
    logger.banner("CELLEX COMPLETE PIPELINE EXECUTION")
    
    pipeline_start = time.time()
    
    # Step 1: Download data
    logger.step(1, 4, "Downloading and preparing datasets")
    if not download_data(logger):
        logger.error("❌ Pipeline failed at data download step")
        return False
    
    # Step 2: Train model
    logger.step(2, 4, "Training cancer detection model")
    results = train_model(config, logger)
    if not results:
        logger.error("❌ Pipeline failed at training step")
        return False
    
    # Step 3: Evaluate model
    logger.step(3, 4, "Evaluating model performance")
    performance = evaluate_model(
        config.data.processed_data_dir + "/unified",
        config.inference.model_path,
        config,
        logger
    )
    if not performance:
        logger.error("❌ Pipeline failed at evaluation step")
        return False
    
    # Step 4: Summary
    logger.step(4, 4, "Generating pipeline summary")
    
    pipeline_time = time.time() - pipeline_start
    
    logger.section("PIPELINE COMPLETION SUMMARY")
    logger.metric("Total Pipeline Time", pipeline_time, "seconds")
    logger.metric("Final Model Accuracy", performance['accuracy'], "")
    logger.metric("Model Precision", performance['precision'], "")
    logger.metric("Model Recall", performance['recall'], "")
    logger.metric("F1 Score", performance['f1_score'], "")
    
    logger.success("🎉 Complete Cellex pipeline executed successfully!")
    return True


def main():
    """Main execution function."""
    import time
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = CellexConfig.load_config(args.config)
    else:
        config = get_config()
    
    # Override config with command line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    # Initialize logger
    logger = get_logger("CellexMain")
    
    # Display welcome banner
    logger.welcome()
    
    # Execute based on mode
    try:
        if args.mode == 'download':
            success = download_data(logger)
            sys.exit(0 if success else 1)
            
        elif args.mode == 'train':
            results = train_model(config, logger)
            sys.exit(0 if results else 1)
            
        elif args.mode == 'predict':
            if not args.image:
                logger.error("❌ --image argument required for prediction mode")
                sys.exit(1)
            
            if not Path(args.image).exists():
                logger.error(f"❌ Image not found: {args.image}")
                sys.exit(1)
            
            result = predict_image(
                args.image,
                args.model_path,
                config,
                args.use_tta,
                logger
            )
            sys.exit(0 if result else 1)
            
        elif args.mode == 'evaluate':
            performance = evaluate_model(
                args.data_dir,
                args.model_path,
                config,
                logger
            )
            sys.exit(0 if performance else 1)
            
        elif args.mode == 'pipeline':
            success = run_complete_pipeline(config, logger)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.warning("⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()