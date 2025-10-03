#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CELLEX CANCER DETECTION SYSTEM - COMPLETE TRAINING PIPELINE
============================================================

Advanced training script with comprehensive features:
- Binary cancer classification (Healthy vs Cancer)
- Data validation and preprocessing checks
- Model training with professional metrics
- Automatic model saving and evaluation
- Detailed progress reporting and logging
- Error handling and recovery
- Performance monitoring
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
from typing import Optional, Dict, Any

def setup_paths():
    """Setup project paths for imports."""
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    
    # Add to Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_path))
    
    return project_root, src_path

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        'torch', 'torchvision', 'timm', 'albumentations', 
        'numpy', 'PIL', 'cv2', 'pandas'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    return missing_modules

def validate_dataset(data_dir: Path) -> Dict[str, Any]:
    """Validate the cancer detection dataset structure and content."""
    validation_results = {
        'valid': False,
        'train_healthy': 0,
        'train_cancer': 0,
        'val_healthy': 0,
        'val_cancer': 0,
        'test_healthy': 0,
        'test_cancer': 0,
        'total': 0,
        'errors': []
    }
    
    try:
        # Check main directory structure
        required_paths = [
            data_dir / "train" / "healthy",
            data_dir / "train" / "cancer",
            data_dir / "val" / "healthy", 
            data_dir / "val" / "cancer",
            data_dir / "test" / "healthy",
            data_dir / "test" / "cancer"
        ]
        
        for path in required_paths:
            if not path.exists():
                validation_results['errors'].append(f"Missing directory: {path}")
                return validation_results
        
        # Count images in each directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for split in ['train', 'val', 'test']:
            for class_name in ['healthy', 'cancer']:
                path = data_dir / split / class_name
                count = sum(1 for f in path.iterdir() 
                           if f.suffix.lower() in image_extensions)
                validation_results[f'{split}_{class_name}'] = count
        
        # Calculate total
        validation_results['total'] = sum(
            validation_results[key] for key in validation_results.keys() 
            if key.endswith(('_healthy', '_cancer'))
        )
        
        # Check minimum requirements
        if validation_results['train_healthy'] < 100:
            validation_results['errors'].append("Too few healthy training images (minimum 100)")
        if validation_results['train_cancer'] < 100:
            validation_results['errors'].append("Too few cancer training images (minimum 100)")
        
        validation_results['valid'] = len(validation_results['errors']) == 0
        
    except Exception as e:
        validation_results['errors'].append(f"Dataset validation error: {str(e)}")
    
    return validation_results

def print_dataset_info(validation_results: Dict[str, Any]):
    """Print detailed dataset information."""
    print("\n" + "="*60)
    print("📊 DATASET VALIDATION RESULTS")
    print("="*60)
    
    if validation_results['valid']:
        print("✅ Dataset validation: PASSED")
    else:
        print("❌ Dataset validation: FAILED")
        for error in validation_results['errors']:
            print(f"   • {error}")
        return
    
    print(f"\n📁 Dataset Statistics:")
    print(f"   Training Set:")
    print(f"     • Healthy images: {validation_results['train_healthy']:,}")
    print(f"     • Cancer images:  {validation_results['train_cancer']:,}")
    print(f"   Validation Set:")
    print(f"     • Healthy images: {validation_results['val_healthy']:,}")
    print(f"     • Cancer images:  {validation_results['val_cancer']:,}")
    print(f"   Test Set:")
    print(f"     • Healthy images: {validation_results['test_healthy']:,}")
    print(f"     • Cancer images:  {validation_results['test_cancer']:,}")
    print(f"\n📈 Total Images: {validation_results['total']:,}")
    
    # Calculate class balance
    total_healthy = (validation_results['train_healthy'] + 
                    validation_results['val_healthy'] + 
                    validation_results['test_healthy'])
    total_cancer = (validation_results['train_cancer'] + 
                   validation_results['val_cancer'] + 
                   validation_results['test_cancer'])
    
    if total_healthy + total_cancer > 0:
        healthy_ratio = total_healthy / (total_healthy + total_cancer) * 100
        cancer_ratio = total_cancer / (total_healthy + total_cancer) * 100
        print(f"📊 Class Distribution:")
        print(f"     • Healthy: {healthy_ratio:.1f}% ({total_healthy:,} images)")
        print(f"     • Cancer:  {cancer_ratio:.1f}% ({total_cancer:,} images)")

def setup_training_environment():
    """Setup and validate training environment."""
    print("🔧 TRAINING ENVIRONMENT SETUP")
    print("="*60)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    missing_modules = check_dependencies()
    if missing_modules:
        print(f"❌ Missing dependencies: {', '.join(missing_modules)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    print("✅ All dependencies available")
    
    # Setup paths
    project_root, src_path = setup_paths()
    print(f"📁 Project root: {project_root}")
    print(f"📁 Source path: {src_path}")
    
    return True

def create_results_directory():
    """Create results directory for training outputs."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path("results") / f"training_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def save_training_config(config, results_dir: Path):
    """Save training configuration for reproducibility."""
    config_dict = {
        'model': {
            'backbone': config.model.backbone,
            'num_classes': config.model.num_classes,
            'pretrained': config.model.pretrained,
            'dropout_rate': config.model.dropout_rate
        },
        'training': {
            'batch_size': config.training.batch_size,
            'num_epochs': config.training.num_epochs,
            'learning_rate': config.training.learning_rate,
            'optimizer': config.training.optimizer
        },
        'data': {
            'image_size': config.data.image_size,
            'augmentation_enabled': config.data.augmentation_enabled
        }
    }
    
    config_file = results_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"💾 Configuration saved: {config_file}")

def list_checkpoints():
    """List all available checkpoints."""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print("❌ No checkpoints directory found")
        return []
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoints_dir.glob("checkpoint_epoch_*.pth"))
    latest_checkpoint = checkpoints_dir / "latest_checkpoint.pth"
    
    if not checkpoint_files and not latest_checkpoint.exists():
        print("📂 No checkpoints found")
        return []
    
    print("\n📂 AVAILABLE CHECKPOINTS")
    print("="*50)
    
    checkpoints = []
    
    # List latest checkpoint first
    if latest_checkpoint.exists():
        try:
            import torch
            checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
            epoch = checkpoint_data.get('epoch', 'unknown')
            accuracy = checkpoint_data.get('best_val_accuracy', 0.0)
            print(f"🔄 latest_checkpoint.pth (Epoch {epoch}, Best Acc: {accuracy:.2f}%)")
            checkpoints.append(('latest', latest_checkpoint, epoch, accuracy))
        except:
            print(f"🔄 latest_checkpoint.pth (corrupted)")
    
    # List epoch checkpoints
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    for checkpoint_file in checkpoint_files:
        try:
            import torch
            checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
            epoch = checkpoint_data.get('epoch', 'unknown')
            accuracy = checkpoint_data.get('best_val_accuracy', 0.0)
            print(f"📁 {checkpoint_file.name} (Epoch {epoch}, Best Acc: {accuracy:.2f}%)")
            checkpoints.append((checkpoint_file.name, checkpoint_file, epoch, accuracy))
        except:
            print(f"❌ {checkpoint_file.name} (corrupted)")
    
    print(f"\n💡 Use --resume <checkpoint> to resume training")
    print(f"💡 Use --resume latest to resume from most recent checkpoint")
    
    return checkpoints

def resolve_checkpoint_path(resume_arg):
    """Resolve checkpoint path from user argument."""
    if not resume_arg:
        return None
    
    checkpoints_dir = Path("checkpoints")
    
    # Handle "latest" keyword
    if resume_arg.lower() == "latest":
        latest_path = checkpoints_dir / "latest_checkpoint.pth"
        if latest_path.exists():
            return str(latest_path)
        else:
            print("❌ No latest checkpoint found")
            return None
    
    # Handle direct path
    if os.path.exists(resume_arg):
        return resume_arg
    
    # Handle checkpoint filename
    checkpoint_path = checkpoints_dir / resume_arg
    if checkpoint_path.exists():
        return str(checkpoint_path)
    
    # Try to match partial names
    if not resume_arg.endswith('.pth'):
        resume_arg += '.pth'
    
    checkpoint_path = checkpoints_dir / resume_arg
    if checkpoint_path.exists():
        return str(checkpoint_path)
    
    print(f"❌ Checkpoint not found: {resume_arg}")
    return None

def main():
    """Main training function with comprehensive error handling."""
    parser = argparse.ArgumentParser(description='Cellex Cancer Detection Training')
    parser.add_argument('--data-dir', type=str, help='Override data directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--model', type=str, help='Model backbone (efficientnet_b0, resnet50, etc.)')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint (use "latest" for most recent)')
    parser.add_argument('--list-checkpoints', action='store_true', help='List available checkpoints')
    parser.add_argument('--validate-only', action='store_true', help='Only validate dataset')
    
    args = parser.parse_args()
    
    # Handle list checkpoints
    if args.list_checkpoints:
        list_checkpoints()
        return True
    
    print("🏥 CELLEX CANCER DETECTION SYSTEM")
    print("="*60)
    print("🎯 Mission: Train AI to detect cancer in medical images")
    print("🔬 Classification: Binary (Healthy vs Cancer)")
    print("🧠 Model: Deep Learning with Attention Mechanisms")
    print("="*60)
    
    start_time = time.time()
    
    # Setup environment
    if not setup_training_environment():
        return False
    
    try:
        # Import training components
        from src.training.train import CellexTrainer
        from config.config import get_config
        from src.utils.logger import CellexLogger
        
        # Initialize logger
        logger = CellexLogger()
        
        # Load configuration
        config = get_config()
        
        # Override config with command line arguments
        if args.epochs:
            config.training.num_epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
        if args.lr:
            config.training.learning_rate = args.lr
        if args.model:
            config.model.backbone = args.model
        
        # Determine data directory
        if args.data_dir:
            data_dir = Path(args.data_dir)
        else:
            data_dir = Path(config.data.processed_data_dir) / "unified"
        
        print(f"\n📁 Using dataset: {data_dir}")
        
        # Validate dataset
        if not data_dir.exists():
            print(f"❌ Dataset not found: {data_dir}")
            print("💡 Please run data download and processing:")
            print("   python src/data/download_data.py")
            print("   python verify_dataset.py")
            return False
        
        validation_results = validate_dataset(data_dir)
        print_dataset_info(validation_results)
        
        if not validation_results['valid']:
            print("\n❌ Dataset validation failed!")
            return False
        
        if args.validate_only:
            print("\n✅ Dataset validation completed successfully!")
            return True
        
        # Create results directory
        results_dir = create_results_directory()
        print(f"\n📁 Results directory: {results_dir}")
        
        # Save configuration
        save_training_config(config, results_dir)
        
        # Print training configuration
        print("\n🔧 TRAINING CONFIGURATION")
        print("="*40)
        print(f"Model: {config.model.backbone}")
        print(f"Classes: {config.model.num_classes} (0=Healthy, 1=Cancer)")
        print(f"Epochs: {config.training.num_epochs}")
        print(f"Batch Size: {config.training.batch_size}")
        print(f"Learning Rate: {config.training.learning_rate}")
        print(f"Image Size: {config.data.image_size}")
        print(f"Augmentation: {config.data.augmentation_enabled}")
        
        # Resolve resume checkpoint path
        resume_path = resolve_checkpoint_path(args.resume) if args.resume else None
        
        # Initialize trainer
        print("\n🚀 INITIALIZING TRAINER")
        print("="*40)
        trainer = CellexTrainer(config, resume_from=resume_path)
        
        # Start training
        print("\n🏋️ STARTING TRAINING")
        print("="*40)
        print("🎯 Goal: Learn to distinguish healthy tissue from cancer")
        print("📊 Metrics: Accuracy, Precision, Recall, F1-Score")
        print("💾 Auto-save: Best model will be saved automatically")
        print("⏱️  Time: Training time will vary based on dataset size")
        
        # Resume from checkpoint if specified
        if resume_path:
            print(f"📂 Resuming from checkpoint: {resume_path}")
            print("💡 Press Ctrl+C anytime to safely stop and save progress")
        else:
            print("💡 Training will save checkpoints every 5 epochs")
            print("💡 Press Ctrl+C anytime to safely stop and save progress")
        
        # Run training
        training_results = trainer.train(str(data_dir))
        
        # Calculate training time
        training_time = time.time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        # Save results
        results_file = results_dir / "training_results.json"
        training_results['training_time_seconds'] = training_time
        training_results['dataset_info'] = validation_results
        
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        # Print final results
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total Training Time: {hours}h {minutes}m {seconds}s")
        print(f"🏆 Best Accuracy: {training_results.get('best_accuracy', 0):.4f}")
        print(f"📁 Results saved to: {results_dir}")
        print(f"🧠 Best model saved automatically")
        print(f"💾 Checkpoints available for future training")
        print("\n🔬 Your cancer detection AI is ready!")
        print("💡 Test it with: python predict_image.py <medical_image.jpg>")
        print("💡 View checkpoints: python train.py --list-checkpoints")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Solution: Install required packages")
        print("   pip install -r requirements.txt")
        return False
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        print("💡 You can resume training later with --resume option")
        return False
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # Save error log
        error_log = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            results_dir = create_results_directory()
            error_file = results_dir / "error_log.json"
            with open(error_file, 'w') as f:
                json.dump(error_log, f, indent=2)
            print(f"🔍 Error details saved to: {error_file}")
        except:
            pass
        
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)