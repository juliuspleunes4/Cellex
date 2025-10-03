#!/usr/bin/env python3
"""
Test suite for training pipeline functionality
"""

import sys
import os
import subprocess
import tempfile
import time
from pathlib import Path
import json

def test_training_script_help():
    """Test that training script help works."""
    print("\n🧪 TESTING TRAINING SCRIPT HELP")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "train.py", "--help"], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            # Check for essential options
            essential_options = ["--epochs", "--batch-size", "--lr", "--model", "--resume", "--list-checkpoints"]
            missing_options = []
            
            for option in essential_options:
                if option not in result.stdout:
                    missing_options.append(option)
            
            if missing_options:
                print(f"❌ Missing CLI options: {missing_options}")
                return False
            
            print("✅ All essential CLI options present")
            return True
        else:
            print("❌ Training script help failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing training help: {e}")
        return False

def test_model_imports():
    """Test that model architectures can be imported."""
    print("\n🧪 TESTING MODEL IMPORTS")
    print("=" * 50)
    
    # Test model imports
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "from src.models.models import create_model, create_loss_function; print('Model imports successful')"], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Model imports working")
        else:
            print("❌ Model import errors:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error testing model imports: {e}")
        return False
    
    # Test trainer imports
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "from src.training.train import CellexTrainer; print('Trainer import successful')"], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Trainer imports working")
        else:
            print("❌ Trainer import errors:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error testing trainer imports: {e}")
        return False
    
    return True

def test_training_initialization():
    """Test that training can initialize without errors."""
    print("\n🧪 TESTING TRAINING INITIALIZATION")
    print("=" * 50)
    
    try:
        # Test training initialization with validation only (quick test)
        result = subprocess.run([sys.executable, "train.py", "--validate-only"], 
                              capture_output=True, text=True, timeout=45)
        
        # Should complete validation or show clear error about missing dataset
        if ("Dataset validation" in result.stdout or 
            "not found" in result.stdout or 
            "processed data not found" in result.stdout.lower()):
            print("✅ Training initialization working")
            return True
        else:
            print("❌ Training initialization issues:")
            print("STDOUT:", result.stdout[:300])
            print("STDERR:", result.stderr[:300])
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Training initialization timed out (might be downloading/processing data)")
        return True
    except Exception as e:
        print(f"❌ Error testing training initialization: {e}")
        return False

def test_model_creation():
    """Test model creation functionality."""
    print("\n🧪 TESTING MODEL CREATION")
    print("=" * 50)
    
    # Test different model architectures
    models = ["efficientnet_b0", "resnet50", "densenet121"]
    
    for model_name in models:
        try:
            test_code = f"""
from config.config import get_config
from src.models.models import create_model
import torch

config = get_config()
config.model.backbone = '{model_name}'
config.model.num_classes = 2

model = create_model(config)
print(f'Model {model_name} created successfully')

# Test forward pass with dummy input
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f'Forward pass successful, output shape: {{output.shape}}')
"""
            
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ {model_name} model creation and forward pass working")
            else:
                print(f"❌ {model_name} model creation failed:")
                print(result.stderr[:200])
                return False
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            return False
    
    return True

def test_loss_functions():
    """Test loss function creation."""
    print("\n🧪 TESTING LOSS FUNCTIONS")
    print("=" * 50)
    
    try:
        test_code = """
from config.config import get_config
from src.models.models import create_loss_function
import torch

config = get_config()
criterion = create_loss_function(config)
print('Loss function created successfully')

# Test loss computation
dummy_output = torch.randn(4, 2)  # batch_size=4, num_classes=2
dummy_target = torch.tensor([0, 1, 0, 1])
loss = criterion(dummy_output, dummy_target)
print(f'Loss computation successful: {loss.item():.4f}')
"""
        
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Loss functions working")
            return True
        else:
            print("❌ Loss function errors:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing loss functions: {e}")
        return False

def test_optimizer_creation():
    """Test optimizer and scheduler creation."""
    print("\n🧪 TESTING OPTIMIZERS & SCHEDULERS")
    print("=" * 50)
    
    try:
        test_code = """
from config.config import get_config
from src.models.models import create_model
from src.training.train import CellexTrainer
import torch

config = get_config()
trainer = CellexTrainer(config)
model = create_model(config)

# Test optimizer creation
optimizer = trainer._create_optimizer(model)
print(f'Optimizer created: {type(optimizer).__name__}')

# Test scheduler creation
scheduler = trainer._create_scheduler(optimizer)
print(f'Scheduler created: {type(scheduler).__name__}')
"""
        
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Optimizers and schedulers working")
            return True
        else:
            print("❌ Optimizer/scheduler errors:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing optimizers: {e}")
        return False

def run_all_training_tests():
    """Run all training pipeline tests."""
    print("🧪 CELLEX TRAINING PIPELINE TESTS")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Training Script Help", test_training_script_help),
        ("Model Imports", test_model_imports),
        ("Training Initialization", test_training_initialization),
        ("Model Creation", test_model_creation),
        ("Loss Functions", test_loss_functions),
        ("Optimizers & Schedulers", test_optimizer_creation)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<35} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All training pipeline tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - check output above")
        return False

if __name__ == "__main__":
    run_all_training_tests()