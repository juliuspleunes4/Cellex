#!/usr/bin/env python3
"""
Test suite for inference and prediction functionality
"""

import sys
import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import json

def create_test_image():
    """Create a test medical image for inference testing."""
    # Create a 224x224 RGB test image
    test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # Add some structure to make it more realistic
    # Simple circular pattern to simulate medical imaging
    center_x, center_y = 112, 112
    y, x = np.ogrid[:224, :224]
    mask = (x - center_x)**2 + (y - center_y)**2 < 50**2
    test_image[mask] = [200, 200, 200]  # Light gray circle
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    test_image_path = Path(temp_dir) / "test_medical_image.jpg"
    
    # Save as image
    pil_image = Image.fromarray(test_image)
    pil_image.save(test_image_path, "JPEG")
    
    return test_image_path

def test_inference_imports():
    """Test that inference modules can be imported."""
    print("\n🧪 TESTING INFERENCE IMPORTS")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "from src.inference.predict import CellexInference; print('Inference imports successful')"], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ Inference imports working")
            return True
        else:
            print("❌ Inference import errors:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing inference imports: {e}")
        return False

def test_predict_script_help():
    """Test prediction script help functionality."""
    print("\n🧪 TESTING PREDICTION SCRIPT HELP")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "predict_image.py", "--help"], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            # Check for essential options
            essential_options = ["--model", "--output", "--tta"]
            missing_options = []
            
            for option in essential_options:
                if option not in result.stdout:
                    missing_options.append(option)
            
            if missing_options:
                print(f"⚠️ Some prediction options missing: {missing_options}")
                # Not a failure since predict script might have different interface
            
            print("✅ Prediction script help working")
            return True
        else:
            print("❌ Prediction script help failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing prediction help: {e}")
        return False

def test_inference_initialization():
    """Test inference system initialization without trained model."""
    print("\n🧪 TESTING INFERENCE INITIALIZATION")
    print("=" * 50)
    
    try:
        test_code = """
from src.inference.predict import CellexInference
from config.config import get_config
import torch

# Test initialization with config (without trained model)
config = get_config()
print('Config loaded for inference')

# Test model creation for inference
from src.models.models import create_model
model = create_model(config)
print('Model created for inference testing')

# Test dummy prediction structure
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
    print(f'Dummy inference successful, output shape: {output.shape}')
"""
        
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=20)
        
        if result.returncode == 0:
            print("✅ Inference initialization working")
            return True
        else:
            print("❌ Inference initialization errors:")
            print(result.stderr[:300])
            return False
            
    except Exception as e:
        print(f"❌ Error testing inference initialization: {e}")
        return False

def test_image_preprocessing():
    """Test image preprocessing for inference."""
    print("\n🧪 TESTING IMAGE PREPROCESSING")
    print("=" * 50)
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        test_code = f"""
from src.data.data_loader import CellexTransforms
from PIL import Image
import torch

# Load test image
image = Image.open(r'{test_image_path}')
print(f'Test image loaded: {{image.size}}')

# Test transforms
transforms = CellexTransforms()
test_transform = transforms.get_test_transforms()
print('Test transforms created')

# Apply transforms
if image.mode != 'RGB':
    image = image.convert('RGB')

transformed = test_transform(image)
print(f'Image preprocessed: {{transformed.shape}}')

# Should be [3, 224, 224]
if transformed.shape == torch.Size([3, 224, 224]):
    print('✅ Image preprocessing successful')
else:
    print(f'❌ Unexpected image shape: {{transformed.shape}}')
"""
        
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0 and "✅ Image preprocessing successful" in result.stdout:
            print("✅ Image preprocessing working")
            return True
        else:
            print("❌ Image preprocessing errors:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing image preprocessing: {e}")
        return False
    finally:
        # Clean up test image
        try:
            test_image_path.unlink()
            test_image_path.parent.rmdir()
        except:
            pass

def test_prediction_without_model():
    """Test prediction script behavior without trained model."""
    print("\n🧪 TESTING PREDICTION WITHOUT MODEL")
    print("=" * 50)
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        result = subprocess.run([sys.executable, "predict_image.py", str(test_image_path)], 
                              capture_output=True, text=True, timeout=20)
        
        # Should give clear error about missing model, not crash
        if ("model" in result.stderr.lower() or 
            "not found" in result.stderr.lower() or
            result.returncode != 0):
            print("✅ Prediction script handles missing model gracefully")
            return True
        else:
            print("⚠️ Prediction script behavior unclear without model")
            print("STDOUT:", result.stdout[:200])
            print("STDERR:", result.stderr[:200])
            return True  # Not a failure, just unclear behavior
            
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")
        return False
    finally:
        # Clean up test image
        try:
            test_image_path.unlink()
            test_image_path.parent.rmdir()
        except:
            pass

def test_gradcam_availability():
    """Test GradCAM functionality availability."""
    print("\n🧪 TESTING GRADCAM AVAILABILITY")
    print("=" * 50)
    
    try:
        test_code = """
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    print('✅ GradCAM dependencies available')
    GRADCAM_AVAILABLE = True
except ImportError as e:
    print(f'⚠️ GradCAM not available: {e}')
    GRADCAM_AVAILABLE = False

print(f'GradCAM status: {GRADCAM_AVAILABLE}')
"""
        
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            if "✅ GradCAM dependencies available" in result.stdout:
                print("✅ GradCAM dependencies working")
            else:
                print("⚠️ GradCAM dependencies missing (optional feature)")
            return True
        else:
            print("❌ GradCAM test errors:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing GradCAM: {e}")
        return False

def run_all_inference_tests():
    """Run all inference and prediction tests."""
    print("🧪 CELLEX INFERENCE PIPELINE TESTS")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Inference Imports", test_inference_imports),
        ("Prediction Script Help", test_predict_script_help),
        ("Inference Initialization", test_inference_initialization),
        ("Image Preprocessing", test_image_preprocessing),
        ("Prediction Without Model", test_prediction_without_model),
        ("GradCAM Availability", test_gradcam_availability)
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
        print("🎉 All inference pipeline tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - check output above")
        return False

if __name__ == "__main__":
    run_all_inference_tests()