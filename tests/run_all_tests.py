#!/usr/bin/env python3
"""
Master test suite for the entire Cellex pipeline
Runs comprehensive tests across all system components
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import importlib.util

def import_test_module(test_file):
    """Dynamically import a test module."""
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_test_suite(test_file, test_function_name):
    """Run a specific test suite."""
    try:
        print(f"\n🚀 Running {test_file.stem.upper()}")
        print("=" * 80)
        
        # Import and run the test module
        test_module = import_test_module(test_file)
        test_function = getattr(test_module, test_function_name)
        
        start_time = time.time()
        result = test_function()
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\n⏱️ Test suite completed in {duration:.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running {test_file.stem}: {e}")
        return False

def check_python_environment():
    """Check if we're in the correct Python environment."""
    print("🐍 PYTHON ENVIRONMENT CHECK")
    print("=" * 80)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️ Not in virtual environment")
    
    # Check key dependencies
    key_packages = ['torch', 'torchvision', 'numpy', 'PIL', 'cv2']
    missing_packages = []
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ Python environment ready for testing")
    return True

def check_project_structure():
    """Check if project structure is correct."""
    print("\n📁 PROJECT STRUCTURE CHECK")
    print("=" * 80)
    
    required_paths = [
        "src/data/download_data.py",
        "src/data/data_loader.py", 
        "src/models/models.py",
        "src/training/train.py",
        "src/inference/predict.py",
        "config/config.py",
        "train.py",
        "predict_image.py"
    ]
    
    missing_files = []
    
    for path in required_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            missing_files.append(path)
            print(f"❌ {path}")
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        return False
    
    print("\n✅ Project structure complete")
    return True

def run_comprehensive_tests():
    """Run all test suites in the correct order."""
    print("🧪 CELLEX COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"🕐 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Environment checks first
    if not check_python_environment():
        print("\n❌ Environment checks failed. Please fix environment setup.")
        return False
    
    if not check_project_structure():
        print("\n❌ Project structure checks failed. Please ensure all files are present.")
        return False
    
    # Test suites to run
    test_suites = [
        (Path("tests/test_data_pipeline.py"), "run_all_data_tests"),
        (Path("tests/test_training_pipeline.py"), "run_all_training_tests"),
        (Path("tests/test_checkpoints.py"), "test_checkpoint_system"),
        (Path("tests/test_inference_pipeline.py"), "run_all_inference_tests")
    ]
    
    results = []
    
    for test_file, test_function in test_suites:
        if test_file.exists():
            result = run_test_suite(test_file, test_function)
            results.append((test_file.stem, result))
        else:
            print(f"⚠️ Test file {test_file} not found, skipping...")
            results.append((test_file.stem, None))
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    total = len([r for r in results if r[1] is not None])
    
    for test_name, result in results:
        if result is None:
            status = "⏭️ SKIPPED"
        elif result:
            status = "✅ PASSED"
            passed += 1
        else:
            status = "❌ FAILED"
        
        print(f"{test_name.replace('_', ' ').title():.<50} {status}")
    
    print("-" * 80)
    print(f"🎯 Overall Results: {passed}/{total} test suites passed")
    
    if passed == total and total > 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Your Cellex system is fully operational!")
        return True
    elif total == 0:
        print("\n⚠️ No tests were run. Please check test file locations.")
        return False
    else:
        print(f"\n⚠️ {total - passed} test suite(s) failed.")
        print("🔧 Please review the failed tests and fix any issues.")
        return False

def run_quick_tests():
    """Run a quick subset of tests for rapid validation."""
    print("⚡ CELLEX QUICK TEST SUITE")
    print("=" * 80)
    
    # Just run basic imports and help commands
    quick_tests = [
        ("Config System", "from config.config import get_config; get_config()"),
        ("Model Imports", "from src.models.models import create_model, create_loss_function"),
        ("Training Help", [sys.executable, "train.py", "--help"]),
        ("Checkpoint Help", [sys.executable, "train.py", "--list-checkpoints"])
    ]
    
    passed = 0
    
    for test_name, test_cmd in quick_tests:
        try:
            if isinstance(test_cmd, str):
                result = subprocess.run([sys.executable, "-c", test_cmd], 
                                      capture_output=True, text=True, timeout=10)
            else:
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ {test_name}")
                passed += 1
            else:
                print(f"❌ {test_name}")
        except Exception as e:
            print(f"❌ {test_name}: {e}")
    
    print(f"\n⚡ Quick tests: {passed}/{len(quick_tests)} passed")
    return passed == len(quick_tests)

def main():
    """Main test runner with options."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        return run_quick_tests()
    else:
        return run_comprehensive_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)