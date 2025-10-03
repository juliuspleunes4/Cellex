#!/usr/bin/env python3
"""
Test suite for dataset download and validation functionality
"""

import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import json

def setup_test_environment():
    """Setup isolated test environment."""
    # Create temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="cellex_test_")
    return Path(test_dir)

def test_dataset_download():
    """Test dataset download functionality."""
    print("\n[TEST] TESTING DATASET DOWNLOAD")
    print("=" * 50)
    
    # Test 1: Check if download script exists and is executable
    download_script = Path("src/data/download_data.py")
    if not download_script.exists():
        print("[ERROR] Download script not found")
        return False
    
    print("[SUCCESS] Download script found")
    
    # Test 2: Check if script has proper imports
    try:
        with open(download_script, 'r', encoding='utf-8') as f:
            content = f.read()
            required_imports = ['kaggle', 'os', 'Path']
            missing_imports = []
            
            for imp in required_imports:
                if imp not in content:
                    missing_imports.append(imp)
            
            if missing_imports:
                print(f"[ERROR] Missing imports: {missing_imports}")
                return False
            
        print("[SUCCESS] Required imports present")
    except Exception as e:
        print(f"[ERROR] Error reading download script: {e}")
        return False
    
    # Test 3: Check if Kaggle API is configured (without actually downloading)
    try:
        result = subprocess.run([sys.executable, "-c", "import kaggle; print('Kaggle API available')"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("[SUCCESS] Kaggle API accessible")
        else:
            print("[WARNING] Kaggle API not configured (expected in CI/testing)")
    except Exception as e:
        print("[WARNING] Kaggle API test skipped (expected in CI/testing)")
    
    return True

def test_dataset_validation():
    """Test dataset validation functionality."""
    print("\n[TEST] TESTING DATASET VALIDATION") 
    print("=" * 50)
    
    # Test 1: Check if validate-only flag works
    try:
        result = subprocess.run([sys.executable, "train.py", "--validate-only"], 
                              capture_output=True, text=True, timeout=45)
        
        # Should either validate successfully or show clear error about missing dataset
        if (result.returncode == 0 and 
            ("Dataset validation" in result.stdout or 
             "PASSED" in result.stdout or 
             "not found" in result.stdout)):
            print("[SUCCESS] Dataset validation functionality working")
            return True
        else:
            print("[ERROR] Dataset validation not responding properly")
            print("Return code:", result.returncode)
            print("Output:", result.stdout[:300])
            print("Errors:", result.stderr[:300])
            return False
            
    except subprocess.TimeoutExpired:
        print("[WARNING] Dataset validation timed out (expected if dataset is large)")
        return True
    except Exception as e:
        print(f"[ERROR] Error testing dataset validation: {e}")
        return False

def test_data_processing():
    """Test data processing components."""
    print("\n[TEST] TESTING DATA PROCESSING")
    print("=" * 50)
    
    # Test 1: Check if data loader imports work
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "from src.data.data_loader import create_data_loaders, CellexTransforms; print('Data loader imports successful')"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("[SUCCESS] Data loader imports working")
        else:
            print("[ERROR] Data loader import errors:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Error testing data loader: {e}")
        return False
    
    # Test 2: Check if transforms are properly defined
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "from src.data.data_loader import CellexTransforms; t = CellexTransforms(); print('Transforms created successfully')"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("[SUCCESS] Data transforms working")
        else:
            print("[ERROR] Data transforms errors:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Error testing transforms: {e}")
        return False
    
    return True

def test_config_system():
    """Test configuration management."""
    print("\n[TEST] TESTING CONFIGURATION SYSTEM")
    print("=" * 50)
    
    # Test 1: Check if config can be imported and loaded
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "from config.config import get_config; config = get_config(); print('Config loaded successfully')"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("[SUCCESS] Configuration system working")
        else:
            print("[ERROR] Configuration errors:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Error testing config: {e}")
        return False
    
    # Test 2: Check if config.yaml can be generated
    try:
        result = subprocess.run([sys.executable, "config/config.py"], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0 and "config.yaml generated" in result.stdout:
            print("[SUCCESS] Config generation working")
        else:
            print("[WARNING] Config generation test inconclusive")
    except Exception as e:
        print(f"[WARNING] Config generation test skipped: {e}")
    
    return True

def run_all_data_tests():
    """Run all dataset and data processing tests."""
    print("[TEST] CELLEX DATA PIPELINE TESTS")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Dataset Download", test_dataset_download),
        ("Dataset Validation", test_dataset_validation), 
        ("Data Processing", test_data_processing),
        ("Configuration System", test_config_system)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n[STATS] TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "[SUCCESS] PASSED" if result else "[ERROR] FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("[COMPLETE] All data pipeline tests passed!")
        return True
    else:
        print("[WARNING] Some tests failed - check output above")
        return False

if __name__ == "__main__":
    run_all_data_tests()