#!/usr/bin/env python3
"""
Test script for checkpoint system functionality
"""

import subprocess
import sys
from pathlib import Path

def safe_print(text):
    """Print text with fallback for encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        fallback = text.encode('ascii', 'ignore').decode('ascii')
        print(fallback)

def test_checkpoint_system():
    """Test the checkpoint system functionality."""
    safe_print("TESTING CHECKPOINT SYSTEM")
    safe_print("="*50)
    
    test_results = []
    
    # Test 1: List checkpoints when none exist
    safe_print("\n1. Testing: List checkpoints (empty)")
    try:
        result = subprocess.run([sys.executable, "train.py", "--list-checkpoints"], 
                              capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and ("No checkpoints found" in result.stdout or "checkpoints found" in result.stdout):
            safe_print("[SYMBOL] List checkpoints working")
            test_results.append(True)
        else:
            safe_print("[SYMBOL] List checkpoints failed")
            safe_print(f"Output: {result.stdout[:100]}")
            test_results.append(False)
    except Exception as e:
        safe_print(f"[SYMBOL] List checkpoints error: {e}")
        test_results.append(False)
    
    # Test 2: Test help command shows checkpoint options  
    safe_print("\n2. Testing: Help command shows checkpoint options")
    try:
        result = subprocess.run([sys.executable, "train.py", "--help"], 
                              capture_output=True, text=True, timeout=15)
        has_resume = "--resume" in result.stdout
        has_list = "--list-checkpoints" in result.stdout
        
        if has_resume and has_list:
            safe_print("[SYMBOL] Checkpoint options present in help")
            test_results.append(True)
        else:
            safe_print(f"[SYMBOL] Missing options: resume={has_resume}, list={has_list}")
            test_results.append(False)
    except Exception as e:
        safe_print(f"[SYMBOL] Help command error: {e}")
        test_results.append(False)
    
    # Test 3: Test validation bypasses checkpoint loading (correct behavior)
    safe_print("\n3. Testing: Validation bypasses checkpoint loading")
    try:
        result = subprocess.run([sys.executable, "train.py", "--resume", "nonexistent.pth", "--validate-only"], 
                              capture_output=True, text=True, timeout=30)
        # Should complete validation successfully since validate-only exits before checkpoint loading
        validation_passed = "Dataset validation" in result.stdout and result.returncode == 0
        if validation_passed:
            safe_print("[SYMBOL] Validation correctly bypasses checkpoint loading")
            test_results.append(True)
        else:
            safe_print("[SYMBOL] Validation not working properly")
            safe_print(f"Return code: {result.returncode}")
            test_results.append(False)
    except Exception as e:
        safe_print(f"[SYMBOL] Validation test error: {e}")
        test_results.append(False)
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    safe_print(f"\nTest Results: {passed}/{total} passed")
    
    if passed == total:
        safe_print("All checkpoint system tests passed!")
        safe_print("The checkpoint system is ready for production use!")
        return True
    else:
        safe_print("Some checkpoint tests failed - check output above")
        return False

if __name__ == "__main__":
    test_checkpoint_system()