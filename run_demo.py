#!/usr/bin/env python
"""
Demo script for Cellex AI platform
This script demonstrates the complete workflow
"""

import sys
import os
import subprocess
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    try:
        import torch
        import flask
        import PIL
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False

def check_model():
    """Check if trained model exists"""
    model_path = 'ml_model/models/checkpoints/best_model.pth'
    if os.path.exists(model_path):
        print(f"✓ Trained model found at {model_path}")
        return True
    else:
        print(f"✗ Trained model not found at {model_path}")
        return False

def train_model():
    """Train the model"""
    print_header("Training Model")
    print("This will train a model on synthetic data (takes ~5 minutes)")
    print("Training parameters: 5 epochs, 50 samples per class\n")
    
    response = input("Do you want to train the model now? (y/n): ")
    if response.lower() == 'y':
        cmd = [
            sys.executable, 'ml_model/train.py',
            '--use-synthetic', '--epochs', '5',
            '--batch-size', '8', '--num-samples', '50'
        ]
        subprocess.run(cmd)
        return True
    else:
        print("Skipping model training.")
        return False

def run_tests():
    """Run unit tests"""
    print_header("Running Tests")
    print("Running unit tests...\n")
    
    # Run model tests
    print("Testing ML model...")
    result1 = subprocess.run(
        [sys.executable, '-m', 'unittest', 'tests.test_model'],
        capture_output=True, text=True
    )
    
    # Run dataset tests
    print("Testing dataset utilities...")
    result2 = subprocess.run(
        [sys.executable, '-m', 'unittest', 'tests.test_dataset'],
        capture_output=True, text=True
    )
    
    if result1.returncode == 0 and result2.returncode == 0:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False

def show_instructions():
    """Show instructions for running the application"""
    print_header("Running the Application")
    
    print("To run the complete Cellex platform:\n")
    
    print("1. Start the Backend API:")
    print("   python backend/app.py")
    print("   API will be available at: http://localhost:5000\n")
    
    print("2. Serve the Frontend:")
    print("   cd frontend")
    print("   python -m http.server 8000")
    print("   Frontend will be available at: http://localhost:8000\n")
    
    print("3. Open your browser and navigate to:")
    print("   http://localhost:8000\n")
    
    print("4. Upload an X-ray image and click 'Analyze Image'\n")
    
    print("API Endpoints:")
    print("  GET  http://localhost:5000/health")
    print("  GET  http://localhost:5000/api/info")
    print("  POST http://localhost:5000/api/predict\n")
    
    print("Example curl command:")
    print("  curl -X POST -F 'file=@image.jpg' http://localhost:5000/api/predict\n")

def main():
    """Main demo function"""
    print_header("Cellex AI Platform Demo")
    print("This script will help you set up and run the Cellex platform")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if model exists
    model_exists = check_model()
    
    # Train model if needed
    if not model_exists:
        print("\nNo trained model found. You need to train a model first.")
        if not train_model():
            print("\nYou can train the model later using:")
            print("  python ml_model/train.py --use-synthetic --epochs 5 --num-samples 50")
            return
    
    # Run tests
    print("\n")
    run_tests()
    
    # Show instructions
    show_instructions()
    
    print_header("Setup Complete")
    print("The Cellex platform is ready to use!")
    print("\nFor detailed documentation, see README.md")

if __name__ == '__main__':
    main()
