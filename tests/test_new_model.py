#!/usr/bin/env python3
"""
Test the new class-balanced model to verify overfitting fixes
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import project modules
from config.config import get_config
from src.models.models import create_model
from src.data.data_loader import CellexDataLoader
import logging

# Get configuration
config = get_config()

def test_new_model():
    """Test the newly trained class-balanced model"""
    print("🧪 TESTING NEW CLASS-BALANCED MODEL")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    # Load best model (epoch 2)
    model_path = project_root / "models" / "best_model_epoch_2.pth"
    if not model_path.exists():
        model_path = project_root / "checkpoints" / "best_checkpoint.pth"
    
    print(f"📂 Loading model: {model_path}")
    
    # Create model
    model = create_model(config)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        accuracy = checkpoint.get('accuracy', 'unknown')
        balanced_accuracy = checkpoint.get('balanced_accuracy', 'unknown')
        
        print(f"✅ Loaded model from epoch {epoch}")
        if accuracy != 'unknown':
            print(f"📊 Training accuracy: {accuracy:.4f}")
        else:
            print(f"📊 Training accuracy: {accuracy}")
        if balanced_accuracy != 'unknown':
            print(f"📊 Balanced accuracy: {balanced_accuracy:.4f}")
        else:
            print(f"📊 Balanced accuracy: {balanced_accuracy}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights")
    
    model = model.to(device)
    model.eval()
    
    print("\n🧠 MODEL ARCHITECTURE")
    print("-" * 40)
    print(f"Model: {config.model.backbone}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Classes: {config.model.num_classes} (0=Healthy, 1=Cancer)")
    
    # Load test data
    print(f"\n📊 LOADING TEST DATA")
    print("-" * 40)
    
    data_loader = CellexDataLoader()
    data_dir = Path(config.data.processed_data_dir) / "unified"
    train_dataset, val_dataset, test_dataset = data_loader.create_datasets(data_dir)
    train_loader, val_loader, test_loader = data_loader.create_data_loaders(
        train_dataset, val_dataset, test_dataset
    )
    
    print(f"Test batches: {len(test_loader)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test on a batch to see predictions
    print(f"\n🔍 TESTING PREDICTIONS")
    print("-" * 40)
    
    with torch.no_grad():
        # Get first test batch
        images, labels = next(iter(test_loader))
        images = images.to(device)
        labels = labels.to(device)
        
        # Get predictions
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(outputs, dim=1)
        
        # Show first 10 predictions
        print("Sample Predictions (First 10):")
        print("Idx | True | Pred | Healthy% | Cancer%  | Confidence")
        print("-" * 55)
        
        for i in range(min(10, len(labels))):
            true_label = labels[i].item()
            pred_label = predicted_classes[i].item()
            healthy_prob = probabilities[i][0].item()
            cancer_prob = probabilities[i][1].item()
            confidence = max(healthy_prob, cancer_prob)
            
            true_name = "Healthy" if true_label == 0 else "Cancer"
            pred_name = "Healthy" if pred_label == 0 else "Cancer"
            status = "✅" if true_label == pred_label else "❌"
            
            print(f"{i:2d}  | {true_name:7} | {pred_name:7} | {healthy_prob:.3f}   | {cancer_prob:.3f}   | {confidence:.3f} {status}")
    
    # Test full dataset accuracy
    print(f"\n📈 FULL TEST SET EVALUATION")
    print("-" * 40)
    
    correct = 0
    total = 0
    healthy_correct = 0
    healthy_total = 0
    cancer_correct = 0
    cancer_total = 0
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            
            # Store for analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                if labels[i] == 0:  # Healthy
                    healthy_total += 1
                    if predicted[i] == 0:
                        healthy_correct += 1
                else:  # Cancer
                    cancer_total += 1
                    if predicted[i] == 1:
                        cancer_correct += 1
    
    # Calculate metrics
    accuracy = correct / total
    healthy_accuracy = healthy_correct / healthy_total if healthy_total > 0 else 0
    cancer_accuracy = cancer_correct / cancer_total if cancer_total > 0 else 0
    balanced_accuracy = (healthy_accuracy + cancer_accuracy) / 2
    
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Healthy Accuracy: {healthy_accuracy:.4f} ({healthy_correct}/{healthy_total})")
    print(f"Cancer Accuracy:  {cancer_accuracy:.4f} ({cancer_correct}/{cancer_total})")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    
    # Analyze prediction distribution
    all_probabilities = np.array(all_probabilities)
    healthy_probs = all_probabilities[:, 0]
    cancer_probs = all_probabilities[:, 1]
    
    print(f"\n📊 PREDICTION ANALYSIS")
    print("-" * 40)
    print(f"Healthy predictions range: {healthy_probs.min():.3f} - {healthy_probs.max():.3f}")
    print(f"Cancer predictions range:  {cancer_probs.min():.3f} - {cancer_probs.max():.3f}")
    print(f"Average healthy confidence: {healthy_probs.mean():.3f}")
    print(f"Average cancer confidence:  {cancer_probs.mean():.3f}")
    
    # Check if still overfitting (always predicting one class)
    unique_predictions = len(set(all_predictions))
    print(f"Unique predictions: {unique_predictions}/2 classes")
    
    if unique_predictions == 1:
        print("⚠️  WARNING: Model still predicting only one class!")
        dominant_class = all_predictions[0]
        print(f"   Always predicting: {'Healthy' if dominant_class == 0 else 'Cancer'}")
    else:
        print("✅ SUCCESS: Model is predicting both classes!")
        healthy_count = sum(1 for p in all_predictions if p == 0)
        cancer_count = sum(1 for p in all_predictions if p == 1)
        print(f"   Healthy predictions: {healthy_count} ({healthy_count/len(all_predictions)*100:.1f}%)")
        print(f"   Cancer predictions:  {cancer_count} ({cancer_count/len(all_predictions)*100:.1f}%)")
    
    print(f"\n✨ COMPARISON WITH PREVIOUS OVERFITTED MODEL")
    print("-" * 40)
    print("Previous model issues:")
    print("❌ Always predicted 'Cancer' with 99.9% confidence")
    print("❌ 0% accuracy on healthy images")
    print("❌ Class imbalance not handled")
    print("❌ No regularization")
    print()
    print("New model improvements:")
    if unique_predictions > 1:
        print("✅ Predicts both healthy and cancer classes")
        print(f"✅ Balanced accuracy: {balanced_accuracy:.1%}")
        print("✅ Class balancing with weighted loss")
        print("✅ Enhanced regularization (dropout + weight decay)")
        
    print(f"\n🎯 SUMMARY")
    print("=" * 60)
    if balanced_accuracy > 0.8 and unique_predictions > 1:
        print("🎉 SUCCESS! Overfitting problem has been FIXED!")
        print(f"   - Model achieves {balanced_accuracy:.1%} balanced accuracy")
        print("   - Both classes are being predicted correctly")
        print("   - Class imbalance has been addressed")
    elif unique_predictions > 1:
        print("✅ Partial success - model predicts both classes but needs improvement")
    else:
        print("❌ Overfitting still present - model needs further adjustments")

if __name__ == "__main__":
    test_new_model()