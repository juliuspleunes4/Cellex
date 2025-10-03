#!/usr/bin/env python3
"""
CELLEX SIMPLE ADVANCED MODEL EVALUATION SCRIPT
==============================================
Comprehensive testing of the cancer detection model with detailed analysis
without external ML dependencies that may have compatibility issues.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import random
from PIL import Image
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import project modules
from config.config import get_config
from src.models.models import create_model
from src.data.data_loader import CellexDataLoader
import logging

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class SimpleAdvancedModelTester:
    """Simple advanced testing suite for the Cellex cancer detection model."""
    
    def __init__(self, model_path=None, config=None):
        self.config = config or get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create results directory
        self.results_dir = Path("results/advanced_testing") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧪 CELLEX ADVANCED MODEL TESTER")
        print("=" * 70)
        print(f"📱 Device: {self.device}")
        print(f"📂 Results will be saved to: {self.results_dir}")
        print()
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = self._load_data()
    
    def _load_model(self, model_path=None):
        """Load the trained model."""
        print("🧠 LOADING TRAINED MODEL")
        print("-" * 50)
        
        # Auto-detect best model if not specified
        if model_path is None:
            models_dir = project_root / "models"
            checkpoints_dir = project_root / "checkpoints"
            
            # Try to find the best model
            for epoch in range(10, 0, -1):  # Check epochs 10 down to 1
                potential_path = models_dir / f"best_model_epoch_{epoch}.pth"
                if potential_path.exists():
                    model_path = potential_path
                    break
            
            if model_path is None:
                model_path = checkpoints_dir / "best_checkpoint.pth"
        
        print(f"📂 Model path: {model_path}")
        
        # Create and load model
        model = create_model(self.config)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            accuracy = checkpoint.get('accuracy', 'unknown')
            print(f"✅ Loaded model from epoch {epoch}")
            if accuracy != 'unknown':
                print(f"📊 Training accuracy: {accuracy:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded model weights")
        
        model = model.to(self.device)
        model.eval()
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🏗️  Architecture: {self.config.model.backbone}")
        print(f"📊 Parameters: {total_params:,}")
        print(f"🎯 Classes: {self.config.model.num_classes} (0=Healthy, 1=Cancer)")
        print()
        
        return model
    
    def _load_data(self):
        """Load test data."""
        print("📊 LOADING TEST DATA")
        print("-" * 50)
        
        data_loader = CellexDataLoader()
        data_dir = Path(self.config.data.processed_data_dir) / "unified"
        train_dataset, val_dataset, test_dataset = data_loader.create_datasets(data_dir)
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=16  # Smaller batch for detailed analysis
        )
        
        print(f"📈 Validation samples: {len(val_dataset):,}")
        print(f"🧪 Test samples: {len(test_dataset):,}")
        print(f"📊 Total evaluation samples: {len(val_dataset) + len(test_dataset):,}")
        print()
        
        return train_loader, val_loader, test_loader
    
    def run_comprehensive_evaluation(self, num_samples_per_class=500):
        """Run comprehensive model evaluation on many samples."""
        print("🔍 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)
        
        # Test on validation set with detailed sampling
        print("📊 Testing on Validation Set...")
        val_results = self._evaluate_dataset_detailed(self.val_loader, "validation", num_samples_per_class)
        
        # Test on test set  
        print("📊 Testing on Test Set...")
        test_results = self._evaluate_dataset_detailed(self.test_loader, "test", num_samples_per_class)
        
        # Combine results
        combined_results = self._combine_results(val_results, test_results)
        
        # Detailed analysis
        self._detailed_class_analysis(combined_results)
        self._confidence_analysis(combined_results)
        self._error_analysis(combined_results)
        
        # Generate visualizations
        self._generate_visualizations(combined_results)
        
        # Save comprehensive report
        self._save_comprehensive_report(combined_results)
        
        return combined_results
    
    def _evaluate_dataset_detailed(self, data_loader, dataset_name, samples_per_class=500):
        """Evaluate model on many samples from each class."""
        results = {
            'predictions': [],
            'probabilities': [],
            'true_labels': [],
            'confidences': [],
            'correct_predictions': [],
            'healthy_samples_tested': 0,
            'cancer_samples_tested': 0
        }
        
        healthy_count = 0
        cancer_count = 0
        max_per_class = samples_per_class
        
        print(f"   Targeting {max_per_class:,} samples per class from {dataset_name}...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                if healthy_count >= max_per_class and cancer_count >= max_per_class:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
                
                # Process each sample in batch
                for i in range(len(images)):
                    label = labels[i].item()
                    
                    # Check if we need more samples of this class
                    if (label == 0 and healthy_count < max_per_class) or \
                       (label == 1 and cancer_count < max_per_class):
                        
                        # Store results
                        results['predictions'].append(predicted_classes[i].item())
                        results['probabilities'].append(probabilities[i].cpu().numpy())
                        results['true_labels'].append(label)
                        
                        # Calculate confidence
                        confidence = torch.max(probabilities[i]).item()
                        results['confidences'].append(confidence)
                        
                        # Check correctness
                        correct = predicted_classes[i].item() == label
                        results['correct_predictions'].append(correct)
                        
                        # Update counters
                        if label == 0:
                            healthy_count += 1
                        else:
                            cancer_count += 1
                
                # Progress update
                if batch_idx % 20 == 0:
                    print(f"     Progress: Healthy {healthy_count:,}/{max_per_class:,} | Cancer {cancer_count:,}/{max_per_class:,}")
        
        results['healthy_samples_tested'] = healthy_count
        results['cancer_samples_tested'] = cancer_count
        
        # Calculate metrics
        predictions = np.array(results['predictions'])
        true_labels = np.array(results['true_labels'])
        correct_predictions = np.array(results['correct_predictions'])
        
        accuracy = np.mean(correct_predictions)
        
        # Per-class accuracy
        healthy_mask = true_labels == 0
        cancer_mask = true_labels == 1
        
        healthy_accuracy = np.mean(correct_predictions[healthy_mask]) if np.sum(healthy_mask) > 0 else 0
        cancer_accuracy = np.mean(correct_predictions[cancer_mask]) if np.sum(cancer_mask) > 0 else 0
        balanced_accuracy = (healthy_accuracy + cancer_accuracy) / 2
        
        print(f"   ✅ {dataset_name.capitalize()} Results:")
        print(f"      Healthy samples tested: {healthy_count:,}")
        print(f"      Cancer samples tested: {cancer_count:,}")
        print(f"      Overall Accuracy: {accuracy:.4f} ({np.sum(correct_predictions)}/{len(correct_predictions)})")
        print(f"      Healthy Accuracy: {healthy_accuracy:.4f}")
        print(f"      Cancer Accuracy:  {cancer_accuracy:.4f}")
        print(f"      Balanced Accuracy: {balanced_accuracy:.4f}")
        print()
        
        results['dataset'] = dataset_name
        results['metrics'] = {
            'accuracy': accuracy,
            'healthy_accuracy': healthy_accuracy,
            'cancer_accuracy': cancer_accuracy,
            'balanced_accuracy': balanced_accuracy
        }
        
        return results
    
    def _combine_results(self, val_results, test_results):
        """Combine validation and test results."""
        combined = {}
        
        for key in ['predictions', 'probabilities', 'true_labels', 'confidences', 'correct_predictions']:
            combined[key] = np.concatenate([val_results[key], test_results[key]])
        
        # Combined metrics
        accuracy = np.mean(combined['correct_predictions'])
        healthy_mask = combined['true_labels'] == 0
        cancer_mask = combined['true_labels'] == 1
        
        healthy_accuracy = np.mean(combined['correct_predictions'][healthy_mask])
        cancer_accuracy = np.mean(combined['correct_predictions'][cancer_mask])
        balanced_accuracy = (healthy_accuracy + cancer_accuracy) / 2
        
        combined['metrics'] = {
            'accuracy': accuracy,
            'healthy_accuracy': healthy_accuracy,
            'cancer_accuracy': cancer_accuracy,
            'balanced_accuracy': balanced_accuracy,
            'total_samples': len(combined['predictions']),
            'healthy_samples': np.sum(healthy_mask),
            'cancer_samples': np.sum(cancer_mask)
        }
        
        # Add individual dataset results
        combined['val_results'] = val_results
        combined['test_results'] = test_results
        
        return combined
    
    def _detailed_class_analysis(self, results):
        """Perform detailed analysis per class."""
        print("🔬 DETAILED CLASS ANALYSIS")
        print("-" * 70)
        
        predictions = np.array(results['predictions'])
        true_labels = np.array(results['true_labels'])
        probabilities = np.array(results['probabilities'])
        confidences = np.array(results['confidences'])
        
        # Manual confusion matrix
        tp = np.sum((true_labels == 1) & (predictions == 1))  # True Cancer
        tn = np.sum((true_labels == 0) & (predictions == 0))  # True Healthy  
        fp = np.sum((true_labels == 0) & (predictions == 1))  # False Cancer
        fn = np.sum((true_labels == 1) & (predictions == 0))  # False Healthy
        
        print("🎯 Confusion Matrix:")
        print("           Predicted")
        print("         Healthy  Cancer")
        print(f"Healthy    {tn:4d}    {fp:4d}")
        print(f"Cancer     {fn:4d}    {tp:4d}")
        print()
        
        # Class-wise statistics
        for class_idx, class_name in enumerate(['Healthy', 'Cancer']):
            class_mask = true_labels == class_idx
            class_predictions = predictions[class_mask]
            class_confidences = confidences[class_mask]
            class_probabilities = probabilities[class_mask][:, class_idx]
            
            correct_mask = class_predictions == class_idx
            
            print(f"📊 {class_name} Class Analysis:")
            print(f"   Total samples: {np.sum(class_mask):,}")
            print(f"   Correct predictions: {np.sum(correct_mask):,} ({np.mean(correct_mask)*100:.2f}%)")
            print(f"   Average confidence: {np.mean(class_confidences):.4f}")
            print(f"   Confidence std: {np.std(class_confidences):.4f}")
            print(f"   Min confidence: {np.min(class_confidences):.4f}")
            print(f"   Max confidence: {np.max(class_confidences):.4f}")
            print(f"   Average class probability: {np.mean(class_probabilities):.4f}")
            print()
        
        # Calculate precision, recall, f1 manually
        precision_healthy = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_healthy = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_healthy = 2 * (precision_healthy * recall_healthy) / (precision_healthy + recall_healthy) if (precision_healthy + recall_healthy) > 0 else 0
        
        precision_cancer = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_cancer = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_cancer = 2 * (precision_cancer * recall_cancer) / (precision_cancer + recall_cancer) if (precision_cancer + recall_cancer) > 0 else 0
        
        print("📈 Performance Metrics:")
        print(f"   Healthy:")
        print(f"     Precision: {precision_healthy:.4f}")
        print(f"     Recall:    {recall_healthy:.4f}")
        print(f"     F1-Score:  {f1_healthy:.4f}")
        print(f"   Cancer:")
        print(f"     Precision: {precision_cancer:.4f}")
        print(f"     Recall:    {recall_cancer:.4f}")
        print(f"     F1-Score:  {f1_cancer:.4f}")
        print()
    
    def _confidence_analysis(self, results):
        """Analyze prediction confidence patterns."""
        print("🎯 CONFIDENCE ANALYSIS")
        print("-" * 70)
        
        confidences = np.array(results['confidences'])
        correct_predictions = np.array(results['correct_predictions'])
        
        # Overall confidence statistics
        print(f"📊 Overall Confidence Statistics:")
        print(f"   Mean confidence: {np.mean(confidences):.4f}")
        print(f"   Std confidence: {np.std(confidences):.4f}")
        print(f"   Min confidence: {np.min(confidences):.4f}")
        print(f"   Max confidence: {np.max(confidences):.4f}")
        print()
        
        # Confidence by correctness
        correct_confidences = confidences[correct_predictions]
        incorrect_confidences = confidences[~correct_predictions]
        
        print(f"📈 Confidence by Prediction Quality:")
        print(f"   Correct predictions - Mean: {np.mean(correct_confidences):.4f}, Std: {np.std(correct_confidences):.4f}")
        if len(incorrect_confidences) > 0:
            print(f"   Incorrect predictions - Mean: {np.mean(incorrect_confidences):.4f}, Std: {np.std(incorrect_confidences):.4f}")
        else:
            print(f"   Incorrect predictions - No errors found!")
        print()
        
        # Confidence thresholds analysis
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        print(f"🎚️  Confidence Threshold Analysis:")
        
        for threshold in thresholds:
            high_conf_mask = confidences >= threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = np.mean(correct_predictions[high_conf_mask])
                coverage = np.mean(high_conf_mask)
                print(f"   Threshold {threshold:.2f}: {np.sum(high_conf_mask):4d} samples ({coverage*100:5.1f}%) - Accuracy: {high_conf_accuracy:.4f}")
        print()
    
    def _error_analysis(self, results):
        """Analyze prediction errors in detail."""
        print("❌ ERROR ANALYSIS")
        print("-" * 70)
        
        predictions = np.array(results['predictions'])
        true_labels = np.array(results['true_labels'])
        probabilities = np.array(results['probabilities'])
        confidences = np.array(results['confidences'])
        
        # Find error cases
        error_mask = predictions != true_labels
        error_indices = np.where(error_mask)[0]
        
        print(f"📊 Error Summary:")
        print(f"   Total errors: {np.sum(error_mask):,} out of {len(predictions):,} ({np.mean(error_mask)*100:.2f}%)")
        
        # False positives and false negatives
        false_positives = np.sum((true_labels == 0) & (predictions == 1))  # Healthy predicted as Cancer
        false_negatives = np.sum((true_labels == 1) & (predictions == 0))  # Cancer predicted as Healthy
        
        print(f"   False Positives (Healthy → Cancer): {false_positives:,}")
        print(f"   False Negatives (Cancer → Healthy): {false_negatives:,}")
        print()
        
        # Analyze error confidence
        if len(error_indices) > 0:
            error_confidences = confidences[error_indices]
            print(f"📈 Error Confidence Analysis:")
            print(f"   Mean error confidence: {np.mean(error_confidences):.4f}")
            print(f"   Std error confidence: {np.std(error_confidences):.4f}")
            print()
            
            # Show worst errors (highest confidence errors)
            print(f"🚨 Worst Errors (High Confidence, Wrong Prediction):")
            sorted_error_indices = error_indices[np.argsort(-error_confidences)]
            
            for i, idx in enumerate(sorted_error_indices[:min(10, len(sorted_error_indices))]):
                true_label = "Healthy" if true_labels[idx] == 0 else "Cancer"
                pred_label = "Healthy" if predictions[idx] == 0 else "Cancer"
                confidence = confidences[idx]
                healthy_prob = probabilities[idx][0]
                cancer_prob = probabilities[idx][1]
                
                print(f"   {i+1:2d}. True: {true_label:7} | Pred: {pred_label:7} | "
                      f"Conf: {confidence:.3f} | Healthy: {healthy_prob:.3f} | Cancer: {cancer_prob:.3f}")
        else:
            print("🎉 No errors found! Perfect performance!")
        print()
    
    def _generate_visualizations(self, results):
        """Generate visualization plots."""
        print("📊 GENERATING VISUALIZATIONS")
        print("-" * 70)
        
        # Set up plotting
        plt.style.use('default')
        
        probabilities = np.array(results['probabilities'])
        true_labels = np.array(results['true_labels'])
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        correct_predictions = np.array(results['correct_predictions'])
        
        # Create a comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Confidence Distribution
        axes[0, 0].hist(confidences[correct_predictions], bins=30, alpha=0.7, label='Correct', color='green', density=True)
        if np.sum(~correct_predictions) > 0:
            axes[0, 0].hist(confidences[~correct_predictions], bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
        axes[0, 0].set_xlabel('Prediction Confidence')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Confidence Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Class Probability Distributions
        healthy_probs_for_healthy = probabilities[true_labels == 0, 0]
        healthy_probs_for_cancer = probabilities[true_labels == 1, 0]
        
        axes[0, 1].hist(healthy_probs_for_healthy, bins=30, alpha=0.7, label='True Healthy', color='lightblue', density=True)
        axes[0, 1].hist(healthy_probs_for_cancer, bins=30, alpha=0.7, label='True Cancer', color='lightcoral', density=True)
        axes[0, 1].set_xlabel('Healthy Class Probability')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Healthy Probability Distribution by True Class')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Accuracy by Confidence Threshold
        thresholds = np.linspace(0.5, 0.99, 50)
        accuracies = []
        coverages = []
        
        for threshold in thresholds:
            high_conf_mask = confidences >= threshold
            if np.sum(high_conf_mask) > 0:
                accuracy = np.mean(correct_predictions[high_conf_mask])
                coverage = np.mean(high_conf_mask)
                accuracies.append(accuracy)
                coverages.append(coverage)
            else:
                accuracies.append(np.nan)
                coverages.append(0)
        
        axes[0, 2].plot(thresholds, accuracies, 'b-', label='Accuracy', linewidth=2)
        axes[0, 2].plot(thresholds, coverages, 'r--', label='Coverage', linewidth=2)
        axes[0, 2].set_xlabel('Confidence Threshold')
        axes[0, 2].set_ylabel('Value')
        axes[0, 2].set_title('Accuracy vs Coverage by Confidence Threshold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix
        tp = np.sum((true_labels == 1) & (predictions == 1))
        tn = np.sum((true_labels == 0) & (predictions == 0))
        fp = np.sum((true_labels == 0) & (predictions == 1))
        fn = np.sum((true_labels == 1) & (predictions == 0))
        
        cm = np.array([[tn, fp], [fn, tp]])
        im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[1, 0].text(j, i, str(cm[i, j]), ha="center", va="center", 
                               color="black" if cm[i, j] < cm.max() / 2 else "white", fontsize=16)
        
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_xticklabels(['Healthy', 'Cancer'])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_yticklabels(['Healthy', 'Cancer'])
        
        # 5. Per-class accuracy
        healthy_acc = results['metrics']['healthy_accuracy']
        cancer_acc = results['metrics']['cancer_accuracy']
        
        classes = ['Healthy', 'Cancer']
        accuracies_bar = [healthy_acc, cancer_acc]
        colors = ['lightblue', 'lightcoral']
        
        bars = axes[1, 1].bar(classes, accuracies_bar, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Per-Class Accuracy')
        axes[1, 1].set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies_bar):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Sample predictions visualization
        sample_indices = np.random.choice(len(predictions), min(100, len(predictions)), replace=False)
        sample_true = true_labels[sample_indices]
        sample_pred = predictions[sample_indices]
        sample_conf = confidences[sample_indices]
        
        correct_samples = sample_true == sample_pred
        
        axes[1, 2].scatter(sample_true[correct_samples], sample_conf[correct_samples], 
                          c='green', alpha=0.6, label='Correct', s=50)
        axes[1, 2].scatter(sample_true[~correct_samples], sample_conf[~correct_samples], 
                          c='red', alpha=0.6, label='Incorrect', s=50)
        axes[1, 2].set_xlabel('True Label (0=Healthy, 1=Cancer)')
        axes[1, 2].set_ylabel('Prediction Confidence')
        axes[1, 2].set_title('Sample Predictions vs Confidence')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "comprehensive_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualizations saved to: {plot_path}")
        print()
    
    def _save_comprehensive_report(self, results):
        """Save comprehensive evaluation report."""
        print("💾 SAVING COMPREHENSIVE REPORT")
        print("-" * 70)
        
        # Prepare report data
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'architecture': self.config.model.backbone,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'classes': self.config.model.num_classes
            },
            'dataset_info': {
                'total_samples': int(results['metrics']['total_samples']),
                'healthy_samples': int(results['metrics']['healthy_samples']),
                'cancer_samples': int(results['metrics']['cancer_samples'])
            },
            'overall_metrics': {
                'accuracy': float(results['metrics']['accuracy']),
                'healthy_accuracy': float(results['metrics']['healthy_accuracy']),
                'cancer_accuracy': float(results['metrics']['cancer_accuracy']),
                'balanced_accuracy': float(results['metrics']['balanced_accuracy'])
            },
            'validation_samples': {
                'healthy': int(results['val_results']['healthy_samples_tested']),
                'cancer': int(results['val_results']['cancer_samples_tested'])
            },
            'test_samples': {
                'healthy': int(results['test_results']['healthy_samples_tested']),
                'cancer': int(results['test_results']['cancer_samples_tested'])
            }
        }
        
        # Add detailed statistics
        confidences = np.array(results['confidences'])
        correct_predictions = np.array(results['correct_predictions'])
        
        report['confidence_analysis'] = {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
        }
        
        if np.sum(correct_predictions) > 0:
            report['confidence_analysis']['mean_correct_confidence'] = float(np.mean(confidences[correct_predictions]))
        if np.sum(~correct_predictions) > 0:
            report['confidence_analysis']['mean_incorrect_confidence'] = float(np.mean(confidences[~correct_predictions]))
        
        # Error analysis
        predictions = np.array(results['predictions'])
        true_labels = np.array(results['true_labels'])
        
        false_positives = int(np.sum((true_labels == 0) & (predictions == 1)))
        false_negatives = int(np.sum((true_labels == 1) & (predictions == 0)))
        
        report['error_analysis'] = {
            'total_errors': int(np.sum(predictions != true_labels)),
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'error_rate': float(np.mean(predictions != true_labels))
        }
        
        # Save JSON report
        json_path = self.results_dir / "comprehensive_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable report
        txt_path = self.results_dir / "comprehensive_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("CELLEX COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Model: {report['model_info']['architecture']}\n")
            f.write(f"Parameters: {report['model_info']['parameters']:,}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total samples evaluated: {report['dataset_info']['total_samples']:,}\n")
            f.write(f"Healthy samples: {report['dataset_info']['healthy_samples']:,}\n")
            f.write(f"Cancer samples: {report['dataset_info']['cancer_samples']:,}\n\n")
            
            f.write("SAMPLES TESTED PER DATASET\n")
            f.write("-" * 30 + "\n")
            f.write(f"Validation - Healthy: {report['validation_samples']['healthy']:,}, Cancer: {report['validation_samples']['cancer']:,}\n")
            f.write(f"Test - Healthy: {report['test_samples']['healthy']:,}, Cancer: {report['test_samples']['cancer']:,}\n\n")
            
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall Accuracy: {report['overall_metrics']['accuracy']:.4f}\n")
            f.write(f"Healthy Accuracy: {report['overall_metrics']['healthy_accuracy']:.4f}\n")
            f.write(f"Cancer Accuracy: {report['overall_metrics']['cancer_accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {report['overall_metrics']['balanced_accuracy']:.4f}\n\n")
            
            f.write("CONFIDENCE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean Confidence: {report['confidence_analysis']['mean_confidence']:.4f}\n")
            f.write(f"Confidence Std: {report['confidence_analysis']['std_confidence']:.4f}\n")
            if 'mean_correct_confidence' in report['confidence_analysis']:
                f.write(f"Correct Predictions Confidence: {report['confidence_analysis']['mean_correct_confidence']:.4f}\n")
            if 'mean_incorrect_confidence' in report['confidence_analysis']:
                f.write(f"Incorrect Predictions Confidence: {report['confidence_analysis']['mean_incorrect_confidence']:.4f}\n")
            f.write("\n")
            
            f.write("ERROR ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Errors: {report['error_analysis']['total_errors']:,}\n")
            f.write(f"False Positives (Healthy → Cancer): {report['error_analysis']['false_positives']:,}\n")
            f.write(f"False Negatives (Cancer → Healthy): {report['error_analysis']['false_negatives']:,}\n")
            f.write(f"Error Rate: {report['error_analysis']['error_rate']:.4f}\n")
        
        print(f"   ✅ JSON report saved to: {json_path}")
        print(f"   ✅ Text report saved to: {txt_path}")
        print()
        
        return report

def main():
    """Main function to run advanced testing."""
    print("🚀 STARTING ADVANCED MODEL TESTING")
    print("=" * 70)
    print()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize tester
    tester = SimpleAdvancedModelTester()
    
    # Run comprehensive evaluation with many samples
    results = tester.run_comprehensive_evaluation(num_samples_per_class=1000)  # Test 1000 samples per class
    
    # Final summary
    print("🎯 FINAL SUMMARY")
    print("=" * 70)
    print(f"✅ Testing completed successfully!")
    print(f"📊 Total samples evaluated: {results['metrics']['total_samples']:,}")
    print(f"🎯 Overall balanced accuracy: {results['metrics']['balanced_accuracy']:.4f}")
    print(f"📈 Healthy samples: {results['metrics']['healthy_samples']:,} (Accuracy: {results['metrics']['healthy_accuracy']:.4f})")
    print(f"🔬 Cancer samples: {results['metrics']['cancer_samples']:,} (Accuracy: {results['metrics']['cancer_accuracy']:.4f})")
    print(f"📂 Results saved to: {tester.results_dir}")
    print()
    print("🎉 Advanced testing complete! Check the results directory for detailed analysis.")

if __name__ == "__main__":
    main()