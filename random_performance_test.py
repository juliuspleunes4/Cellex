#!/usr/bin/env python3
"""
True Random Sampling Performance Test for Cellex
================================================
This script performs real-world scenario testing by randomly selecting
different subsets of images for each test run, providing more realistic
performance estimates with natural variation.
"""

import sys
import time
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import modules
from config.config import get_config
from src.models.models import create_model
from src.data.data_loader import CellexTransforms

class RandomSamplingTester:
    """Random sampling tester for true real-world performance assessment"""
    
    def __init__(self, model_path="checkpoints/best_checkpoint.pth"):
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load model
        self.model = self._load_model()
        
        # Use basic transforms for inference
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Collect all available images
        self.all_images = self._collect_all_images()
        
    def _load_model(self):
        """Load the trained model"""
        print("🧠 Loading model...")
        model = create_model(self.config)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"✅ Loaded model from epoch {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded model weights")
            
        model = model.to(self.device)
        model.eval()
        return model
    
    def _collect_all_images(self):
        """Collect all available test images with their labels"""
        print("📦 Collecting all available test images...")
        
        data_path = Path("data/processed/unified")
        all_images = []
        
        # Check both validation and test directories
        for split in ["val", "test"]:
            split_path = data_path / split
            if not split_path.exists():
                print(f"⚠️  Warning: {split_path} not found, skipping...")
                continue
                
            # Healthy images (label 0)
            healthy_path = split_path / "healthy"
            if healthy_path.exists():
                for img_path in healthy_path.glob("*.jpeg"):
                    all_images.append((str(img_path), 0, split))
                for img_path in healthy_path.glob("*.jpg"):
                    all_images.append((str(img_path), 0, split))
                for img_path in healthy_path.glob("*.png"):
                    all_images.append((str(img_path), 0, split))
            
            # Cancer images (label 1)
            cancer_path = split_path / "cancer"
            if cancer_path.exists():
                for img_path in cancer_path.glob("*.jpeg"):
                    all_images.append((str(img_path), 1, split))
                for img_path in cancer_path.glob("*.jpg"):
                    all_images.append((str(img_path), 1, split))
                for img_path in cancer_path.glob("*.png"):
                    all_images.append((str(img_path), 1, split))
        
        # Separate by class
        healthy_images = [(path, label, split) for path, label, split in all_images if label == 0]
        cancer_images = [(path, label, split) for path, label, split in all_images if label == 1]
        
        print(f"📊 Total images found:")
        print(f"   💚 Healthy: {len(healthy_images)}")
        print(f"   🔴 Cancer:  {len(cancer_images)}")
        print(f"   📈 Total:   {len(all_images)}")
        
        return {'healthy': healthy_images, 'cancer': cancer_images, 'all': all_images}
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms (using PyTorch transforms)
            image_tensor = self.transform(image)
            
            return image_tensor
            
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            return None
    
    def run_random_test(self, samples_per_class=500, test_name=None):
        """Run a single random sampling test"""
        
        if test_name is None:
            test_name = f"random_test_{datetime.now().strftime('%H%M%S')}"
        
        print(f"\n🎲 RANDOM SAMPLING TEST: {test_name}")
        print("=" * 60)
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Target samples per class: {samples_per_class}")
        
        # Randomly sample images from each class
        available_healthy = len(self.all_images['healthy'])
        available_cancer = len(self.all_images['cancer'])
        
        actual_healthy_samples = min(samples_per_class, available_healthy)
        actual_cancer_samples = min(samples_per_class, available_cancer)
        
        print(f"🔀 Randomly selecting from {available_healthy} healthy and {available_cancer} cancer images...")
        
        # Random sampling without replacement
        selected_healthy = random.sample(self.all_images['healthy'], actual_healthy_samples)
        selected_cancer = random.sample(self.all_images['cancer'], actual_cancer_samples)
        
        # Combine and shuffle
        test_samples = selected_healthy + selected_cancer
        random.shuffle(test_samples)  # Shuffle the order of testing
        
        print(f"✅ Selected: {len(selected_healthy)} healthy + {len(selected_cancer)} cancer = {len(test_samples)} total")
        
        # Test each sample
        results = {
            'predictions': [],
            'true_labels': [],
            'probabilities': [],
            'confidences': [],
            'image_paths': [],
            'correct': []
        }
        
        correct_count = 0
        healthy_correct = 0
        cancer_correct = 0
        healthy_tested = 0
        cancer_tested = 0
        
        print("🧪 Testing samples...")
        
        with torch.no_grad():
            for i, (image_path, true_label, split) in enumerate(test_samples):
                # Load and preprocess image
                image_tensor = self._load_and_preprocess_image(image_path)
                
                if image_tensor is None:
                    continue
                
                # Add batch dimension and move to device
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # Get prediction
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted = torch.argmax(output, dim=1).item()
                confidence = torch.max(probabilities).item()
                
                # Store results
                results['predictions'].append(predicted)
                results['true_labels'].append(true_label)
                results['probabilities'].append(probabilities.cpu().numpy()[0])
                results['confidences'].append(confidence)
                results['image_paths'].append(image_path)
                
                # Track correctness
                is_correct = predicted == true_label
                results['correct'].append(is_correct)
                correct_count += is_correct
                
                # Class-specific tracking
                if true_label == 0:  # Healthy
                    healthy_tested += 1
                    healthy_correct += is_correct
                else:  # Cancer
                    cancer_tested += 1
                    cancer_correct += is_correct
                
                # Progress update
                if (i + 1) % 200 == 0 or (i + 1) == len(test_samples):
                    accuracy_so_far = correct_count / (i + 1)
                    print(f"   Progress: {i+1}/{len(test_samples)} | Accuracy so far: {accuracy_so_far:.4f}")
        
        # Calculate final metrics
        total_tested = len(results['predictions'])
        overall_accuracy = correct_count / total_tested if total_tested > 0 else 0
        healthy_accuracy = healthy_correct / healthy_tested if healthy_tested > 0 else 0
        cancer_accuracy = cancer_correct / cancer_tested if cancer_tested > 0 else 0
        balanced_accuracy = (healthy_accuracy + cancer_accuracy) / 2
        
        # Calculate additional statistics
        confidences = np.array(results['confidences'])
        correct_mask = np.array(results['correct'])
        
        metrics = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'total_tested': total_tested,
            'healthy_tested': healthy_tested,
            'cancer_tested': cancer_tested,
            'overall_accuracy': overall_accuracy,
            'healthy_accuracy': healthy_accuracy,
            'cancer_accuracy': cancer_accuracy,
            'balanced_accuracy': balanced_accuracy,
            'correct_count': correct_count,
            'healthy_correct': healthy_correct,
            'cancer_correct': cancer_correct,
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'mean_correct_confidence': float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else 0,
            'mean_incorrect_confidence': float(np.mean(confidences[~correct_mask])) if np.any(~correct_mask) else 0,
        }
        
        # Print results
        print(f"\n🎯 TEST RESULTS: {test_name}")
        print("-" * 50)
        print(f"📊 Total tested: {total_tested}")
        print(f"💚 Healthy: {healthy_tested} tested, {healthy_correct} correct ({healthy_accuracy:.6f})")
        print(f"🔴 Cancer:  {cancer_tested} tested, {cancer_correct} correct ({cancer_accuracy:.6f})")
        print(f"🎯 Overall accuracy: {overall_accuracy:.6f} ({correct_count}/{total_tested})")
        print(f"⚖️  Balanced accuracy: {balanced_accuracy:.6f}")
        print(f"🎚️  Mean confidence: {metrics['mean_confidence']:.4f} ± {metrics['std_confidence']:.4f}")
        
        return metrics, results
    
    def run_multiple_tests(self, num_tests=5, samples_per_class=500):
        """Run multiple random tests to assess performance variation"""
        
        print("🚀 MULTIPLE RANDOM SAMPLING TESTS")
        print("=" * 70)
        print(f"🔢 Number of tests: {num_tests}")
        print(f"📊 Samples per class per test: {samples_per_class}")
        print(f"🎲 Each test uses different randomly selected images")
        
        all_metrics = []
        
        for test_num in range(1, num_tests + 1):
            test_name = f"RandomTest_{test_num}"
            metrics, _ = self.run_random_test(samples_per_class, test_name)
            all_metrics.append(metrics)
            
            print(f"✅ Test {test_num}/{num_tests} complete: {metrics['balanced_accuracy']:.6f} balanced accuracy")
        
        # Analyze variation across tests
        self._analyze_test_variation(all_metrics)
        
        return all_metrics
    
    def _analyze_test_variation(self, all_metrics):
        """Analyze variation across multiple test runs"""
        
        print(f"\n📈 PERFORMANCE VARIATION ANALYSIS")
        print("=" * 70)
        
        # Extract key metrics
        balanced_accuracies = [m['balanced_accuracy'] for m in all_metrics]
        overall_accuracies = [m['overall_accuracy'] for m in all_metrics]
        healthy_accuracies = [m['healthy_accuracy'] for m in all_metrics]
        cancer_accuracies = [m['cancer_accuracy'] for m in all_metrics]
        confidences = [m['mean_confidence'] for m in all_metrics]
        
        def analyze_metric(values, name):
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"\n{name}:")
            print(f"   📊 Individual results: {[f'{v:.6f}' for v in values]}")
            print(f"   📈 Mean: {mean_val:.6f}")
            print(f"   📏 Std:  {std_val:.6f}")
            print(f"   📉 Range: {min_val:.6f} - {max_val:.6f}")
            print(f"   📊 95% CI: {mean_val - 1.96*std_val:.6f} - {mean_val + 1.96*std_val:.6f}")
            
            return mean_val, std_val
        
        balanced_mean, balanced_std = analyze_metric(balanced_accuracies, "🎯 BALANCED ACCURACY")
        overall_mean, overall_std = analyze_metric(overall_accuracies, "📊 OVERALL ACCURACY")
        healthy_mean, healthy_std = analyze_metric(healthy_accuracies, "💚 HEALTHY ACCURACY")
        cancer_mean, cancer_std = analyze_metric(cancer_accuracies, "🔴 CANCER ACCURACY")
        conf_mean, conf_std = analyze_metric(confidences, "🎚️ MEAN CONFIDENCE")
        
        # Performance assessment
        print(f"\n🔍 PERFORMANCE ASSESSMENT")
        print("-" * 40)
        
        if balanced_std < 0.005:  # Less than 0.5% variation
            print("✅ EXCELLENT: Very consistent performance across different image samples")
        elif balanced_std < 0.01:  # Less than 1% variation
            print("✅ GOOD: Reasonably consistent performance with acceptable variation")
        elif balanced_std < 0.02:  # Less than 2% variation
            print("⚠️  MODERATE: Some performance variation - model may be sensitive to sample selection")
        else:
            print("❌ HIGH VARIATION: Significant performance changes across samples - investigate further")
        
        print(f"\n📋 SUMMARY STATISTICS")
        print("-" * 30)
        print(f"🎯 Balanced Accuracy: {balanced_mean:.4f} ± {balanced_std:.4f}")
        print(f"📊 Overall Accuracy:  {overall_mean:.4f} ± {overall_std:.4f}")
        print(f"🎚️ Mean Confidence:   {conf_mean:.4f} ± {conf_std:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/random_sampling_analysis_{timestamp}.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_tests': len(all_metrics),
            'summary_statistics': {
                'balanced_accuracy': {'mean': float(balanced_mean), 'std': float(balanced_std)},
                'overall_accuracy': {'mean': float(overall_mean), 'std': float(overall_std)},
                'healthy_accuracy': {'mean': float(healthy_mean), 'std': float(healthy_std)},
                'cancer_accuracy': {'mean': float(cancer_mean), 'std': float(cancer_std)},
                'mean_confidence': {'mean': float(conf_mean), 'std': float(conf_std)}
            },
            'individual_tests': all_metrics
        }
        
        Path("results").mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Detailed results saved to: {results_file}")

def main():
    """Main function to run random sampling tests"""
    
    print("🎲 CELLEX RANDOM SAMPLING PERFORMANCE TESTER")
    print("=" * 80)
    print("This script performs true real-world testing by randomly selecting")
    print("different image subsets for each test run.")
    print()
    
    # Initialize tester
    tester = RandomSamplingTester()
    
    # Run multiple random tests
    num_tests = 5
    samples_per_class = 500
    
    print(f"🚀 Starting {num_tests} random tests with {samples_per_class} samples per class...")
    
    all_metrics = tester.run_multiple_tests(num_tests=num_tests, samples_per_class=samples_per_class)
    
    print(f"\n🎉 RANDOM SAMPLING TESTING COMPLETE!")
    print("=" * 50)
    print("✅ All tests completed successfully")
    print(f"📊 {num_tests} different random sample sets tested")
    print(f"🎯 Performance variation analyzed and saved")

if __name__ == "__main__":
    main()