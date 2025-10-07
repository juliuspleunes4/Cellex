#!/usr/bin/env python3
"""
Quick runner for advanced model testing
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import and run the test
from tests.test_advanced_model import SimpleAdvancedModelTester, set_seed

if __name__ == "__main__":
    set_seed(42)
    
    # Use the model from checkpoints folder
    model_path = "checkpoints/best_checkpoint.pth"
    
    print(f"🚀 Running advanced model test with: {model_path}")
    
    tester = SimpleAdvancedModelTester(model_path=model_path)
    results = tester.run_comprehensive_evaluation(num_samples_per_class=1000)
    
    print('\n🎯 FINAL SUMMARY')
    print('=' * 70)
    print(f'✅ Testing completed successfully!')
    print(f'📊 Total samples evaluated: {results["metrics"]["total_samples"]:,}')
    print(f'🎯 Overall balanced accuracy: {results["metrics"]["balanced_accuracy"]:.4f}')
    print(f'📈 Healthy samples: {results["metrics"]["healthy_samples"]:,} (Accuracy: {results["metrics"]["healthy_accuracy"]:.4f})')
    print(f'🔬 Cancer samples: {results["metrics"]["cancer_samples"]:,} (Accuracy: {results["metrics"]["cancer_accuracy"]:.4f})')
    print(f"📂 Results saved to: {tester.results_dir}")