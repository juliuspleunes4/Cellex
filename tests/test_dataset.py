"""
Tests for dataset handling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
import shutil
from ml_model.dataset import generate_synthetic_dataset, XrayDataset, get_data_transforms


class TestDataset(unittest.TestCase):
    """Test cases for dataset utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_synthetic_dataset_generation(self):
        """Test synthetic dataset can be generated"""
        num_samples = 10
        image_paths, labels = generate_synthetic_dataset(
            self.temp_dir, 
            num_samples_per_class=num_samples
        )
        
        # Check correct number of samples
        self.assertEqual(len(image_paths), num_samples * 2)
        self.assertEqual(len(labels), num_samples * 2)
        
        # Check labels are correct
        self.assertEqual(labels.count(0), num_samples)
        self.assertEqual(labels.count(1), num_samples)
        
        # Check files exist
        for path in image_paths:
            self.assertTrue(os.path.exists(path))
    
    def test_dataset_class(self):
        """Test XrayDataset class"""
        # Generate sample data
        num_samples = 5
        image_paths, labels = generate_synthetic_dataset(
            self.temp_dir,
            num_samples_per_class=num_samples
        )
        
        # Create dataset
        transform = get_data_transforms(augment=False)
        dataset = XrayDataset(image_paths, labels, transform=transform)
        
        # Check length
        self.assertEqual(len(dataset), len(image_paths))
        
        # Check sample can be retrieved
        image, label = dataset[0]
        self.assertEqual(image.shape[0], 3)  # 3 channels
        self.assertIn(label, [0, 1])
    
    def test_data_transforms(self):
        """Test data transforms"""
        # Training transforms
        train_transform = get_data_transforms(augment=True)
        self.assertIsNotNone(train_transform)
        
        # Validation transforms
        val_transform = get_data_transforms(augment=False)
        self.assertIsNotNone(val_transform)


if __name__ == '__main__':
    unittest.main()
