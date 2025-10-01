"""
Tests for the Cellex CNN model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from ml_model.model import CellexCNN, create_model


class TestCellexModel(unittest.TestCase):
    """Test cases for CellexCNN model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_classes = 2
        self.batch_size = 4
        self.image_size = 224
        
    def test_model_creation(self):
        """Test model can be created"""
        model = create_model(num_classes=self.num_classes)
        self.assertIsInstance(model, CellexCNN)
        
    def test_model_forward_pass(self):
        """Test forward pass with dummy data"""
        model = create_model(num_classes=self.num_classes)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
        
    def test_model_output_range(self):
        """Test model output can be converted to probabilities"""
        model = create_model(num_classes=self.num_classes)
        model.eval()
        
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Check probabilities sum to 1
        prob_sum = probabilities.sum().item()
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
        
        # Check probabilities are in valid range
        self.assertTrue(torch.all(probabilities >= 0))
        self.assertTrue(torch.all(probabilities <= 1))


if __name__ == '__main__':
    unittest.main()
