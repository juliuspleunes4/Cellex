"""
Dataset handling for medical image classification
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class XrayDataset(Dataset):
    """
    Custom Dataset for loading X-ray images
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels (0 or 1)
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms(augment=True):
    """
    Get data transformations for training and validation
    
    Args:
        augment: Whether to apply data augmentation (for training)
        
    Returns:
        transform: torchvision transforms
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(train_paths, train_labels, val_paths, val_labels, 
                       batch_size=32, num_workers=4):
    """
    Create training and validation data loaders
    
    Args:
        train_paths: List of training image paths
        train_labels: List of training labels
        val_paths: List of validation image paths
        val_labels: List of validation labels
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader: DataLoader instances
    """
    train_transform = get_data_transforms(augment=True)
    val_transform = get_data_transforms(augment=False)
    
    train_dataset = XrayDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = XrayDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def generate_synthetic_dataset(output_dir, num_samples_per_class=100):
    """
    Generate synthetic dataset for demonstration purposes
    Creates random images to simulate X-ray data
    
    Args:
        output_dir: Directory to save synthetic images
        num_samples_per_class: Number of samples to generate per class
        
    Returns:
        image_paths, labels: Lists of generated image paths and labels
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cancerous'), exist_ok=True)
    
    image_paths = []
    labels = []
    
    # Generate normal images (label 0)
    for i in range(num_samples_per_class):
        # Create grayscale-like image with lighter tones
        img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = os.path.join(output_dir, 'normal', f'normal_{i:04d}.jpg')
        img.save(img_path)
        image_paths.append(img_path)
        labels.append(0)
    
    # Generate cancerous images (label 1)
    for i in range(num_samples_per_class):
        # Create images with darker patches to simulate abnormalities
        img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        # Add some darker regions
        img_array[80:140, 80:140, :] = np.random.randint(30, 80, (60, 60, 3))
        img = Image.fromarray(img_array)
        img_path = os.path.join(output_dir, 'cancerous', f'cancerous_{i:04d}.jpg')
        img.save(img_path)
        image_paths.append(img_path)
        labels.append(1)
    
    return image_paths, labels
