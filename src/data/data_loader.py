"""
CELLEX CANCER DETECTION SYSTEM - DATA PREPROCESSING
=================================================
Professional data loading and preprocessing pipeline.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from config.config import get_config


class CellexDataset(Dataset):
    """
    Professional dataset class for cancer detection in X-ray images.
    
    Features:
    - Robust image loading with error handling
    - Advanced augmentation pipeline
    - Memory-efficient loading
    - Support for multiple image formats
    - Professional logging
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[int],
                 transform: Optional[A.Compose] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 mode: str = "train"):
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        
        self.logger = get_logger(f"CellexDataset-{mode}")
        
        # Validate inputs
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
        
        self.logger.info(f"[STATS] Dataset initialized: {len(image_paths):,} samples")
        self.logger.info(f"[SYMBOL] Image size: {image_size[0]}x{image_size[1]}")
        
        # Calculate class distribution
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        for class_id, count in class_dist.items():
            percentage = (count / len(labels)) * 100
            class_name = "Normal" if class_id == 0 else "Cancer"
            self.logger.info(f"[SYMBOL] Class {class_id} ({class_name}): {count:,} samples ({percentage:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and preprocess a single sample."""
        try:
            # Load image
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # Load image with error handling
            image = self._load_image(image_path)
            
            # Apply transforms
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # Default normalization
                image = cv2.resize(image, self.image_size)
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)
            
            return image, label
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error loading image {idx}: {str(e)}")
            # Return a blank image and label 0 as fallback
            blank_image = torch.zeros(3, *self.image_size)
            return blank_image, 0
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image with robust error handling."""
        try:
            # Try loading with PIL first
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                if image.mode == 'L':  # Grayscale
                    image = image.convert('RGB')
                else:
                    image = image.convert('RGB')
            
            # Convert to numpy array
            image = np.array(image)
            
            # Validate image
            if image.size == 0:
                raise ValueError("Empty image")
                
            return image
            
        except Exception as e:
            self.logger.warning(f"[WARNING]  PIL failed for {image_path}, trying OpenCV: {str(e)}")
            
            # Fallback to OpenCV
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError("OpenCV couldn't load image")
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
                
            except Exception as e2:
                self.logger.error(f"[ERROR] Both PIL and OpenCV failed for {image_path}: {str(e2)}")
                # Return a blank image as last resort
                return np.ones((*self.image_size, 3), dtype=np.uint8) * 128


class CellexTransforms:
    """Professional image transformation pipeline for cancer detection."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.logger = get_logger("CellexTransforms")
    
    def get_train_transforms(self) -> A.Compose:
        """Get training augmentations - aggressive but medical-appropriate."""
        transforms = A.Compose([
            # Geometric transformations
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.3
            ),
            
            # Medical-appropriate augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            
            # Cutout for regularization
            A.CoarseDropout(
                max_holes=8,
                max_height=self.image_size[0] // 8,
                max_width=self.image_size[1] // 8,
                min_holes=1,
                fill_value=0,
                p=0.2
            ),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225],   # ImageNet stds
                max_pixel_value=255.0
            ),
            
            # Convert to tensor
            ToTensorV2()
        ])
        
        self.logger.info("[SYMBOL] Training transforms created with medical-appropriate augmentations")
        return transforms
    
    def get_val_transforms(self) -> A.Compose:
        """Get validation transforms - minimal processing."""
        transforms = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        self.logger.info("[TRANSFORMS] Validation transforms created")
        return transforms
    
    def get_test_transforms(self) -> A.Compose:
        """Get test transforms - same as validation."""
        return self.get_val_transforms()
    
    def get_tta_transforms(self) -> List[A.Compose]:
        """Get Test Time Augmentation transforms."""
        tta_transforms = []
        
        # Original
        tta_transforms.append(self.get_test_transforms())
        
        # Horizontal flip
        tta_transforms.append(A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
        
        # Slight rotations
        for angle in [-5, 5]:
            tta_transforms.append(A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Rotate(limit=angle, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
        
        # Brightness variations
        for brightness in [-0.1, 0.1]:
            tta_transforms.append(A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
        
        self.logger.info(f"[PROGRESS] Created {len(tta_transforms)} TTA transforms")
        return tta_transforms


class CellexDataLoader:
    """Professional data loading system for Cellex."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger("CellexDataLoader")
        
        # Initialize transforms
        self.transforms = CellexTransforms(
            image_size=self.config.data.image_size
        )
    
    def create_datasets(self, 
                       data_dir: Union[str, Path]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train, validation, and test datasets.
        
        Args:
            data_dir: Directory containing the organized dataset
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        self.logger.section("CREATING DATASETS")
        
        data_path = Path(data_dir)
        
        # Load datasets for each split
        datasets = {}
        
        for split in ['train', 'val', 'test']:
            split_path = data_path / split
            
            if not split_path.exists():
                self.logger.error(f"[ERROR] Split directory not found: {split_path}")
                continue
            
            # Collect images and labels
            image_paths = []
            labels = []
            
            # Healthy class (label 0)
            healthy_path = split_path / "healthy"
            if healthy_path.exists():
                healthy_images = list(healthy_path.glob("*.jpg")) + \
                              list(healthy_path.glob("*.png")) + \
                              list(healthy_path.glob("*.jpeg"))
                
                image_paths.extend([str(p) for p in healthy_images])
                labels.extend([0] * len(healthy_images))
            
            # Cancer class (label 1)
            cancer_path = split_path / "cancer"
            if cancer_path.exists():
                cancer_images = list(cancer_path.glob("*.jpg")) + \
                              list(cancer_path.glob("*.png")) + \
                              list(cancer_path.glob("*.jpeg"))
                
                image_paths.extend([str(p) for p in cancer_images])
                labels.extend([1] * len(cancer_images))
            
            # Get appropriate transforms
            if split == 'train':
                transform = self.transforms.get_train_transforms()
            else:
                transform = self.transforms.get_val_transforms()
            
            # Create dataset
            dataset = CellexDataset(
                image_paths=image_paths,
                labels=labels,
                transform=transform,
                image_size=self.config.data.image_size,
                mode=split
            )
            
            datasets[split] = dataset
            self.logger.success(f"[SUCCESS] {split.capitalize()} dataset created: {len(dataset):,} samples")
        
        return datasets['train'], datasets['val'], datasets['test']
    
    def create_data_loaders(self, 
                           train_dataset: Dataset,
                           val_dataset: Dataset,
                           test_dataset: Dataset,
                           batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training, validation, and testing."""
        
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        self.logger.subsection("CREATING DATA LOADERS")
        
        # Optimize number of workers for better GPU utilization  
        # Windows compatibility: use 0 workers to avoid multiprocessing issues
        import platform
        if platform.system() == 'Windows':
            optimal_workers = 0  # Single-threaded for Windows stability
            persistent_workers = False  # Cannot use persistent workers with num_workers=0
            prefetch_factor = None  # Cannot use prefetch_factor with num_workers=0
        else:
            optimal_workers = min(8, os.cpu_count() or 4)
            persistent_workers = True
            prefetch_factor = 2
        
        # Training loader with shuffle
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=optimal_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=persistent_workers,  # Keep workers alive between epochs
            prefetch_factor=prefetch_factor  # Prefetch more batches for better GPU utilization
        )
        
        # Validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=optimal_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        
        # Test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=optimal_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        
        self.logger.success(f"[SUCCESS] Data loaders created (batch_size={batch_size})")
        self.logger.info(f"[PROGRESS] Train batches: {len(train_loader):,}")
        self.logger.info(f"[PROGRESS] Val batches: {len(val_loader):,}")
        self.logger.info(f"[PROGRESS] Test batches: {len(test_loader):,}")
        
        return train_loader, val_loader, test_loader
    
    def get_class_weights(self, dataset: Dataset) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets."""
        labels = [dataset[i][1] for i in range(len(dataset))]
        unique, counts = np.unique(labels, return_counts=True)
        
        # Calculate inverse frequency weights
        total_samples = len(labels)
        weights = total_samples / (len(unique) * counts)
        
        self.logger.info("[BALANCE]  Class weights calculated:")
        for class_id, weight in zip(unique, weights):
            class_name = "Normal" if class_id == 0 else "Cancer"
            self.logger.info(f"   Class {class_id} ({class_name}): {weight:.4f}")
        
        return torch.tensor(weights, dtype=torch.float32)


def create_data_loaders(data_dir: Union[str, Path], config=None, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create all data loaders.
    
    Args:
        data_dir: Directory containing organized dataset
        config: Configuration object
        batch_size: Override batch size from config
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    loader = CellexDataLoader(config)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = loader.create_datasets(data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = loader.create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading pipeline
    from config.config import CellexConfig
    
    config = CellexConfig()
    logger = get_logger("DataTest")
    
    logger.welcome()
    logger.section("DATA PIPELINE TESTING")
    
    # Test transforms
    logger.subsection("Testing Transforms")
    transforms = CellexTransforms()
    
    train_transform = transforms.get_train_transforms()
    val_transform = transforms.get_val_transforms()
    
    logger.success("[SUCCESS] Transforms created successfully")
    
    # Test with dummy data
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    try:
        augmented = train_transform(image=dummy_image)
        logger.success(f"[SUCCESS] Training transform output shape: {augmented['image'].shape}")
        
        augmented = val_transform(image=dummy_image)
        logger.success(f"[SUCCESS] Validation transform output shape: {augmented['image'].shape}")
        
    except Exception as e:
        logger.error(f"[ERROR] Transform test failed: {str(e)}")
    
    logger.success("[COMPLETE] Data pipeline tests completed!")