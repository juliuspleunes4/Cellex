"""
CELLEX CANCER DETECTION SYSTEM - DATA DOWNLOADER
===============================================
Professional data pipeline for downloading and preparing cancer detection datasets.
"""

import os
import zipfile
import kaggle
import requests
from pathlib import Path
from typing import Dict, List
import sys
import json
import shutil

try:
    from colorama import Fore, Style
except ImportError:
    class MockColor:
        YELLOW = RESET_ALL = ""
    Fore = Style = MockColor()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from config.config import get_config


class CellexDataDownloader:
    """Professional data downloading and preparation system."""
    
    def __init__(self):
        self.logger = get_logger("DataDownloader")
        self.config = get_config()
        
        # Create data directories
        self.raw_data_path = Path(self.config.data.raw_data_dir)
        self.processed_data_path = Path(self.config.data.processed_data_dir)
        
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations - VERIFIED PUBLIC CANCER DETECTION DATASETS
        self.datasets = {
            "chest_cancer_ct": {
                "name": "Chest CT-Scan Cancer Detection",
                "kaggle_path": "mohamedhanyyy/chest-ctscan-images",
                "description": "CT scans for chest cancer detection with tumor annotations",
                "size_gb": 0.8,
                "samples": 1000,
                "classes": ["Cancer", "Normal"]
            },
            "lung_colon_cancer": {
                "name": "Lung and Colon Cancer Histopathological Images",
                "kaggle_path": "andrewmvd/lung-and-colon-cancer-histopathological-images",
                "description": "Histopathological images for lung cancer classification",
                "size_gb": 1.2,
                "samples": 25000,
                "classes": ["Lung benign tissue", "Lung adenocarcinoma", "Lung squamous cell carcinoma", "Colon adenocarcinoma", "Colon benign tissue"]
            },
            "brain_tumor_classification": {
                "name": "Brain Tumor Classification (MRI)",
                "kaggle_path": "sartajbhuvaji/brain-tumor-classification-mri",
                "description": "MRI images for brain tumor detection and classification",
                "size_gb": 0.2,
                "samples": 3264,
                "classes": ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"]
            },
            "skin_cancer_mnist": {
                "name": "Skin Cancer MNIST: HAM10000",
                "kaggle_path": "kmader/skin-cancer-mnist-ham10000",
                "description": "Large collection of multi-source dermatoscopic images for skin cancer classification",
                "size_gb": 2.5,
                "samples": 10015,
                "classes": ["Melanocytic nevi", "Melanoma", "Benign keratosis-like lesions", "Basal cell carcinoma", "Actinic keratoses", "Vascular lesions", "Dermatofibroma"]
            }
        }
    
    def check_kaggle_setup(self) -> bool:
        """Verify Kaggle API setup."""
        try:
            # Check if kaggle.json exists
            kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
            
            if not kaggle_path.exists():
                self.logger.error("❌ Kaggle API credentials not found!")
                self.logger.info("Please follow these steps:")
                self.logger.info("1. Go to https://www.kaggle.com/account")
                self.logger.info("2. Click 'Create New API Token'")
                self.logger.info("3. Save kaggle.json to ~/.kaggle/kaggle.json")
                self.logger.info("4. Run: chmod 600 ~/.kaggle/kaggle.json")
                return False
            
            # Test API access
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            self.logger.success("Kaggle API setup verified!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Kaggle setup error: {str(e)}")
            return False
    
    def display_dataset_info(self):
        """Display information about available datasets."""
        self.logger.section("AVAILABLE DATASETS FOR CANCER DETECTION")
        
        total_size = 0
        total_samples = 0
        
        for dataset_id, info in self.datasets.items():
            self.logger.info(f"\n📊 {info['name']}")
            self.logger.info(f"   Path: {info['kaggle_path']}")
            self.logger.info(f"   Size: {info['size_gb']:.1f} GB")
            self.logger.info(f"   Samples: {info['samples']:,}")
            self.logger.info(f"   Classes: {', '.join(info['classes'])}")
            self.logger.info(f"   Description: {info['description']}")
            
            total_size += info['size_gb']
            total_samples += info['samples']
        
        self.logger.subsection("SUMMARY")
        self.logger.metric("Total Size", total_size, "GB")
        self.logger.metric("Total Samples", total_samples, "images")
        
    def download_dataset(self, dataset_key: str, force_download: bool = False) -> bool:
        """Download a specific dataset."""
        if dataset_key not in self.datasets:
            self.logger.error(f"❌ Unknown dataset: {dataset_key}")
            return False
        
        dataset_info = self.datasets[dataset_key]
        kaggle_path = dataset_info['kaggle_path']
        dataset_name = dataset_info['name']
        
        # Check if already downloaded
        dataset_dir = self.raw_data_path / dataset_key
        if dataset_dir.exists() and not force_download:
            self.logger.success(f"✅ {dataset_name} already downloaded")
            return True
        
        try:
            self.logger.subsection(f"DOWNLOADING {dataset_name.upper()}")
            self.logger.info(f"📥 Downloading from: {kaggle_path}")
            self.logger.info(f"💾 Size: {dataset_info['size_gb']:.1f} GB")
            
            # Create dataset directory
            dataset_dir.mkdir(exist_ok=True)
            
            # Download using Kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            api.dataset_download_files(
                kaggle_path,
                path=str(dataset_dir),
                unzip=True
            )
            
            self.logger.success(f"✅ {dataset_name} downloaded successfully!")
            
            # Verify download
            if self._verify_download(dataset_dir, dataset_info):
                self.logger.success("✅ Download verification passed")
                return True
            else:
                self.logger.error("❌ Download verification failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Download failed for {dataset_name}: {str(e)}")
            return False
    
    def _verify_download(self, dataset_dir: Path, dataset_info: Dict) -> bool:
        """Verify that the dataset was downloaded correctly."""
        try:
            # Check if directory exists and has content
            if not dataset_dir.exists():
                return False
            
            # Count files
            file_count = len(list(dataset_dir.rglob("*.*")))
            
            # Basic verification - should have some files
            if file_count > 100:  # Reasonable threshold
                self.logger.info(f"📁 Found {file_count:,} files in dataset")
                return True
            else:
                self.logger.warning(f"⚠️  Only {file_count} files found - may be incomplete")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Verification error: {str(e)}")
            return False
    
    def download_all_datasets(self, force_download: bool = False):
        """Download all configured datasets."""
        self.logger.section("DOWNLOADING ALL CANCER DETECTION DATASETS")
        
        successful_downloads = 0
        total_datasets = len(self.datasets)
        
        for i, dataset_key in enumerate(self.datasets.keys(), 1):
            self.logger.step(i, total_datasets, f"Downloading {dataset_key}")
            
            if self.download_dataset(dataset_key, force_download):
                successful_downloads += 1
            
            self.logger.progress(i, total_datasets, "Overall Progress")
        
        # Summary
        self.logger.subsection("DOWNLOAD SUMMARY")
        self.logger.metric("Successful Downloads", successful_downloads, f"/{total_datasets}")
        
        if successful_downloads == total_datasets:
            self.logger.success("🎉 All datasets downloaded successfully!")
        else:
            failed = total_datasets - successful_downloads
            self.logger.warning(f"⚠️  {failed} dataset(s) failed to download")
        
        return successful_downloads == total_datasets
    
    def _is_unified_dataset_ready(self):
        """Check if unified dataset already exists and has content."""
        unified_path = self.processed_data_path / "unified"
        
        if not unified_path.exists():
            return False
            
        # Check if all required folders exist and have content
        required_paths = [
            unified_path / "train" / "healthy",
            unified_path / "train" / "cancer",
            unified_path / "val" / "healthy", 
            unified_path / "val" / "cancer",
            unified_path / "test" / "healthy",
            unified_path / "test" / "cancer"
        ]
        
        for path in required_paths:
            if not path.exists():
                return False
            # Check if folder has images (at least 10 files to be safe)
            image_count = len([f for f in path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            if image_count < 10:
                return False
                
        return True
    
    def create_unified_dataset(self):
        """Create a unified dataset structure for training."""
        self.logger.section("CREATING UNIFIED DATASET STRUCTURE")
        
        # Check if already processed
        unified_path = self.processed_data_path / "unified"
        if self._is_unified_dataset_ready():
            self.logger.info("✅ Unified cancer dataset already exists and ready!")
            return unified_path
        
        import random
        import shutil
        from collections import defaultdict
        
        # Create unified structure
        unified_path = self.processed_data_path / "unified"
        train_path = unified_path / "train"
        val_path = unified_path / "val"
        test_path = unified_path / "test"
        
        for split_path in [train_path, val_path, test_path]:
            (split_path / "healthy").mkdir(parents=True, exist_ok=True)
            (split_path / "cancer").mkdir(parents=True, exist_ok=True)
        
        self.logger.info("📁 Created directory structure")
        
        # Collect all images with labels
        all_images = {"healthy": [], "cancer": []}
        
        self.logger.info("🔍 Processing cancer datasets...")
        
        # Process Chest CT Cancer Detection
        chest_ct_path = self.raw_data_path / "chest_cancer_ct" / "Data"
        if chest_ct_path.exists():
            self.logger.info("📊 Processing Chest CT Cancer dataset...")
            for split in ["train", "test", "valid"]:
                split_path = chest_ct_path / split
                if split_path.exists():
                    # Normal images -> healthy
                    normal_dirs = [d for d in split_path.iterdir() if d.is_dir() and "normal" in d.name.lower()]
                    for normal_dir in normal_dirs:
                        for img in normal_dir.glob("*"):
                            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                all_images["healthy"].append(img)
                    
                    # Cancer images -> cancer  
                    cancer_dirs = [d for d in split_path.iterdir() if d.is_dir() and 
                                 any(cancer_type in d.name.lower() for cancer_type in 
                                     ['adenocarcinoma', 'carcinoma', 'cancer', 'tumor'])]
                    for cancer_dir in cancer_dirs:
                        for img in cancer_dir.glob("*"):
                            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                all_images["cancer"].append(img)
        
        # Process Lung/Colon Histopathological
        lung_colon_path = self.raw_data_path / "lung_colon_cancer" / "lung_colon_image_set"
        if lung_colon_path.exists():
            self.logger.info("📊 Processing Lung/Colon Histopathological dataset...")
            
            # Lung images
            lung_path = lung_colon_path / "lung_image_sets"
            if lung_path.exists():
                # Normal lung tissue -> healthy
                lung_normal = lung_path / "lung_n"
                if lung_normal.exists():
                    for img in lung_normal.glob("*"):
                        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            all_images["healthy"].append(img)
                
                # Cancer lung tissue -> cancer
                for cancer_type in ["lung_aca", "lung_scc"]:  # adenocarcinoma, squamous cell
                    cancer_dir = lung_path / cancer_type
                    if cancer_dir.exists():
                        for img in cancer_dir.glob("*"):
                            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                all_images["cancer"].append(img)
            
            # Colon images (optional - can include for broader cancer detection)
            colon_path = lung_colon_path / "colon_image_sets"
            if colon_path.exists():
                # Normal colon tissue -> healthy
                colon_normal = colon_path / "colon_n"
                if colon_normal.exists():
                    for img in colon_normal.glob("*"):
                        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            all_images["healthy"].append(img)
                
                # Cancer colon tissue -> cancer
                colon_cancer = colon_path / "colon_aca"
                if colon_cancer.exists():
                    for img in colon_cancer.glob("*"):
                        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            all_images["cancer"].append(img)
        
        # Process Brain Tumor Classification
        brain_path = self.raw_data_path / "brain_tumor_classification"
        if brain_path.exists():
            self.logger.info("📊 Processing Brain Tumor MRI dataset...")
            for split in ["Training", "Testing"]:
                split_path = brain_path / split
                if split_path.exists():
                    # No tumor -> healthy
                    no_tumor = split_path / "no_tumor"
                    if no_tumor.exists():
                        for img in no_tumor.glob("*"):
                            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                all_images["healthy"].append(img)
                    
                    # All tumor types -> cancer
                    tumor_dirs = [d for d in split_path.iterdir() if d.is_dir() and "tumor" in d.name and "no_" not in d.name]
                    for tumor_dir in tumor_dirs:
                        for img in tumor_dir.glob("*"):
                            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                all_images["cancer"].append(img)
        
        # Shuffle the data
        random.seed(self.config.seed)
        random.shuffle(all_images["healthy"])
        random.shuffle(all_images["cancer"])
        
        # Log collection results
        healthy_count = len(all_images["healthy"])
        cancer_count = len(all_images["cancer"])
        total_count = healthy_count + cancer_count
        
        self.logger.info(f"📊 Collected {healthy_count:,} healthy images")
        self.logger.info(f"📊 Collected {cancer_count:,} cancer images")
        self.logger.info(f"📊 Total: {total_count:,} images")
        
        # Split into train/val/test
        def split_data(images, train_ratio, val_ratio, test_ratio):
            n = len(images)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            return {
                "train": images[:train_end],
                "val": images[train_end:val_end],
                "test": images[val_end:]
            }
        
        healthy_splits = split_data(all_images["healthy"], 
                                  self.config.data.train_split,
                                  self.config.data.val_split, 
                                  self.config.data.test_split)
        
        cancer_splits = split_data(all_images["cancer"],
                                 self.config.data.train_split,
                                 self.config.data.val_split,
                                 self.config.data.test_split)
        
        # Copy files to unified structure
        self.logger.info("📁 Organizing images into train/val/test splits...")
        
        copy_count = 0
        for split in ["train", "val", "test"]:
            split_path = unified_path / split
            
            # Copy healthy images
            for i, src_img in enumerate(healthy_splits[split]):
                dst_img = split_path / "healthy" / f"healthy_{split}_{i:05d}{src_img.suffix}"
                shutil.copy2(src_img, dst_img)
                copy_count += 1
                
                if copy_count % 1000 == 0:
                    self.logger.info(f"📋 Copied {copy_count:,} images...")
            
            # Copy cancer images  
            for i, src_img in enumerate(cancer_splits[split]):
                dst_img = split_path / "cancer" / f"cancer_{split}_{i:05d}{src_img.suffix}"
                shutil.copy2(src_img, dst_img)
                copy_count += 1
                
                if copy_count % 1000 == 0:
                    self.logger.info(f"� Copied {copy_count:,} images...")
        
        self.logger.success(f"✅ Successfully organized {copy_count:,} images!")
        
        # Final statistics
        self.logger.subsection("DATASET ORGANIZATION")
        self.logger.info(f"📁 Unified dataset path: {unified_path}")
        
        for split in ["train", "val", "test"]:
            split_path = unified_path / split
            healthy_count = len(list((split_path / "healthy").glob("*")))
            cancer_count = len(list((split_path / "cancer").glob("*")))
            self.logger.info(f"📊 {split.capitalize():>5} - Healthy: {healthy_count:,}, Cancer: {cancer_count:,}")
        
        return unified_path
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        stats = {}
        
        for dataset_key in self.datasets.keys():
            dataset_dir = self.raw_data_path / dataset_key
            if dataset_dir.exists():
                file_count = len(list(dataset_dir.rglob("*.jpg")) + 
                              list(dataset_dir.rglob("*.png")) + 
                              list(dataset_dir.rglob("*.jpeg")))
                
                stats[dataset_key] = {
                    "files": file_count,
                    "size_mb": sum(f.stat().st_size for f in dataset_dir.rglob("*.*")) / (1024 * 1024)
                }
        
        return stats
    
    def cleanup_downloads(self):
        """Clean up temporary download files."""
        self.logger.info("🧹 Cleaning up temporary files...")
        
        # Remove zip files
        for zip_file in self.raw_data_path.rglob("*.zip"):
            zip_file.unlink()
            self.logger.info(f"🗑️  Removed: {zip_file.name}")
        
        self.logger.success("✅ Cleanup completed")


def main():
    """Main function for data downloading."""
    downloader = CellexDataDownloader()
    
    # Welcome message
    downloader.logger.welcome()
    
    # Check Kaggle setup
    if not downloader.check_kaggle_setup():
        return False
    
    # Display dataset information
    downloader.display_dataset_info()
    
    # Ask user for confirmation  
    print(f"\n{Fore.YELLOW}Do you want to download all datasets? (y/N): {Style.RESET_ALL}", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        # Download all datasets
        success = downloader.download_all_datasets()
        
        if success:
            # Create unified structure automatically
            try:
                unified_path = downloader.create_unified_dataset()
                downloader.logger.success(f"✅ Unified cancer detection dataset ready at: {unified_path}")
            except Exception as e:
                downloader.logger.error(f"❌ Failed to create unified dataset: {str(e)}")
                downloader.logger.info("💡 Raw datasets are available - you can process manually later")
            
            # Show statistics
            stats = downloader.get_dataset_statistics()
            downloader.logger.subsection("FINAL STATISTICS")
            
            for dataset, stat in stats.items():
                downloader.logger.metric(f"{dataset} files", stat['files'], "images")
                downloader.logger.metric(f"{dataset} size", stat['size_mb'], "MB")
            
            # Cleanup
            downloader.cleanup_downloads()
            
            downloader.logger.success("🎉 Data pipeline setup completed successfully!")
        else:
            downloader.logger.error("❌ Data download failed")
            return False
    else:
        downloader.logger.info("📋 Dataset download skipped")
    
    return True


if __name__ == "__main__":
    main()