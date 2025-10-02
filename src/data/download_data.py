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
        
        # Dataset configurations
        self.datasets = {
            "nih_chest_xrays": {
                "name": "NIH Chest X-Ray Dataset",
                "kaggle_path": "nih-chest-xrays/data",
                "description": "Large dataset of chest X-rays with disease classifications",
                "size_gb": 42.0,
                "samples": 112120,
                "classes": ["Normal", "Various diseases including cancer"]
            },
            "chest_xray_pneumonia": {
                "name": "Chest X-Ray Images (Pneumonia)",
                "kaggle_path": "paultimothymooney/chest-xray-pneumonia",
                "description": "Chest X-ray dataset for pneumonia detection",
                "size_gb": 1.2,
                "samples": 5863,
                "classes": ["Normal", "Pneumonia"]
            },
            "pulmonary_abnormalities": {
                "name": "Pulmonary Chest X-Ray Abnormalities",
                "kaggle_path": "kmader/pulmonary-chest-xray-abnormalities",
                "description": "Chest X-rays with various pulmonary abnormalities",
                "size_gb": 0.8,
                "samples": 3000,
                "classes": ["Normal", "Various abnormalities"]
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
    
    def create_unified_dataset(self):
        """Create a unified dataset structure for training."""
        self.logger.section("CREATING UNIFIED DATASET STRUCTURE")
        
        # Create unified structure
        unified_path = self.processed_data_path / "unified"
        train_path = unified_path / "train"
        val_path = unified_path / "val"
        test_path = unified_path / "test"
        
        for split_path in [train_path, val_path, test_path]:
            (split_path / "normal").mkdir(parents=True, exist_ok=True)
            (split_path / "cancer").mkdir(parents=True, exist_ok=True)
        
        self.logger.success("✅ Unified dataset structure created")
        
        # Dataset statistics
        self.logger.subsection("DATASET ORGANIZATION")
        self.logger.info(f"📁 Unified dataset path: {unified_path}")
        self.logger.info(f"📊 Train split: {self.config.data.train_split:.0%}")
        self.logger.info(f"📊 Validation split: {self.config.data.val_split:.0%}")
        self.logger.info(f"📊 Test split: {self.config.data.test_split:.0%}")
        
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
            # Create unified structure
            downloader.create_unified_dataset()
            
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