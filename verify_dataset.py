#!/usr/bin/env python3
"""
Quick verification script for Cellex cancer detection dataset setup.

This script checks that:
1. Datasets are downloaded correctly
2. Unified processing completed successfully  
3. Training data is ready

Run this after downloading datasets to verify everything is ready for training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.download_data import CellexDataDownloader
from utils.logger import CellexLogger

def main():
    """Verify the cancer detection dataset is ready for training."""
    logger = CellexLogger()
    
    logger.section("[SYMBOL] CELLEX DATASET VERIFICATION")
    
    try:
        # Initialize downloader
        downloader = CellexDataDownloader()
        
        # Check if raw datasets exist
        raw_path = downloader.raw_data_path
        if not raw_path.exists():
            logger.error("[ERROR] No raw data folder found - datasets not downloaded")
            logger.info("[SYMBOL] Run 'python src/data/download_data.py' first")
            return False
        
        # Check each expected dataset
        expected_datasets = [
            "chest-ct-scan-data",
            "lung-and-colon-cancer-histopathological-images", 
            "brain-tumor-mri-dataset",
            "skin-cancer-mnist-ham10000"
        ]
        
        missing_datasets = []
        for dataset in expected_datasets:
            if not (raw_path / dataset).exists():
                missing_datasets.append(dataset)
        
        if missing_datasets:
            logger.warning(f"[WARNING]  Missing datasets: {missing_datasets}")
            logger.info("[SYMBOL] Some datasets may not have downloaded completely")
        else:
            logger.success("[SUCCESS] All raw datasets present")
        
        # Check unified dataset
        unified_path = downloader.processed_data_path / "unified"
        if not unified_path.exists():
            logger.warning("[WARNING]  Unified dataset not found - processing needed")
            logger.info("[FIX] Creating unified dataset...")
            unified_path = downloader.create_unified_dataset()
        
        # Verify unified dataset content
        if downloader._is_unified_dataset_ready():
            logger.success("[SUCCESS] Unified cancer detection dataset ready!")
            
            # Show statistics
            stats = downloader.get_dataset_statistics()
            if stats:
                logger.info(f"[STATS] Dataset Statistics:")
                for category, splits in stats.items():
                    if isinstance(splits, dict):
                        total = sum(splits.values())
                        logger.info(f"   {category.title()}: {total:,} images")
                        for split, count in splits.items():
                            logger.info(f"     - {split}: {count:,}")
            
            logger.success("[LAUNCH] Ready for model training!")
            return True
        else:
            logger.error("[ERROR] Unified dataset processing failed")
            logger.info("[SYMBOL] Try running: python src/data/download_data.py")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)