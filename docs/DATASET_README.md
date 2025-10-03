# 🏥 Cellex Cancer Detection Dataset

This dataset provides a unified cancer detection system with **29,264 medical images** classified as healthy vs cancer across multiple modalities (chest CT, histopathological, brain MRI, skin).

## 🚀 Quick Start for Developers

### 1. Download & Setup (Automatic Processing)
```bash
python src/data/download_data.py
```
This will:
- Download 4 cancer datasets from Kaggle (~39,000 images)
- **Automatically create unified training structure**
- Split into train/val/test with binary healthy/cancer labels

### 2. Verify Everything is Ready
```bash
python verify_dataset.py
```
This checks that your dataset is processed and ready for training.

## 📊 Dataset Structure (After Processing)

```
data/processed/unified/
├── train/
│   ├── healthy/     # 10,715 images
│   └── cancer/      # 18,549 images  
├── val/
│   ├── healthy/     # 2,304 images
│   └── cancer/      # 4,085 images
└── test/
    ├── healthy/     # 2,307 images
    └── cancer/      # 4,084 images
```

## 🔧 For Other Developers

### No Manual Processing Needed! 
The unified dataset creation is **automatic** - just run the download script and it handles everything.

### If You Hit Issues:
1. **Empty unified folder?** - The script now auto-processes after downloads
2. **Missing datasets?** - Check your Kaggle API credentials
3. **Want to verify?** - Run `python verify_dataset.py`

### Source Datasets:
- **Chest CT Scan**: Lung cancer detection  
- **Histopathological**: Lung & colon cancer cells
- **Brain MRI**: Tumor detection
- **Skin Cancer**: HAM10000 dermatology dataset

## 🎯 Ready for Training

Once processed, you can immediately start training with the binary classification:
- **Target Classes**: `healthy` vs `cancer`
- **Balanced Splits**: Stratified train/val/test
- **Multiple Modalities**: CT, MRI, histology, dermatology

The dataset is designed for robust cancer detection across different imaging types!