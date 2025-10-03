# 👩‍💻 Developer Quick Setup Guide

## For New Developers Joining This Project

### 1. Initial Setup
```bash
# Clone and enter project
cd Cellex

# Install dependencies  
pip install -r requirements.txt

# Setup Kaggle API (if not done)
# Place kaggle.json in ~/.kaggle/ or C:\Users\{username}\.kaggle\
```

### 2. Download Cancer Detection Dataset
```bash
# This downloads AND processes automatically!
python src/data/download_data.py
```

**What this does:**
- Downloads 4 cancer datasets (~4GB total)
- **Automatically creates unified training structure**
- Splits into train/val/test (70/15/15)
- Creates binary healthy vs cancer classification

### 3. Verify Everything Worked
```bash
# Quick verification that dataset is ready
python verify_dataset.py
```

**Expected output:**
- ✅ All raw datasets present  
- ✅ Unified cancer detection dataset ready!
- 📊 Dataset statistics showing ~29K images
- 🚀 Ready for model training!

## 🛠️ Troubleshooting

### Problem: "Missing datasets" warning
**Solution:** Check Kaggle API setup, some datasets may not have downloaded completely.

### Problem: "Unified dataset not found"
**Solution:** The script auto-processes after download. If it failed, re-run:
```bash
python src/data/download_data.py
```

### Problem: Download fails with 403 errors  
**Solution:** Check your Kaggle account has accepted the dataset terms of use.

## 🎯 What You Get

After successful setup, you'll have:
- **29,264 medical images** ready for training
- **Binary classification**: healthy vs cancer
- **Multiple modalities**: CT, MRI, histology, dermatology  
- **Balanced splits** across train/val/test
- **No manual processing needed!**

The unified dataset creation is fully automated - you can go straight from download to training!

---
*Having issues? Check the main DATASET_README.md for more details.*