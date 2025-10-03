# 🔧 Troubleshooting Guide

This guide helps you resolve common issues when working with the Cellex Cancer Detection Platform.

## 📋 Quick Reference

### Training Issues

```bash
# Dataset not found error
python verify_dataset.py          # Check if dataset is properly processed
python src/data/download_data.py  # Re-download and process if needed

# Out of memory errors  
python train.py --batch-size 16   # Reduce batch size for limited GPU memory
python train.py --batch-size 8    # Further reduce for very limited memory

# Slow training
python train.py --batch-size 64   # Increase batch size if you have sufficient GPU memory
# Training automatically uses GPU if available

# Check training progress
# Look for files in: results/training_YYYYMMDD_HHMMSS/
# Best model saved as: models/best_model_epoch_XX.pth
```

## 🚨 Common Development Issues

### Kaggle API Setup

```bash
# Error: "403 Forbidden" when downloading
# Solution: Verify kaggle.json credentials
kaggle datasets list  # Test API access

# Error: "Dataset not found"
# Solution: Accept dataset terms on Kaggle website first
```

### CUDA/GPU Issues

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Training automatically detects and uses available hardware
python train.py  # Uses GPU if available, falls back to CPU
```

### Memory Issues

```bash
# Reduce batch size for limited RAM
python train.py --batch-size 16

# Mixed precision training is automatically enabled on compatible GPUs
python train.py --epochs 50  # Automatically uses optimal settings
```

## 💾 Checkpoint & Resume Issues

### "No checkpoints found"
```bash
# First time training - no checkpoints exist yet
python train.py  # Start fresh training
```

### "Checkpoint not found: latest"  
```bash
# No previous training exists
python train.py --list-checkpoints  # Check what's available
python train.py                     # Start fresh training
```

### "Corrupted checkpoint"
```bash
# Checkpoint file is damaged
python train.py --list-checkpoints  # Find working checkpoint
python train.py --resume checkpoint_epoch_15.pth  # Use older checkpoint
```

### Training Interruption Recovery
```bash
# After system crash or unexpected shutdown:
python train.py --list-checkpoints   # Check what's available
python train.py --resume latest      # Continue from last save
```

## 🏃‍♂️ Performance Issues

### GPU Memory Errors
```bash
# Reduce batch size for limited GPU memory
python train.py --batch-size 16   # Or even smaller: --batch-size 8
```

### Training Too Slow
```bash
# Increase batch size if you have sufficient GPU memory  
python train.py --batch-size 64   # Or larger if your GPU supports it
```

### Dataset Validation Failures
```bash
# Re-download and process datasets
python src/data/download_data.py

# Validate dataset structure  
python train.py --validate-only
```

## 💡 Best Practices

### Checkpoint Management
- 💡 **Always use `--resume latest`** when continuing work
- 💡 **Save checkpoints frequently** (every 5 epochs by default)  
- 💡 **Keep multiple checkpoints** for recovery from corruption
- 💡 **Monitor disk space** - each checkpoint is ~100-500MB
- 💡 **Use Ctrl+C to safely stop** training (auto-saves before exit)

### Development Workflow
- ✅ Run `python verify_dataset.py` before training
- ✅ Use `python train.py --validate-only` to check configuration
- ✅ Monitor training progress in real-time with live updates
- ✅ Keep backups of working checkpoints
- ✅ Test with smaller batch sizes first on new hardware

## 🆘 Getting Help

If you encounter issues not covered here:

1. Check the [changelog](CHANGELOG.md) for recent fixes
2. Review test results with `python -m pytest tests/ -v`
3. Verify your environment setup
4. Check the logs in `results/training_YYYYMMDD_HHMMSS/`

---

**Need additional support?** Visit our [GitHub Issues](https://github.com/juliuspleunes4/cellex/issues) for community assistance.