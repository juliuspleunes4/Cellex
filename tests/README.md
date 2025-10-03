# Cellex Test Suite

Comprehensive testing framework for the Cellex Cancer Detection System.

## Quick Start

### Run All Tests
```bash
# Comprehensive test suite (recommended)
python tests/run_all_tests.py

# Quick validation tests
python tests/run_all_tests.py --quick
```

### Run Individual Test Suites
```bash
# Test data pipeline (dataset download, validation, processing)
python tests/test_data_pipeline.py

# Test training pipeline (models, training, optimizers)
python tests/test_training_pipeline.py

# Test checkpoint system (save, resume, list)
python tests/test_checkpoints.py

# Test inference pipeline (prediction, preprocessing, GradCAM)
python tests/test_inference_pipeline.py
```

## Test Coverage

### 🗄️ Data Pipeline Tests
- **Dataset Download**: Kaggle integration, script validation
- **Dataset Validation**: Structure checks, class distribution
- **Data Processing**: Transforms, data loaders, preprocessing
- **Configuration**: Config loading, YAML generation

### 🏋️ Training Pipeline Tests
- **Training Scripts**: CLI options, help system, initialization
- **Model Architecture**: EfficientNet, ResNet, DenseNet creation
- **Training Components**: Loss functions, optimizers, schedulers
- **Forward Pass**: Model inference with dummy data

### 💾 Checkpoint System Tests
- **CLI Commands**: `--list-checkpoints`, `--resume`, `--help`
- **Error Handling**: Missing checkpoints, invalid paths
- **Resume Functionality**: State restoration validation

### 🔍 Inference Pipeline Tests  
- **Prediction Scripts**: CLI interface, model loading
- **Image Processing**: Preprocessing, transforms, format handling
- **GradCAM**: Attention visualization dependencies
- **Error Handling**: Missing models, invalid inputs

## Test Results

Each test suite provides:
- ✅ **PASSED**: Test completed successfully
- ❌ **FAILED**: Test encountered errors (needs fixing)
- ⚠️ **WARNING**: Test completed with warnings (optional features)
- ⏭️ **SKIPPED**: Test file not found or not applicable

## Environment Requirements

The tests automatically check for:
- Python 3.8+
- Virtual environment activation
- Required dependencies (PyTorch, etc.)
- Project structure completeness

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
pip install -r requirements.txt
```

#### Virtual Environment Not Active
```bash
# Windows
.\.venv\Scripts\Activate.ps1

# Linux/macOS  
source .venv/bin/activate
```

#### Project Structure Issues
Ensure all core files exist:
- `src/` directory with all modules
- `config/config.py` 
- Main scripts (`train.py`, `predict_image.py`)

### Test-Specific Issues

#### Dataset Tests Failing
- Kaggle API not configured (expected in CI)
- Dataset not downloaded (run `python src/data/download_data.py`)

#### Training Tests Failing
- Missing PyTorch installation
- GPU/CUDA configuration issues

#### Inference Tests Failing
- Missing optional dependencies (GradCAM)
- PIL/OpenCV image processing issues

## CI/CD Integration

For automated testing:

```bash
# In CI pipeline
python tests/run_all_tests.py --quick  # Fast validation
python tests/run_all_tests.py          # Full test suite
```

Exit codes:
- `0`: All tests passed
- `1`: Some tests failed

## Development Workflow

1. **Before committing**: `python tests/run_all_tests.py --quick`
2. **Before releases**: `python tests/run_all_tests.py`  
3. **After major changes**: Run individual test suites
4. **Environment setup**: Tests validate proper installation

---

**💡 Pro Tip**: Run `python tests/run_all_tests.py --quick` for rapid validation during development!