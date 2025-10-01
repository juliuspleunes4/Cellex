# Cellex 🏥

**Cellex: AI for Early Insight**

An AI-powered healthcare platform that analyzes X-ray images to detect potentially cancerous cells using deep learning.

## 🚀 Features

- **Deep Learning Model**: Custom CNN architecture built with PyTorch for binary classification
- **Training Pipeline**: Complete training workflow with data augmentation, metrics tracking, and model checkpointing
- **REST API**: Flask-based backend API for image upload and prediction
- **Modern Frontend**: Clean, responsive web interface for image analysis
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix
- **Synthetic Dataset Generation**: Built-in synthetic data generator for testing and demonstration

## 📋 Project Structure

```
Cellex/
├── ml_model/              # Machine Learning components
│   ├── model.py          # CNN architecture (CellexCNN)
│   ├── dataset.py        # Dataset handling and transforms
│   ├── train.py          # Training script
│   ├── utils/
│   │   ├── predict.py    # Prediction utilities
│   │   └── __init__.py
│   ├── models/           # Saved model checkpoints
│   └── datasets/         # Training data
├── backend/              # Flask API
│   ├── app.py           # API endpoints
│   └── uploads/         # Uploaded images
├── frontend/            # Web interface
│   ├── index.html      # Main HTML
│   ├── style.css       # Styling
│   └── script.js       # Frontend logic
├── tests/              # Test files
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/juliuspleunes4/Cellex.git
   cd Cellex
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### 1. Train the Model

Train the model using synthetic data (for demonstration):

```bash
python ml_model/train.py --use-synthetic --epochs 20 --batch-size 16 --num-samples 200
```

**Training Options:**
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 16)
- `--learning-rate`: Learning rate (default: 0.001)
- `--num-samples`: Number of synthetic samples per class (default: 200)
- `--output-dir`: Directory to save models (default: ml_model/models)

The training script will:
- Generate synthetic X-ray images
- Train the model with data augmentation
- Save checkpoints and the best model
- Output training metrics (loss, accuracy, precision, recall, F1-score)

### 2. Start the Backend API

```bash
python backend/app.py
```

The API will start at `http://localhost:5000`

**API Endpoints:**
- `GET /health` - Health check
- `GET /api/info` - API information
- `POST /api/predict` - Image prediction (upload image file)

### 3. Open the Frontend

Open `frontend/index.html` in your web browser, or serve it using a simple HTTP server:

```bash
# Python 3
cd frontend
python -m http.server 8000

# Then visit http://localhost:8000 in your browser
```

### 4. Analyze X-ray Images

1. Upload an X-ray image (PNG, JPG, JPEG)
2. Click "Analyze Image"
3. View the results:
   - Classification (Normal vs. Potentially Cancerous)
   - Confidence score
   - Probability distribution

## 🧠 Model Architecture

The CellexCNN model uses a custom convolutional neural network architecture:

- **3 Convolutional Blocks**: Each with two conv layers, batch normalization, max pooling, and dropout
- **Channel Progression**: 3 → 32 → 64 → 128 → 256
- **Global Average Pooling**: Reduces spatial dimensions
- **Fully Connected Layers**: 256 → 512 → 256 → 2 (with dropout)
- **Output**: 2 classes (Normal, Potentially Cancerous)

**Key Features:**
- Batch normalization for stable training
- Dropout layers to prevent overfitting
- Data augmentation (rotation, flipping, color jitter)
- Adam optimizer with learning rate scheduling

## 📊 Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Proportion of correct positive predictions
- **Recall**: Proportion of actual positives identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/false positives and negatives

## 🔬 Using Real Medical Data

To use real medical imaging data:

1. Organize your data in the following structure:
   ```
   ml_model/datasets/
   ├── normal/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── cancerous/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

2. Modify the `dataset.py` to load your data structure

3. Train without the `--use-synthetic` flag:
   ```bash
   python ml_model/train.py --data-dir ml_model/datasets/your_data
   ```

## ⚠️ Important Disclaimer

**This is a demonstration project for educational purposes only.**

- This tool should **NOT** be used for actual medical diagnosis
- Always consult qualified healthcare professionals for medical advice
- Real medical AI systems require:
  - Extensive validation on diverse datasets
  - Regulatory approval (FDA, CE marking, etc.)
  - Clinical trials and peer review
  - Proper data privacy and security measures

## 🧪 Testing

Run tests (if available):
```bash
python -m pytest tests/
```

## 📝 Development

### Adding New Features

1. **Custom Model Architecture**: Modify `ml_model/model.py`
2. **Data Processing**: Update `ml_model/dataset.py`
3. **API Endpoints**: Add to `backend/app.py`
4. **Frontend Features**: Update `frontend/script.js` and `frontend/index.html`

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Comment complex logic

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with relevant regulations when working with medical data.

## 🙏 Acknowledgments

- Built with PyTorch and Flask
- Inspired by medical AI research
- Created for learning and demonstration purposes

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Remember**: This is a demonstration tool. Never use it for actual medical decisions!
