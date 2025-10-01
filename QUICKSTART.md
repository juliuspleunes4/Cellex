# Cellex Quick Start Guide

Get Cellex up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- pip

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (First Time Only)

Train a model on synthetic data for demonstration:

```bash
python ml_model/train.py --use-synthetic --epochs 5 --num-samples 50
```

This takes about 4-5 minutes on CPU.

### 3. Start the Backend API

In one terminal:

```bash
python backend/app.py
```

The API will start at `http://localhost:5000`

### 4. Start the Frontend

In another terminal:

```bash
cd frontend
python -m http.server 8000
```

The frontend will be available at `http://localhost:8000`

### 5. Use the Application

1. Open your browser and go to `http://localhost:8000`
2. Upload an X-ray image (or use one from `ml_model/datasets/synthetic/`)
3. Click "Analyze Image"
4. View the results!

## Testing the API Directly

Test the API endpoints with curl:

```bash
# Health check
curl http://localhost:5000/health

# API information
curl http://localhost:5000/api/info

# Predict with an image
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/api/predict
```

## Running Tests

Run the unit tests:

```bash
python -m unittest tests.test_model
python -m unittest tests.test_dataset
```

## Using the Demo Script

Alternatively, use the demo script for guided setup:

```bash
python run_demo.py
```

## Troubleshooting

### Model not found error
If you get a "model not found" error, train the model first:
```bash
python ml_model/train.py --use-synthetic --epochs 5 --num-samples 50
```

### Port already in use
If port 5000 or 8000 is already in use, you can:
- Stop the service using that port
- Or modify the port in `backend/app.py` or the frontend server command

### Dependencies issues
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CORS errors
The backend has CORS enabled by default. If you still get CORS errors, make sure:
- The backend is running on port 5000
- The frontend JavaScript is correctly pointing to `http://localhost:5000`

## Next Steps

- See [README.md](README.md) for detailed documentation
- Train with more epochs for better accuracy
- Try using real medical imaging datasets (see README.md for instructions)
- Customize the model architecture in `ml_model/model.py`
- Extend the API with additional endpoints

## Architecture Overview

```
┌─────────────────┐
│    Frontend     │  (HTML/CSS/JS)
│  Port: 8000     │
└────────┬────────┘
         │
         │ HTTP POST (image)
         ▼
┌─────────────────┐
│   Flask API     │  (Python)
│  Port: 5000     │
└────────┬────────┘
         │
         │ Prediction
         ▼
┌─────────────────┐
│  PyTorch Model  │  (CNN)
│   CellexCNN     │
└─────────────────┘
```

## Important Notes

⚠️ **Medical Disclaimer**: This is a demonstration tool for educational purposes only. Never use it for actual medical diagnosis. Always consult qualified healthcare professionals.

📚 **Synthetic Data**: The default training uses synthetic data. For real use cases, you would need actual medical imaging datasets with proper labels and permissions.

🔒 **Security**: This is a development setup. For production, you would need:
- HTTPS/SSL certificates
- Authentication and authorization
- Input validation and sanitization
- Rate limiting
- Proper error handling
- HIPAA compliance (if handling real medical data)

---

**Need help?** Check the [README.md](README.md) or open an issue on GitHub.
