"""
Flask backend API for Cellex
Handles image upload and prediction
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'backend/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global predictor instance
predictor = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the trained model"""
    global predictor
    
    model_path = 'ml_model/models/checkpoints/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Please train the model first using: python ml_model/train.py")
        return False
    
    try:
        from ml_model.utils.predict import CellexPredictor
        predictor = CellexPredictor(model_path)
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Cellex API',
        'model_loaded': predictor is not None
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expects an image file in the request
    """
    try:
        # Check if model is loaded
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.',
                'details': 'Run: python ml_model/train.py'
            }), 503
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'allowed_types': list(ALLOWED_EXTENSIONS)
            }), 400
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = predictor.predict(image)
        
        # Add additional metadata
        result['filename'] = secure_filename(file.filename)
        result['image_size'] = image.size
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/api/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        'name': 'Cellex API',
        'version': '1.0.0',
        'description': 'AI-powered healthcare platform for X-ray image analysis',
        'endpoints': {
            '/health': 'Health check',
            '/api/predict': 'Image classification endpoint (POST)',
            '/api/info': 'API information'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB'
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'max_size': '16MB'
    }), 413


def main():
    """Run the Flask application"""
    print("=" * 60)
    print("Cellex Backend API")
    print("=" * 60)
    
    # Try to load model
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n" + "!" * 60)
        print("WARNING: Model not loaded!")
        print("The API will start, but predictions will fail.")
        print("Train the model first: python ml_model/train.py")
        print("!" * 60 + "\n")
    
    # Start server
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /health       - Health check")
    print("  GET  /api/info     - API information")
    print("  POST /api/predict  - Image prediction")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()
