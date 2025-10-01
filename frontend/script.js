// Cellex Frontend JavaScript
// Handles image upload, API communication, and results display

const API_URL = 'http://localhost:5000';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewSection = document.getElementById('previewSection');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// State
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAPIHealth();
});

// Setup Event Listeners
function setupEventListeners() {
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Buttons
    analyzeBtn.addEventListener('click', handleAnalyze);
    clearBtn.addEventListener('click', handleClear);
}

// File Selection Handler
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Drag and Drop Handlers
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showError('Please drop a valid image file');
    }
}

// Process Selected File
function processFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload PNG, JPG, or JPEG image.');
        return;
    }

    // Validate file size (16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File is too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;

    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        analyzeBtn.disabled = false;
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

// Clear Selection
function handleClear() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    analyzeBtn.disabled = true;
    hideResults();
    hideError();
}

// Analyze Image
async function handleAnalyze() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    // Show loading
    showLoading();
    hideError();
    hideResults();

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send request to API
        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        showError(`Analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Display Results
function displayResults(data) {
    // Update classification
    const resultClass = document.getElementById('resultClass');
    const resultConfidence = document.getElementById('resultConfidence');
    const resultBadge = document.getElementById('resultBadge');

    resultClass.textContent = data.class;
    resultConfidence.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    
    // Update badge
    resultBadge.textContent = data.class;
    resultBadge.className = 'result-badge';
    if (data.class_id === 0) {
        resultBadge.classList.add('normal');
    } else {
        resultBadge.classList.add('cancerous');
    }

    // Update probability bars
    const normalProb = data.probabilities.normal * 100;
    const cancerousProb = data.probabilities.cancerous * 100;

    const normalBar = document.getElementById('normalBar');
    const normalValue = document.getElementById('normalValue');
    const cancerousBar = document.getElementById('cancerousBar');
    const cancerousValue = document.getElementById('cancerousValue');

    normalBar.style.width = `${normalProb}%`;
    normalValue.textContent = `${normalProb.toFixed(1)}%`;
    cancerousBar.style.width = `${cancerousProb}%`;
    cancerousValue.textContent = `${cancerousProb.toFixed(1)}%`;

    // Show results section
    resultsSection.style.display = 'block';

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show/Hide Loading
function showLoading() {
    loadingIndicator.style.display = 'block';
    analyzeBtn.disabled = true;
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
    analyzeBtn.disabled = false;
}

// Show/Hide Error
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
}

function hideError() {
    errorMessage.style.display = 'none';
}

// Show/Hide Results
function hideResults() {
    resultsSection.style.display = 'none';
}

// Check API Health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('✓ API is healthy');
            if (!data.model_loaded) {
                console.warn('⚠ Model is not loaded. Train the model first.');
            }
        }
    } catch (error) {
        console.error('✗ Cannot connect to API:', error);
        console.log('Make sure the backend is running: python backend/app.py');
    }
}
