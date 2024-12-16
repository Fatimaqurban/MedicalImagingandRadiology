## Project Overview
Our system is a Flask-based web application that provides medical image classification capabilities for various disease categories including Brain Tumors, Eye Diseases, Lung Diseases, and Skin Diseases. The application supports both pre-trained models and the ability to train new models with custom datasets.

## System Architecture

### Core Components
1. **Flask Backend Server**: Handles all API endpoints and model operations
2. **TensorFlow/Keras**: Powers the deep learning models
3. **Pre-trained Models**: Ready-to-use models for specific disease categories
4. **Custom Model Training Pipeline**: Supports training new models with user data
5. **File Management System**: Handles dataset uploads and organization

### Directory Structure
```
root/
├── uploads/               # Temporary storage for uploaded zip files
├── extracted/            # Extracted dataset files
├── test_uploads/        # Storage for test images
├── datasets/            # Combined dataset storage
├── Pre-trained Models:
│   ├── BrainTumor_model.h5
│   ├── EyeDisease_model.h5
│   ├── LungDisease_model.h5
│   └── SkinDisease_model.keras
└── disease_config.py    # Disease categories configuration
```

## Core Functionalities

### 1. Disease Category Management
- Maintains a configuration of supported disease categories
- Dynamically updates categories when new classes are added
- Provides API endpoints to retrieve available categories and classes

### 2. Model Management
Available Models:
- MobileNetV2
- ResNet50
- InceptionV3

Pre-trained Models:
- Brain Tumor Detection
- Eye Disease Classification
- Lung Disease Detection
- Skin Disease Classification

### 3. Dataset Processing Pipeline

#### Upload Process
1. Receives ZIP file containing categorized images
2. Validates file structure and contents
3. Extracts and organizes images into appropriate directories
4. Performs dataset validation checks:
   - Minimum 10 images per class
   - Valid image formats (PNG, JPG, JPEG)
   - Proper folder structure

#### Dataset Validation Features
- Fuzzy matching for class names
- Automatic class mapping
- New class detection
- Dataset merging capabilities

### 4. Model Training Workflow

1. **Dataset Preparation**
   - Image preprocessing
   - Data augmentation
   - Train/validation split (80/20)

2. **Model Architecture**
   - Base model selection (MobileNetV2/ResNet50/InceptionV3)
   - Transfer learning implementation
   - Custom classification layers

3. **Training Configuration**
   - Learning rate: 1e-4
   - Loss function: Categorical Crossentropy
   - Metrics: Accuracy
   - Batch size: 32
   - Image size: 224x224

### 5. Prediction Pipeline

Two prediction pathways:
1. **Custom-trained Models** (`/predict` endpoint)
2. **Pre-trained Models** (`/pretrainedmodels` endpoint)

Prediction Process:
1. Image preprocessing
2. Model inference
3. Confidence threshold checking (0.6)
4. Class label mapping
5. Result generation

## API Endpoints

### 1. Configuration Endpoints
- `GET /get-disease-categories`: Retrieve available disease categories
- `POST /get-model-classes`: Get classes for a specific disease category
- `POST /check-existing-model`: Check for existing models

### 2. Data Management Endpoints
- `POST /upload`: Handle dataset uploads and processing
- `POST /select-model`: Initiate model training

### 3. Prediction Endpoints
- `POST /predict`: Custom model predictions
- `POST /pretrainedmodels`: Pre-trained model predictions

## Error Handling

The system implements comprehensive error handling for:
- Invalid file formats
- Incorrect folder structures
- Insufficient training data
- Invalid disease categories
- Model loading failures
- Prediction errors

## Security Features
1. Secure filename handling
2. File extension validation
3. Directory access control
4. CORS support
5. Secret key implementation

## Performance Considerations

### Model Training
- Minimal data augmentation to reduce training time
- Transfer learning for faster convergence
- Configurable epoch count
- Batch processing

### Inference
- Efficient image preprocessing
- Confidence thresholding
- Memory management for uploaded files

## Future Enhancement Possibilities
1. Model versioning system
2. Advanced data augmentation options
3. Model performance analytics
4. Batch prediction capabilities
5. API authentication
6. Extended disease categories
7. Model explainability features
8. Dataset quality assessment tools

## Technical Requirements
- Python 3.x
- TensorFlow 2.x
- Flask
- NumPy
- Werkzeug
- Flask-CORS
- Sufficient storage for datasets and models
- GPU recommended for training (optional)

