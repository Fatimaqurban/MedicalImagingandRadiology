## Project Overview
This system is a Flask-based web application that provides medical image classification capabilities for various disease categories including Brain Tumors, Eye Diseases, Lung Diseases, and Skin Diseases. The application supports both pre-trained models and the ability to train new models with custom datasets.

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


# Medical Imaging Web Application

This project implements a web-based application for training and testing deep learning models on medical imaging data. It allows users to upload a dataset, train a model, and test the model's predictions on new images.




<!--## Project Structure

The application is organized into the following directory structure:
medical_imaging_app/
   ├── app.py # Main Flask application file 
   ├── frontend
      ├── src
         ├── pages
             ├── ML_UI.js
             └── TrainedModels.js
   ├── uploads/ # Directory for uploaded zip files 
   ├── extracted/ # Directory for extracted datasets 
   ├── models/ # Directory for trained model files 
   └── test_uploads/ # Directory for uploaded test images
   -->


## Installation

### Prerequisites

Before you begin, ensure you have the following installed on your system:
- Python 3.x
- Pip (Python package installer)

### Setup

To set up the project, follow these steps:

1. **Clone the Repository:**
   git clone https://github.com/Fatimaqurban/MedicalImagingandRadiology.git
   cd medical_imaging_app
2. **Install Required Packages:**
   pip install flask  flask-cors tensorflow werkzeug transformers tailwind torch numpy pillow torchvision react-router-dom lucide-react axios

## Usage
Running the Application (run the below commands simultaneously)
To run the  backend, execute the following command from the root of the project directory:
  python app.py

To run the frontend, execute the following command from the frontend directory for running the of the root folder:
  npm start



## Using the Application
1. **Upload Dataset:**
   Navigate to the home page (http://127.0.0.1:5000/).
   Use the upload form to upload a dataset zip file formatted as specified.
   ![image](https://github.com/user-attachments/assets/688bf1c1-3470-4cb7-9503-930951b25bfd)


3. **Train Model:**
  After uploading the dataset, select a model for training.
  The application will train the model and redirect you to the testing page upon completion.
  ![image](https://github.com/user-attachments/assets/0755b4f3-fea1-470e-99e3-1901f0768be3)


5. **Test Model:**
  On the testing page, upload an image to evaluate the trained model.
  The result will be displayed on the same page.
  ![image](https://github.com/user-attachments/assets/f46fa9f0-674e-4641-8d7c-2ba2ddce6c9f)


## Built With
  Flask - The web framework used.
  TensorFlow - The machine learning framework used.
  Werkzeug - A comprehensive WSGI web application library.

# Models being used in Application
1. MobileNetV2
2. ResNet50
3. InceptionV3




# Skin Cancer Detection Model
The files upload in the skin cancer folder are for detecting the skin cancer, the model has been trained on mobile net v2 architecture and currently diagnosis three types of skin cancer namely
1. Nevus
2. Melanoma
3. Pigmented Benign Keratosis

## Usage
The folder interface has the code for the frontend page for uploading the image for detection, before are the attached images of frontend
![Screenshot 2024-11-10 180706](https://github.com/user-attachments/assets/eb2153b4-fcf5-429b-b4e3-acfcca92c922)

![Screenshot 2024-11-10 180815](https://github.com/user-attachments/assets/b6996d16-1156-489b-b513-e3bc3fc89084)

![Screenshot 2024-11-10 180834](https://github.com/user-attachments/assets/dbecef16-7e57-4176-ad64-ce7d16fdbdb2)


While the folder prediction has the model file for prediction and the backend code in flask for processing the image from the frontend and using the model to detect the type of cancer




