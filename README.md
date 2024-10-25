# Medical Imaging Web Application

This project implements a web-based application for training and testing deep learning models on medical imaging data. It allows users to upload a dataset, train a model, and test the model's predictions on new images.

<!--## Project Structure

The application is organized into the following directory structure:
medical_imaging_app/
   ├── app.py # Main Flask application file 
   ├── templates/ # HTML templates for the web interface
      ├── index.html  
      ├── select_model.html 
      └── test_model.html 
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
   pip install flask tensorflow werkzeug

## Usage
Running the Application
To run the application, execute the following command from the root of the project directory:
  python app.py

This will start the Flask server, and the application will be accessible via http://127.0.0.1:5000/ on your browser.

## Using the Application
1. **Upload Dataset:**
   Navigate to the home page (http://127.0.0.1:5000/).
   Use the upload form to upload a dataset zip file formatted as specified.
   ![image](https://github.com/user-attachments/assets/0e7df06a-506e-479d-a7e1-736aba0a8982)


3. **Train Model:**
  After uploading the dataset, select a model for training.
  The application will train the model and redirect you to the testing page upon completion.
  ![image](https://github.com/user-attachments/assets/9014a53d-269f-4eb6-8a44-06bf7b295ebd)


5. **Test Model:**
  On the testing page, upload an image to evaluate the trained model.
  The result will be displayed on the same page.
  ![image](https://github.com/user-attachments/assets/f46fa9f0-674e-4641-8d7c-2ba2ddce6c9f)


## Built With
  Flask - The web framework used.
  TensorFlow - The machine learning framework used.
  Werkzeug - A comprehensive WSGI web application library.

# Models being used in Application
### MobileNetV2
### ResNet50
### InceptionV3


