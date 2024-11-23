# app.py

import os
import zipfile
import json
import glob
import numpy as np
import tensorflow as tf
from io import BytesIO
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.models import Model
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from difflib import get_close_matches

from disease_config import DISEASE_CATEGORIES

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key'

# Configuration
UPLOAD_FOLDER = 'uploads'
EXTRACT_FOLDER = 'extracted'
MODEL_FOLDER = 'models'
TEST_UPLOAD_FOLDER = 'test_uploads'
ALLOWED_EXTENSIONS = {'zip'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure necessary directories exist
for folder in [UPLOAD_FOLDER, EXTRACT_FOLDER, MODEL_FOLDER, TEST_UPLOAD_FOLDER]:
    os.makedirs(folder, exist_ok=True)

AVAILABLE_MODELS = {
    'MobileNetV2': MobileNetV2,
    'ResNet50': ResNet50,
    'InceptionV3': InceptionV3
}

# Pre-trained models in the root directory
PRETRAINED_MODEL_PATHS = {
    'Brain Tumor': 'BrainTumor_model.h5',
    'Eye Disease': 'EyeDisease_model.h5',
    'Lung Disease': 'LungDisease_model.h5'
}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def allowed_image(filename):
    """Check if the file is an allowed image type."""
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS)

# Endpoint to get disease categories
@app.route('/get-disease-categories', methods=['GET'])
def get_disease_categories():
    return jsonify(DISEASE_CATEGORIES), 200

# Endpoint to get existing model classes
@app.route('/get-model-classes', methods=['POST'])
def get_model_classes():
    data = request.get_json()
    disease_category = data.get('disease_category')
    if not disease_category or disease_category not in DISEASE_CATEGORIES:
        return jsonify({'error': 'Invalid disease category.'}), 400

    model_path = get_existing_model(disease_category)
    if model_path:
        class_indices_path = model_path.replace('.h5', '_class_indices.json')
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            classes = list(class_indices.values())
            return jsonify({'classes': classes}), 200
        else:
            # Use disease_config.py as fallback
            default_classes = DISEASE_CATEGORIES.get(disease_category, [])
            return jsonify({'classes': default_classes}), 200
    else:
        # Return default classes from disease_config.py
        default_classes = DISEASE_CATEGORIES.get(disease_category, [])
        return jsonify({'classes': default_classes}), 200

def get_existing_model(disease_category):
    # First, check if pre-trained model exists in root directory
    pretrained_model_path = PRETRAINED_MODEL_PATHS.get(disease_category)
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        return pretrained_model_path
    # Else, check in models folder
    model_files = glob.glob(
        os.path.join(MODEL_FOLDER, f"{disease_category}_Model_v*.h5"))
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        return latest_model
    return None

def load_model_and_classes(model_path):
    model = models.load_model(model_path)
    class_indices_path = model_path.replace('.h5', '_class_indices.json')
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
    else:
        # Use disease_config.py as fallback
        disease_category = get_disease_category_from_model_path(model_path)
        class_indices = {cls: idx for idx, cls in enumerate(
            DISEASE_CATEGORIES.get(disease_category, []))}
    return model, class_indices

def get_disease_category_from_model_path(model_path):
    filename = os.path.basename(model_path)
    for category in DISEASE_CATEGORIES.keys():
        if category.replace(' ', '') in filename.replace('_Model', ''):
            return category
    return None

def update_model_for_new_classes(model, old_num_classes, new_num_classes):
    # Freeze all layers except the last few
    for layer in model.layers[:-1]:
        layer.trainable = False
    # Replace the last Dense layer with a new one
    x = model.layers[-2].output  # Get output of the layer before last
    new_output = layers.Dense(new_num_classes, activation='softmax',
                              name='predictions')(x)
    new_model = Model(inputs=model.input, outputs=new_output)
    return new_model

def validate_dataset(dataset_dir, disease_category, existing_classes):
    expected_classes = DISEASE_CATEGORIES.get(disease_category, [])
    actual_classes = os.listdir(dataset_dir)

    # Use fuzzy matching for class names
    class_mapping = {}
    new_classes = []
    for actual_class in actual_classes:
        match = get_close_matches(
            actual_class.lower(),
            [cls.lower() for cls in expected_classes],
            n=1, cutoff=0.8)
        if match:
            matched_class = expected_classes[
                [cls.lower() for cls in expected_classes].index(match[0])]
            class_mapping[actual_class] = matched_class
        else:
            # For new classes, accept them
            class_mapping[actual_class] = actual_class  # Keep original name
            if actual_class not in new_classes:
                new_classes.append(actual_class)

    # Check images in each class
    for actual_class in actual_classes:
        class_path = os.path.join(dataset_dir, actual_class)
        images = os.listdir(class_path)
        if len(images) < 10:
            return (False, f"Class '{actual_class}' must have at least "
                           f"10 images.", None)
        for img_name in images:
            if not allowed_image(img_name):
                return (False, f"Invalid image format: {img_name}", None)
    return True, class_mapping, new_classes

def update_disease_config(disease_category, new_classes):
    # Update the DISEASE_CATEGORIES with new classes
    existing_classes = DISEASE_CATEGORIES.get(disease_category, [])
    updated_classes = existing_classes.copy()
    for cls in new_classes:
        if cls not in existing_classes:
            updated_classes.append(cls)
    DISEASE_CATEGORIES[disease_category] = updated_classes
    # Save to disease_config.py
    with open('disease_config.py', 'w') as f:
        f.write(f'DISEASE_CATEGORIES = {json.dumps(DISEASE_CATEGORIES)}')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    disease_category = request.form.get('disease_category')

    if not disease_category or disease_category not in DISEASE_CATEGORIES:
        return jsonify({'error': 'Invalid or missing disease category.'}), 400

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(zip_path)
        print(f"File saved to {zip_path}")

        # Extract the zip file
        try:
            extract_folder_name = filename.rsplit('.', 1)[0]
            extract_path = os.path.join(EXTRACT_FOLDER, extract_folder_name)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"File extracted to {extract_path}")
        except zipfile.BadZipFile:
            print("Error: Bad zip file.")
            return jsonify({'error': 'Uploaded file is not a valid zip '
                           'file.'}), 400

        # Validate folder structure
        items = os.listdir(extract_path)
        if len(items) != 1:
            print("Error: Invalid folder structure.")
            return jsonify({'error': 'Invalid folder structure. Ensure the '
                           'zip contains a single dataset folder with class '
                           'subfolders.'}), 400

        dataset_folder_name = items[0]
        dataset_dir = os.path.join(extract_path, dataset_folder_name)

        if not os.path.isdir(dataset_dir):
            print("Error: Dataset folder is not a directory.")
            return jsonify({'error': 'Invalid dataset folder structure.'}), 400

        # Get existing classes from disease_config.py
        existing_classes = DISEASE_CATEGORIES.get(disease_category, [])

        # Validate dataset
        is_valid, class_mapping, new_classes = validate_dataset(
            dataset_dir, disease_category, existing_classes.copy())
        if not is_valid:
            # class_mapping contains the error message here
            return jsonify({'error': class_mapping}), 400

        if not new_classes:
            # No new classes, all classes already exist
            return jsonify({
                'message': 'Your desired features are already available in '
                           'the model and ready to be tested. No need to '
                           'retrain.',
                'retraining_needed': False,
                'existing_classes': existing_classes,
                'disease_category': disease_category
            }), 200

        # Proceed with training
        updated_classes = existing_classes + new_classes
        # Return JSON response with class distribution and other info
        return jsonify({
            'message': 'File successfully uploaded and extracted.',
            'class_distribution': class_mapping,
            'extract_folder_name': extract_folder_name,
            'dataset_folder_name': dataset_folder_name,
            'disease_category': disease_category,
            'updated_classes': updated_classes,
            'new_classes': new_classes,
            'retraining_needed': True
        }), 200

    return jsonify({'error': 'Allowed file type is zip.'}), 400

@app.route('/select-model/<dataset_folder_name>', methods=['POST'])
def select_model(dataset_folder_name):
    data = request.get_json()
    selected_model = data.get('model')
    disease_category = data.get('disease_category')
    updated_classes = data.get('updated_classes')  # List of updated classes
    new_classes = data.get('new_classes', [])  # List of new classes

    if not disease_category or disease_category not in DISEASE_CATEGORIES:
        return jsonify({"error": "Invalid disease category."}), 400

    if selected_model not in AVAILABLE_MODELS:
        return jsonify({"error": "Invalid model selected."}), 400

    # Get existing classes from disease_config.py
    existing_classes = DISEASE_CATEGORIES.get(disease_category, [])

    # Combine existing classes and new classes
    all_classes = existing_classes + [cls for cls in new_classes if cls not in existing_classes]

    # Path to dataset folder
    dataset_dir = os.path.join(EXTRACT_FOLDER, dataset_folder_name)

    # Data Generators with minimal augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # 20% for validation
    )

    # Image size depends on the model
    img_height, img_width = 224, 224

    # Prepare data generators
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        classes=all_classes  # Use all classes
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        classes=all_classes  # Use all classes
    )

    num_classes = len(all_classes)

    # Check for existing model
    model_path = get_existing_model(disease_category)
    if model_path:
        # Load existing model
        model = models.load_model(model_path)
        # Update model for new classes
        model = update_model_for_new_classes(
            model, model.output_shape[-1], num_classes)
    else:
        # Build new model
        base_model_class = AVAILABLE_MODELS[selected_model]
        base_model = base_model_class(
            weights='imagenet', include_top=False,
            input_shape=(img_height, img_width, 3))

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        predictions = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

    # Compile model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(
        train_generator,
        epochs=5,  # Adjust as needed
        validation_data=validation_generator
    )

    # Save model
    version = 1
    model_save_path = os.path.join(
        MODEL_FOLDER, f"{disease_category}_Model_v{version}.h5")
    while os.path.exists(model_save_path):
        version += 1
        model_save_path = os.path.join(
            MODEL_FOLDER, f"{disease_category}_Model_v{version}.h5")
    model.save(model_save_path)

    # Save class indices
    class_indices = train_generator.class_indices
    # Invert class_indices to map indices to class names
    class_indices = {v: k for k, v in class_indices.items()}
    class_indices_path = model_save_path.replace('.h5', '_class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f)

    # Update disease_config
    update_disease_config(disease_category, new_classes)

    return jsonify({"message": f"Model {selected_model} trained and saved "
                               f"successfully!"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # This endpoint is used for models trained through the application
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    disease_category = request.form.get('disease_category')

    if not disease_category or disease_category not in DISEASE_CATEGORIES:
        return jsonify({'error': 'Invalid or missing disease category.'}), 400

    model_path = get_existing_model(disease_category)
    if not model_path:
        return jsonify({'error': 'No trained model exists for the selected '
                                 'category.'}), 400

    # Load model and class indices
    model, class_indices = load_model_and_classes(model_path)
    # Reverse class indices
    class_labels = {int(v): k for k, v in class_indices.items()}

    # Preprocess image
    img_height, img_width = model.input_shape[1], model.input_shape[2]

    # Read the image data from the uploaded file
    image = keras.preprocessing.image.load_img(
        BytesIO(file.read()), target_size=(img_height, img_width)
    )

    # Reset the file pointer to the beginning
    file.stream.seek(0)

    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = input_arr / 255.0

    # Make prediction
    predictions = model.predict(input_arr)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_label = class_labels.get(predicted_class_idx, 'Unknown')

    # Optionally, get probability/confidence
    confidence = float(np.max(predictions[0]))

    return jsonify({'predicted_class': predicted_class_label,
                    'confidence': confidence}), 200

@app.route('/pretrainedmodels', methods=['POST'])
def pretrained_models_predict():
    # This endpoint uses the pre-trained models in the root directory
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    disease_category = request.form.get('disease_category')

    if not disease_category or disease_category not in PRETRAINED_MODEL_PATHS:
        return jsonify({'error': 'Invalid or missing disease category.'}), 400

    model_path = PRETRAINED_MODEL_PATHS.get(disease_category)
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'Pre-trained model not found for the '
                                 'selected category.'}), 400

    # Load model and class indices
    model, class_indices = load_model_and_classes(model_path)
    # Reverse class indices
    class_labels = {int(v): k for k, v in class_indices.items()}

    # Preprocess image
    img_height, img_width = model.input_shape[1], model.input_shape[2]

    # Read the image data from the uploaded file
    image = keras.preprocessing.image.load_img(
        BytesIO(file.read()), target_size=(img_height, img_width)
    )

    # Reset the file pointer to the beginning
    file.stream.seek(0)

    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = input_arr / 255.0

    # Make prediction
    predictions = model.predict(input_arr)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_label = class_labels.get(predicted_class_idx, 'Unknown')

    # Optionally, get probability/confidence
    confidence = float(np.max(predictions[0]))

    return jsonify({'predicted_class': predicted_class_label,
                    'confidence': confidence}), 200

@app.route('/check-existing-model', methods=['POST'])
def check_existing_model():
    data = request.get_json()
    disease_category = data.get('disease_category')

    if not disease_category or disease_category not in DISEASE_CATEGORIES:
        return jsonify({'error': 'Invalid disease category.'}), 400

    model_path = get_existing_model(disease_category)
    if model_path:
        class_indices_path = model_path.replace('.h5', '_class_indices.json')
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            classes = [value for key, value in class_indices.items()]
            return jsonify({'model_name': os.path.basename(model_path),
                            'classes': classes}), 200
        else:
            # Use disease_config.py as fallback
            classes = DISEASE_CATEGORIES.get(disease_category, [])
            return jsonify({'model_name': os.path.basename(model_path),
                            'classes': classes}), 200
    else:
        return jsonify({'message': 'No model found for the selected '
                                   'category.'}), 404

if __name__ == '__main__':
    app.run(debug=True)
