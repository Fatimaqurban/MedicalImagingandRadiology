import os
import zipfile
import json
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key

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

# Available Models
AVAILABLE_MODELS = {
    'MobileNetV2': MobileNetV2,
    'ResNet50': ResNet50,
    'InceptionV3': InceptionV3
}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_image(filename):
    """Check if the file is an allowed image type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def validate_folder_structure(extracted_path):
    """
    Validate that the extracted zip has the required folder structure.
    Expected: A single dataset folder containing class subfolders.
    """
    items = os.listdir(extracted_path)
    print("Items in extracted_path:", items)  # Debug print
    if len(items) != 1:
        print("Validation Failed: Zip should contain exactly one root folder.")
        return False
    dataset_folder_name = items[0]
    dataset_folder = os.path.join(extracted_path, dataset_folder_name)
    if not os.path.isdir(dataset_folder):
        print(f"Validation Failed: {dataset_folder_name} is not a directory.")
        return False
    class_folders = os.listdir(dataset_folder)
    print("Class folders found:", class_folders)  # Debug print
    if len(class_folders) < 2:
        print("Validation Failed: Less than two class folders found.")
        return False
    for folder in class_folders:
        folder_path = os.path.join(dataset_folder, folder)
        if not os.path.isdir(folder_path):
            print(f"Validation Failed: {folder} is not a directory.")
            return False
    return True

def check_class_distribution(dataset_dir):
    """Check the distribution of classes in the dataset."""
    class_folders = os.listdir(dataset_dir)
    distribution = {}
    for folder in class_folders:
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if allowed_image(f)])
            distribution[folder] = num_images
    return distribution

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle the upload of the dataset zip file."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(zip_path)
            print(f"File saved to {zip_path}")

            # Extract the zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    extract_path = os.path.join(EXTRACT_FOLDER, filename.rsplit('.', 1)[0])
                    zip_ref.extractall(extract_path)
                print(f"File extracted to {extract_path}")
            except zipfile.BadZipFile:
                flash('Uploaded file is not a valid zip file.')
                print("Error: Bad zip file.")
                return redirect(request.url)

            # Validate folder structure
            if not validate_folder_structure(extract_path):
                flash('Invalid folder structure. Ensure the zip contains a single dataset folder with class subfolders.')
                print("Error: Invalid folder structure.")
                return redirect(request.url)

            # Get dataset folder name
            dataset_folder_name = os.listdir(extract_path)[0]
            # Save dataset_folder_name in session or pass it along
            dataset_name = dataset_folder_name  # This will be passed to the next route

            # Check class distribution
            dataset_dir = os.path.join(extract_path, dataset_folder_name)
            distribution = check_class_distribution(dataset_dir)
            print("Class Distribution:", distribution)
            flash(f'Class Distribution: {distribution}')

            flash('File successfully uploaded and extracted.')
            return redirect(url_for('select_model', dataset_folder_name=dataset_folder_name, extract_folder_name=filename.rsplit('.', 1)[0]))
        else:
            flash('Allowed file type is zip.')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/select_model/<extract_folder_name>/<dataset_folder_name>', methods=['GET', 'POST'])
def select_model(extract_folder_name, dataset_folder_name):
    """Allow the user to select a model and initiate training."""
    if request.method == 'POST':
        selected_model = request.form.get('model')
        if selected_model not in AVAILABLE_MODELS:
            flash('Invalid model selected.')
            print("Error: Invalid model selected.")
            return redirect(request.url)

        # Path to dataset folder
        dataset_dir = os.path.join(EXTRACT_FOLDER, extract_folder_name, dataset_folder_name)

        # Data Generators with Validation Split
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 20% for validation
        )

        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )

        # Check class distribution
        distribution = check_class_distribution(dataset_dir)
        print("Training Class Distribution:", distribution)
        flash(f'Training Class Distribution: {distribution}')

        # Calculate class weights to handle imbalance
        classes = list(train_generator.class_indices.keys())
        class_weights_calculated = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights_dict = dict(enumerate(class_weights_calculated))
        print("Class Weights:", class_weights_dict)
        flash(f'Class Weights: {class_weights_dict}')

        # Build the Model
        base_model_class = AVAILABLE_MODELS[selected_model]
        base_model = base_model_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Compile the Model
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the Model
        try:
            model.fit(
                train_generator,
                epochs=2,
                validation_data=validation_generator,
                class_weight=class_weights_dict
            )
        except Exception as e:
            flash(f'Error during training: {str(e)}')
            print(f"Error during training: {str(e)}")
            return redirect(request.url)

        # Save the Model
        model_filename = f"{dataset_folder_name}_{selected_model}.h5"
        model_save_path = os.path.join(MODEL_FOLDER, model_filename)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        # Save class indices
        class_indices = train_generator.class_indices
        with open(os.path.join(MODEL_FOLDER, f"{dataset_folder_name}_{selected_model}_class_indices.json"), 'w') as f:
            json.dump(class_indices, f)
        print(f"Class indices saved to {os.path.join(MODEL_FOLDER, f'{dataset_folder_name}_{selected_model}_class_indices.json')}")

        flash(f'Model {selected_model} trained and saved successfully!')
        return redirect(url_for('test_model', dataset_folder_name=dataset_folder_name, model_name=selected_model))
    else:
        return render_template('select_model.html', models=AVAILABLE_MODELS.keys(), dataset=dataset_folder_name)

@app.route('/test_model/<dataset_folder_name>/<model_name>', methods=['GET', 'POST'])
def test_model(dataset_folder_name, model_name):
    """Allow the user to upload a test image and get a prediction."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading.')
            return redirect(request.url)
        if file and allowed_image(file.filename):
            filename = secure_filename(file.filename)
            test_image_path = os.path.join(TEST_UPLOAD_FOLDER, filename)
            file.save(test_image_path)
            print(f"Test image saved to {test_image_path}")

            # Load the Model
            model_filename = f"{dataset_folder_name}_{model_name}.h5"
            model_path = os.path.join(MODEL_FOLDER, model_filename)
            if not os.path.exists(model_path):
                flash('Model file not found. Please train the model first.')
                print("Error: Model file not found.")
                return redirect(url_for('upload_file'))
            try:
                model = tf.keras.models.load_model(model_path)
                print(f"Model {model_filename} loaded successfully.")
            except Exception as e:
                flash(f'Error loading model: {str(e)}')
                print(f"Error loading model: {str(e)}")
                return redirect(request.url)

            # Load class indices
            class_indices_path = os.path.join(MODEL_FOLDER, f"{dataset_folder_name}_{model_name}_class_indices.json")
            if not os.path.exists(class_indices_path):
                flash('Class indices file not found.')
                print("Error: Class indices file not found.")
                return redirect(request.url)
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            # Reverse the class indices to get class labels
            class_labels = {v: k for k, v in class_indices.items()}
            print("Class Labels:", class_labels)

            # Preprocess the Image
            try:
                img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                img_array /= 255.0
                print("Image preprocessed successfully.")
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                print(f"Error processing image: {str(e)}")
                return redirect(request.url)

            # Make Prediction
            try:
                prediction = model.predict(img_array)
                probability = prediction[0][0]
                # Determine class based on probability and class mapping
                threshold = 0.5  # You can adjust this threshold if needed
                predicted_class = 1 if probability > threshold else 0
                result = class_labels.get(predicted_class, "Unknown")
                print(f"Prediction Probability: {probability:.4f}, Predicted Class: {result}")
            except Exception as e:
                flash(f'Error during prediction: {str(e)}')
                print(f"Error during prediction: {str(e)}")
                return redirect(request.url)

            # Move the uploaded image to static folder for display
            static_test_upload_folder = os.path.join('static', 'test_uploads')
            os.makedirs(static_test_upload_folder, exist_ok=True)
            static_test_image_path = os.path.join(static_test_upload_folder, filename)
            # Remove if already exists to avoid errors
            if os.path.exists(static_test_image_path):
                os.remove(static_test_image_path)
            os.rename(test_image_path, static_test_image_path)
            print(f"Test image moved to {static_test_image_path}")

            flash(f'Prediction Probability: {probability:.4f}')
            flash(f'Prediction Result: {result}')
            return redirect(request.url)
        else:
            flash('Allowed image types are png, jpg, jpeg.')
            return redirect(request.url)
    return render_template('test_model.html', dataset=dataset_folder_name, model_name=model_name)

# Route to serve uploaded test images
@app.route('/static/test_uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join('static', 'test_uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True)
