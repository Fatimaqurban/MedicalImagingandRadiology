import os
import zipfile
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
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

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle the upload, extraction, and validation of the dataset zip file, and return class distribution."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
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
            return jsonify({'error': 'Uploaded file is not a valid zip file.'}), 400

        # Validate folder structure
        items = os.listdir(extract_path)
        if len(items) != 1:
            print("Error: Invalid folder structure.")
            return jsonify({'error': 'Invalid folder structure. Ensure the zip contains a single dataset folder with class subfolders.'}), 400

        dataset_folder_name = items[0]
        dataset_dir = os.path.join(extract_path, dataset_folder_name)

        if not os.path.isdir(dataset_dir):
            print("Error: Dataset folder is not a directory.")
            return jsonify({'error': 'Invalid dataset folder structure.'}), 400

        # Check class distribution
        class_distribution = {}
        class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
        if len(class_folders) < 2:
            print("Error: Dataset should contain at least two classes.")
            return jsonify({'error': 'Dataset should contain at least two classes.'}), 400

        for class_folder in class_folders:
            class_path = os.path.join(dataset_dir, class_folder)
            num_files = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_distribution[class_folder] = num_files

        # Return JSON response with class distribution and other info
        return jsonify({
            'message': 'File successfully uploaded and extracted.',
            'class_distribution': class_distribution,
            'extract_folder_name': extract_folder_name,
            'dataset_folder_name': dataset_folder_name
        }), 200

    return jsonify({'error': 'Allowed file type is zip.'}), 400

@app.route('/select-model/<dataset_folder_name>', methods=['POST'])
def select_model(dataset_folder_name):
    """Allow the user to select a model and initiate training."""
    data = request.get_json()
    selected_model = data.get('model')

    if selected_model not in AVAILABLE_MODELS:
        return jsonify({"error": "Invalid model selected."}), 400

    # Path to dataset folder
    dataset_dir = os.path.join(EXTRACT_FOLDER, dataset_folder_name, "dataset")

    # Data Generators with minimal augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% for validation
    )


    try:
        # Load training and validation data
        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(224, 224),
            batch_size=8,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(224, 224),
            batch_size=8,
            class_mode='categorical',
            subset='validation'
        )

        num_classes = train_generator.num_classes
        print(f"Number of classes detected: {num_classes}")
        if num_classes <= 1:
            return jsonify({"error": "Dataset should contain at least two classes."}), 400

        # Model Building with fine-tuning
        base_model_class = AVAILABLE_MODELS[selected_model]
        base_model = base_model_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Unfreeze last 10 layers for fine-tuning
        for layer in base_model.layers[-10:]:
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)  # Reduced dense layer size
        x = Dropout(0.5)(x)  # Dropout rate of 50%
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile Model with a smaller learning rate
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        # Train Model with more epochs
        model.fit(
            train_generator,
            epochs=1,  # Increased epochs for better learning
            validation_data=validation_generator
        )

        # Save Model and Class Indices
        model_filename = f"{dataset_folder_name}_{selected_model}.h5"
        model_save_path = os.path.join(MODEL_FOLDER, model_filename)
        model.save(model_save_path)

        class_indices = train_generator.class_indices
        with open(os.path.join(MODEL_FOLDER, f"{dataset_folder_name}_{selected_model}_class_indices.json"), 'w') as f:
            json.dump(class_indices, f)

        # Respond with success message
        return jsonify({"message": f"Model {selected_model} trained and saved successfully!"}), 200

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({"error": f"Error during training: {str(e)}"}), 500





@app.route('/test-model/<dataset_folder_name>/<model_name>', methods=['POST'])
def test_model(dataset_folder_name, model_name):
    """Allow the user to upload a test image and get a prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400
    if file and allowed_image(file.filename):
        filename = secure_filename(file.filename)
        test_image_path = os.path.join(TEST_UPLOAD_FOLDER, filename)
        file.save(test_image_path)
        print(f"Test image saved to {test_image_path}")

        # Load the Model
        model_filename = f"{dataset_folder_name}_{model_name}.h5"
        model_path = os.path.join(MODEL_FOLDER, model_filename)
        if not os.path.exists(model_path):
            print("Error: Model file not found.")
            return jsonify({'error': 'Model file not found. Please train the model first.'}), 400
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model {model_filename} loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return jsonify({'error': f'Error loading model: {str(e)}'}), 500

        # Load class indices
        class_indices_path = os.path.join(MODEL_FOLDER, f"{dataset_folder_name}_{model_name}_class_indices.json")
        if not os.path.exists(class_indices_path):
            print("Error: Class indices file not found.")
            return jsonify({'error': 'Class indices file not found.'}), 400
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
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500


        try:
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)
            probability = prediction[0][predicted_class_index]
            confidence_threshold = 0.7  

            if probability < confidence_threshold:
                result = "Unknown"
            else:
                result = class_labels.get(predicted_class_index, "Unknown")
            
            print(f"Prediction Probability: {probability:.4f}, Predicted Class: {result}")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

        # Optionally, remove the uploaded test image to save space
        # os.remove(test_image_path)

        # Return the prediction result as JSON
        return jsonify({
            'result': result,
            'probability': float(probability)
        }), 200
    else:
        return jsonify({'error': 'Allowed image types are png, jpg, jpeg.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
