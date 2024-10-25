import os
import zipfile
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def validate_folder_structure(extracted_path):
    """
    Validate that the extracted zip has the required folder structure.
    Expected: A single dataset folder containing class subfolders.
    """
    items = os.listdir(extracted_path)
    if len(items) != 1:
        return False
    dataset_folder = os.path.join(extracted_path, items[0])
    if not os.path.isdir(dataset_folder):
        return False
    class_folders = os.listdir(dataset_folder)
    if len(class_folders) < 2:
        # At least two classes are needed for binary classification
        return False
    for folder in class_folders:
        folder_path = os.path.join(dataset_folder, folder)
        if not os.path.isdir(folder_path):
            return False
    return True

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

            # Extract the zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    extract_path = os.path.join(EXTRACT_FOLDER, filename.rsplit('.', 1)[0])
                    zip_ref.extractall(extract_path)
            except zipfile.BadZipFile:
                flash('Uploaded file is not a valid zip file.')
                return redirect(request.url)

            # Validate folder structure
            if not validate_folder_structure(extract_path):
                flash('Invalid folder structure. Ensure the zip contains a single dataset folder with class subfolders.')
                return redirect(request.url)

            flash('File successfully uploaded and extracted.')
            dataset_name = os.listdir(extract_path)[0]
            return redirect(url_for('select_model', dataset=dataset_name))
        else:
            flash('Allowed file type is zip.')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/select_model/<dataset>', methods=['GET', 'POST'])
def select_model(dataset):
    """Allow the user to select a model and initiate training."""
    if request.method == 'POST':
        selected_model = request.form.get('model')
        if selected_model not in AVAILABLE_MODELS:
            flash('Invalid model selected.')
            return redirect(request.url)
        
        # Path to dataset folder
        dataset_dir = os.path.join(EXTRACT_FOLDER, dataset)
        
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

        # Print Class Indices
        print("Class Indices:", train_generator.class_indices)

        # Calculate Class Weights
        train_labels = train_generator.classes
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = dict(zip(np.unique(train_labels), class_weights))
        print("Class Weights:", class_weights)

        # Build the Model
        base_model_class = AVAILABLE_MODELS[selected_model]
        base_model = base_model_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze base model layers
        for layer in base_model.layers[:-10]:  # Unfreeze last 10 layers
            layer.trainable = False
        for layer in base_model.layers[-10:]:
            layer.trainable = True

        # Compile the Model
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        # Save the Model Path
        model_filename = f"{dataset}_{selected_model}.h5"
        model_save_path = os.path.join(MODEL_FOLDER, model_filename)

        # Define Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(filepath=model_save_path, save_best_only=True)
        ]

        # Train the Model
        try:
            model.fit(
                train_generator,
                epochs=50,
                validation_data=validation_generator,
                class_weight=class_weights,
                callbacks=callbacks
            )
        except Exception as e:
            flash(f'Error during training: {str(e)}')
            return redirect(request.url)

        # Save the Model (Redundant due to ModelCheckpoint, but kept for compatibility)
        model.save(model_save_path)

        flash(f'Model {selected_model} trained and saved successfully!')
        return redirect(url_for('test_model', dataset=dataset, model_name=selected_model))

    return render_template('select_model.html', models=AVAILABLE_MODELS.keys(), dataset=dataset)

@app.route('/test_model/<dataset>/<model_name>', methods=['GET', 'POST'])
def test_model(dataset, model_name):
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

            # Load the Model
            model_filename = f"{dataset}_{model_name}.h5"
            model_path = os.path.join(MODEL_FOLDER, model_filename)
            if not os.path.exists(model_path):
                flash('Model file not found. Please train the model first.')
                return redirect(url_for('upload_file'))
            try:
                model = tf.keras.models.load_model(model_path)
            except Exception as e:
                flash(f'Error loading model: {str(e)}')
                return redirect(request.url)

            # Preprocess the Image
            try:
                img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                img_array /= 255.0
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)

            # Make Prediction
            try:
                prediction = model.predict(img_array)
                probability = prediction[0][0]
                print("Prediction Probability:", probability)
                disease_present = probability > 0.5  # Adjust threshold if necessary
                result = 'Disease Present' if disease_present else 'Disease Absent'
                flash(f'Prediction Probability: {probability:.4f}')
                flash(f'Prediction Result: {result}')
                return redirect(request.url)
            except Exception as e:
                flash(f'Error during prediction: {str(e)}')
                return redirect(request.url)
        else:
            flash('Allowed image types are png, jpg, jpeg.')
            return redirect(request.url)
    return render_template('test_model.html', dataset=dataset, model_name=model_name)

if __name__ == '__main__':
    app.run(debug=True)
