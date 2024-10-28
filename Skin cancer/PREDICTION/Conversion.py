from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS
from PIL import Image
import time

app = Flask(__name__)
CORS(app)

# Load your saved Keras model
model = tf.keras.models.load_model("my_model5.keras")
class_names = ['melanoma','nevus', 'pigmented benign keratosis']
# Preprocessing function
def preprocess_image(image_path):
    img_height = 180  # Your model's expected height
    img_width = 180   # Your model's expected width
    
    # Load and resize the image
    img = Image.open(image_path)
    img = img.resize((img_width, img_height))

    # Convert the image to array
    img_array = np.array(img)
    
    # Expand dimensions to match the input shape required by the model (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Rescale the pixel values (to [0, 1] range)
    img_array = img_array / 255.0  # Assuming model was trained with Rescaling(1.0/255)

    return img_array

@app.route("/classify", methods=["POST"])
def classify_image():
    # Get the uploaded image from the request
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    # Save the uploaded image temporarily
    filename = f"uploads/image_{time.time()}.jpg"
    image.save(filename)

    # Preprocess the image
    preprocessed_image = preprocess_image(filename)

    if preprocessed_image is not None:
        # Make predictions using the model
        prediction = model.predict(preprocessed_image)

        # Convert prediction to class label
        predicted_class = int(np.argmax(prediction))  # Get index of highest probability class
        # Get the disease name by mapping the predicted class index to the class name
        predicted_class_name = class_names[predicted_class]
        return jsonify({
            "class": predicted_class,
            "prediction": prediction.tolist(),
            "status": "200"
        })
    else:
        return jsonify({"error": "Error preprocessing image"}), 400

if __name__ == "__main__":
    app.run(debug=True)  # Set debug=True for automatic reloading during development
