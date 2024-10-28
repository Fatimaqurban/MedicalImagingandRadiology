import React, { useState } from 'react';
import './App.css'; // Import your CSS file for styling

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictedDisease, setPredictedDisease] = useState('');
  const [error, setError] = useState('');
  const [diseaseName, setDiseaseName] = useState('');

  // Function to handle file input change
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setSelectedImage(file);
  };

  // Function to handle cancel button click
  const handleCancel = () => {
    setSelectedImage(null);
    setPredictedDisease('');
    setError('');
    setDiseaseName('');
  };

  // Function to handle image upload and prediction
  const handleUpload = async () => {
    if (!selectedImage) return;

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await fetch('http://127.0.0.1:5000/classify', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        setPredictedDisease(data.class);
        setDiseaseName(getDiseaseName(data.class));
      } else {
        setError('Failed to classify image');
      }
    } catch (error) {
      setError('Error sending image to backend');
    }
  };

  // Function to get disease name from predicted class
  const getDiseaseName = (predictedClass) => {
    switch (predictedClass) {
      case 0:
        return 'Melanoma';
      case 1:
        return 'Nevus';
      case 2:
        return 'Pigmented Benign Keratosis';
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Skin Cancer Detection</h1>
      </header>
      <div className="upload-container">
        <input id="file-upload" type="file" onChange={handleImageChange} style={{ display: 'none' }} />
        <label htmlFor="file-upload" className="upload-btn">Choose File</label>
        {selectedImage && (
          <div className="image-preview">
            <img src={URL.createObjectURL(selectedImage)} alt="Uploaded" />
            <button className="cancel-button" onClick={handleCancel}>Cancel</button>
            <button className="upload-button" onClick={handleUpload}>Upload & Predict</button>
          </div>
        )}
      </div>
      {diseaseName && (
        <div className="prediction-result">
          <h2>Predicted Disease:</h2>
          <p>{diseaseName}</p>
        </div>
      )}
      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}

export default App;