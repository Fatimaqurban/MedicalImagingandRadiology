import React, { useState } from 'react';
import axios from 'axios';
import { Camera, CheckCircle, XCircle } from 'lucide-react';

const MachineLearningApp = () => {
  const [file, setFile] = useState(null);
  const [classDistribution, setClassDistribution] = useState(null);
  const [extractFolderName, setExtractFolderName] = useState('');
  const [datasetFolderName, setDatasetFolderName] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [error, setError] = useState(null);
  const [currentSlide, setCurrentSlide] = useState(0);

  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setClassDistribution(response.data.class_distribution);
      setExtractFolderName(response.data.extract_folder_name);
      setDatasetFolderName(response.data.dataset_folder_name);
      console.log(extractFolderName)
      console.log(datasetFolderName)
      setCurrentSlide(1);
    } catch (error) {
      console.error(error);
      setError(`Error uploading file: ${error.response?.data?.error || error.message}`);
    }
  };

  const handleModelSelection = (e) => {
    setSelectedModel(e.target.value);
  };

  const handleTrainModel = async () => {
    if (!selectedModel) {
      setError('Please select a model.');
      return;
    }
    try {
      await axios.post(`http://127.0.0.1:5000/select-model/${datasetFolderName}`, { model: selectedModel }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      setCurrentSlide(2);
    } catch (error) {
      console.error(error);
      setError(`Error training model: ${error.response?.data?.error || error.message}`);
    }
  };

  const handlePrediction = async () => {
    const testFile = file; // Use the uploaded file
    if (!testFile) {
      setError('Please upload a test image.');
      return;
    }
    const formData = new FormData();
    formData.append('file', testFile);

    try {
      const response = await axios.post(`http://127.0.0.1:5000/test-model/${datasetFolderName}/${selectedModel}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setPrediction(response.data.result);
      setProbability(response.data.probability);
    } catch (error) {
      console.error(error);
      setError(`Error making prediction: ${error.response?.data?.error || error.message}`);
    }
  };

  const availableModels = ['MobileNetV2', 'ResNet50', 'InceptionV3'];

  return (
    <div className="container mx-auto py-12 px-4 grid grid-cols-1 md:grid-cols-3 gap-8">
      <div className={`transition duration-500 ease-in-out transform ${currentSlide === 0 ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-10 pointer-events-none'}`}>
        <div className="bg-gradient-to-b from-blue-100 to-blue-300 rounded-xl shadow-xl p-8 space-y-6">
          <h2 className="text-3xl font-semibold text-gray-800">Upload Dataset</h2>
          <p className="text-gray-600">Structure your dataset like this:</p>
          <pre className="bg-gray-200 p-4 rounded-lg text-sm font-mono whitespace-pre-wrap leading-relaxed">
            DatasetFolder/
                class1/
                    img1.jpg
                class2/
                    img2.jpg
          </pre>
          <label className="flex items-center justify-center bg-white border border-blue-200 rounded-lg cursor-pointer py-4 shadow hover:shadow-md transition duration-300">
            <Camera className="mr-2 text-blue-500" size={24} />
            <span className="text-blue-600 font-medium">Upload File</span>
            <input type="file" className="hidden" onChange={handleFileUpload} />
          </label>
          {error && (
            <div className="flex items-center space-x-2 text-red-500 mt-4">
              <XCircle size={24} />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      <div className={`transition duration-500 ease-in-out transform ${currentSlide === 1 ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-10 pointer-events-none'}`}>
        <div className="bg-gradient-to-b from-green-100 to-green-300 rounded-xl shadow-xl p-8 space-y-6">
          <h2 className="text-3xl font-semibold text-gray-800">Select Model</h2>
          <div className="space-y-2">
            <label htmlFor="model" className="text-gray-700 font-medium">Model:</label>
            <select
              id="model"
              value={selectedModel}
              onChange={handleModelSelection}
              className="w-full py-2 px-3 rounded-lg border border-green-200 bg-white shadow-md focus:outline-none focus:ring-2 focus:ring-green-300"
            >
              <option value="">Select a model</option>
              {availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={handleTrainModel}
            className="w-full py-2 px-4 bg-green-500 hover:bg-green-600 text-white rounded-lg shadow-md hover:shadow-lg transition duration-300"
          >
            Train Model
          </button>
          {error && (
            <div className="flex items-center space-x-2 text-red-500 mt-4">
              <XCircle size={24} />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      <div className={`transition duration-500 ease-in-out transform ${currentSlide === 2 ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-10 pointer-events-none'}`}>
        <div className="bg-gradient-to-b from-purple-100 to-purple-300 rounded-xl shadow-xl p-8 space-y-6">
          <h2 className="text-3xl font-semibold text-gray-800">Test Model</h2>
          <label className="flex items-center justify-center bg-white border border-purple-200 rounded-lg cursor-pointer py-4 shadow hover:shadow-md transition duration-300">
            <Camera className="mr-2 text-purple-500" size={24} />
            <span className="text-purple-600 font-medium">Upload Test File</span>
            <input type="file" className="hidden" onChange={(e) => setFile(e.target.files[0])} />
          </label>
          <button
            onClick={handlePrediction}
            className="w-full py-2 px-4 bg-purple-500 hover:bg-purple-600 text-white rounded-lg shadow-md hover:shadow-lg transition duration-300"
          >
            Predict
          </button>
          {prediction && (
            <div className="flex flex-col items-start space-y-2 text-green-600 mt-4">
              <div className="flex items-center space-x-2">
                <CheckCircle size={24} />
                <span className="text-lg">Prediction: {prediction}</span>
              </div>
              <span className="text-lg">Probability: {probability ? probability.toFixed(4) : ''}</span>
            </div>
          )}
          {error && (
            <div className="flex items-center space-x-2 text-red-500 mt-4">
              <XCircle size={24} />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MachineLearningApp;
