import React, { useState } from 'react';
import { Camera, CheckCircle, XCircle, Loader, Brain, Eye } from 'lucide-react';
import axios from 'axios';

const Trained_model = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);

  const conditions = [
    { id: '0', name: 'AMD', description: 'Age-related Macular Degeneration' },
    { id: '1', name: 'Cataract', description: 'Clouding of the eye\'s natural lens' },
    { id: '2', name: 'Diabetes', description: 'Diabetic Retinopathy' },
    { id: '3', name: 'Glaucoma', description: 'Damage to the optic nerve' },
    { id: '4', name: 'Hypertension', description: 'High blood pressure affecting the eyes' },
    { id: '5', name: 'Myopia', description: 'Nearsightedness' },
    { id: '6', name: 'Normal', description: 'No detected eye conditions' },
    { id: '7', name: 'Other', description: 'Other images' }
  ];

  const handleFileSelect = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;
    setFile(uploadedFile);
    setFileName(uploadedFile.name);
    setError(null);
    setPrediction(null);
  };

  const handlePrediction = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPrediction(response.data.predicted_class);
    } catch (error) {
      setError(`Prediction failed: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-16 px-4">
      {/* Navigation */}
      <nav className="bg-white shadow-md fixed top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center space-x-3">
              <Brain className="h-8 w-8 text-emerald-500" />
              <span className="text-xl font-bold text-gray-800">Medical Imaging and Radiology</span>
            </div>
            
            <div className="hidden md:flex space-x-10">
              <button className="px-5 py-2 text-gray-600 hover:text-gray-800 focus:outline-none">
                Home
              </button>
              <button className="px-5 py-2 text-gray-600 hover:text-gray-800 focus:outline-none">
                About
              </button>
              <button className="px-5 py-2 text-gray-600 hover:text-gray-800 focus:outline-none">
                Help
              </button>
            </div>
            
            <div className="md:hidden">
              <button 
                onClick={() => setShowMobileMenu(!showMobileMenu)} 
                className="text-gray-600 hover:text-gray-800 focus:outline-none"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {showMobileMenu && (
          <div className="md:hidden px-4">
            <div className="px-2 pt-2 pb-3 space-y-1">
              <button className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-800 focus:outline-none">
                Home
              </button>
              <button className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-800 focus:outline-none">
                About
              </button>
              <button className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-800 focus:outline-none">
                Help
              </button>
            </div>
          </div>
        )}
      </nav>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto mt-16">
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-6">Eye Disease Classification</h2>
          
          {/* Detectable Conditions */}
          <div className="mb-8 bg-emerald-50 border border-emerald-200 rounded-xl p-6">
            <div className="flex items-start space-x-3">
              <Eye className="h-6 w-6 text-emerald-500 mt-1 flex-shrink-0" />
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-emerald-700 mb-3">
                  Detectable Eye Conditions
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {conditions.map((condition) => (
                    <div 
                      key={condition.id}
                      className="bg-white rounded-lg p-3 border border-emerald-100"
                    >
                      <span className="font-medium text-emerald-600">{condition.name}</span>
                      <p className="text-sm text-gray-600 mt-1">{condition.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Upload Requirements */}
          <div className="mb-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
            <div className="flex items-start space-x-3">
              <Camera className="h-6 w-6 text-blue-500 mt-1 flex-shrink-0" />
              <div>
                <h3 className="text-lg font-semibold text-blue-700 mb-2">
                  Upload Requirements
                </h3>
                <ul className="space-y-2 text-blue-600 text-sm">
                  <li>• Upload a clear image of the eye</li>
                  <li>• Ensure proper lighting and focus</li>
                  <li>• Supported formats: JPG, JPEG</li>
                  <li>• Maximum file size: 5MB</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Upload Section */}
          <div className="border-2 border-dashed border-emerald-200 rounded-xl p-8 text-center mb-6">
            <input
              type="file"
              onChange={handleFileSelect}
              className="hidden"
              id="image-upload"
              accept="image/*"
            />
            <label
              htmlFor="image-upload"
              className="flex flex-col items-center cursor-pointer"
            >
              <Camera className="w-12 h-12 text-emerald-500 mb-4" />
              {fileName ? (
                <span className="text-emerald-500 font-medium">{fileName}</span>
              ) : (
                <span className="text-gray-500">Click to upload or drag and drop</span>
              )}
              <span className="text-sm text-gray-400 mt-2">
                JPG or PNG up to 5MB
              </span>
            </label>
          </div>

          {/* Predict Button */}
          <button
            onClick={handlePrediction}
            disabled={isLoading || !file}
            className="w-full py-4 px-6 bg-emerald-500 text-white rounded-xl font-medium
              hover:bg-emerald-600 transition-colors duration-300 disabled:opacity-50 
              flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader className="animate-spin" />
                <span>Analyzing Image...</span>
              </>
            ) : (
              'Analyze Image'
            )}
          </button>

          {/* Prediction Result */}
          {prediction && (
            <div className="mt-6 bg-gray-50 rounded-xl p-6">
              <div className="flex items-center space-x-3 text-emerald-600">
                <CheckCircle size={24} />
                <div>
                  <span className="text-lg font-medium">Prediction Result:</span>
                  <p className="text-xl font-bold mt-1">{prediction}</p>
                </div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4 flex items-center space-x-3">
              <XCircle className="text-red-500 flex-shrink-0" />
              <p className="text-red-600">{error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Trained_model;