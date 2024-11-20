import React, { useState } from 'react';
import axios from 'axios';
import { Camera, CheckCircle, XCircle, Loader, Upload, Brain, FileQuestion, ArrowRight, FolderTree, Home, Settings, Database, HelpCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const MachineLearningApp = () => {
  const [file, setFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [classDistribution, setClassDistribution] = useState(null);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const [extractFolderName, setExtractFolderName] = useState('');
  const [datasetFolderName, setDatasetFolderName] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [error, setError] = useState(null);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [testFileName, setTestFileName] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const navigate = useNavigate();

  const handleFileSelect = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;
    setFile(uploadedFile);
    setFileName(uploadedFile.name);
    setError(null);
  };

  const handleFileUpload = async () => {
    if (!file) {
      setError('Please select a dataset first');
      return;
    }
    
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setClassDistribution(response.data.class_distribution);
      setExtractFolderName(response.data.extract_folder_name);
      setDatasetFolderName(response.data.dataset_folder_name);
      setCurrentSlide(1);
    } catch (error) {
      setError(`Upload failed: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelSelection = (e) => {
    setSelectedModel(e.target.value);
    setError(null);
  };

  const handleTrainModel = async () => {
    if (!selectedModel) {
      setError('Please select a model first');
      return;
    }
    
    setIsLoading(true);
    setIsTraining(true);
    setError(null);

    try {
      await axios.post(
        `http://127.0.0.1:5000/select-model/${datasetFolderName}`,
        { model: selectedModel },
        { headers: { 'Content-Type': 'application/json' }}
      );
      setCurrentSlide(2);
    } catch (error) {
      setError(`Training failed: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsLoading(false);
      setIsTraining(false);
    }
  };

  const handlePrediction = async () => {
    if (!testFile) {
      setError('Please upload a test image');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setPrediction(null);
    setProbability(null);

    const formData = new FormData();
    formData.append('file', testFile);

    try {
      const response = await axios.post(
        `http://127.0.0.1:5000/test-model/${datasetFolderName}/${selectedModel}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }}
      );
      setPrediction(response.data.result);
      setProbability(response.data.probability);
    } catch (error) {
      setError(`Prediction failed: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const availableModels = ['MobileNetV2', 'ResNet50', 'InceptionV3'];

  const steps = [
    { title: 'Upload Dataset', icon: Upload, color: 'emerald' },
    { title: 'Select Model', icon: Brain, color: 'emerald' },
    { title: 'Test Model', icon: FileQuestion, color: 'emerald' }
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-16 px-4">
      
      <nav className="bg-white shadow-md fixed top-0 left-0 right-0 z-50">
  <div className="max-w-7xl mx-auto px-6"> {/* Add padding here for spacing */}
    <div className="flex justify-between items-center h-20"> {/* Adjust height for better alignment */}
      {/* Logo */}
      <div className="flex items-center space-x-3">
        <Brain className="h-8 w-8 text-emerald-500" />
        <span className="text-xl font-bold text-gray-800">Medical Imaging</span>
      </div>
      
      {/* Main Nav */}
      <div className="hidden md:flex space-x-10"> {/* Increase space between nav items */}
        <button className="px-5 py-2 text-gray-600 hover:text-gray-800 focus:outline-none">
          Home
        </button>
        <button 
          onClick={() => navigate('/pretrainedModels')} 
          className="px-5 py-2 text-gray-600 hover:text-gray-800 focus:outline-none"
        >
          Trained Models
        </button>
        <button className="px-5 py-2 text-gray-600 hover:text-gray-800 focus:outline-none">
          Settings
        </button>
      </div>
      
      {/* Mobile Menu Button */}
      <div className="md:hidden">
        <button onClick={() => setShowMobileMenu(!showMobileMenu)} className="text-gray-600 hover:text-gray-800 focus:outline-none">
          {/* Menu Icon */}
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16"></path>
          </svg>
        </button>
      </div>
    </div>
  </div>

  {/* Mobile Menu */}
  {showMobileMenu && (
    <div className="md:hidden px-4"> {/* Add padding for mobile items */}
      <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
        <button className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-800 focus:outline-none">
          Home
        </button>
        <button className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-800 focus:outline-none">
          Trained Models
        </button>
        <button className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-800 focus:outline-none">
          Settings
        </button>
      </div>
    </div>
  )}
</nav>



      {/* Progress Bar */}
      <div className="max-w-4xl mx-auto mb-16 mt-8">
        <div className="flex justify-between">
          {steps.map((step, index) => (
            <div key={index} className="flex flex-col items-center w-1/3">
              <div className={`flex items-center justify-center w-12 h-12 rounded-full mb-4 transition-colors duration-300
                ${currentSlide >= index ? 'bg-emerald-500 text-white' : 'bg-gray-200 text-gray-400'}`}>
                <step.icon size={24} />
              </div>
              <span className={`text-sm font-medium transition-colors duration-300
                ${currentSlide >= index ? 'text-emerald-500' : 'text-gray-400'}`}>
                {step.title}
              </span>
            </div>
          ))}
        </div>
        <div className="relative mt-4">
          <div className="absolute top-0 left-0 w-full h-1 bg-gray-200 rounded">
            <div 
              className="absolute top-0 left-0 h-full bg-emerald-500 rounded transition-all duration-300"
              style={{ width: `${(currentSlide / 2) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto">
        <div className="relative min-h-[400px]">
          {/* Upload Dataset Section */}
          <div className={`transition-all duration-500 ease-in-out transform absolute w-full
            ${currentSlide === 0 ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-full'}`}>
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-6">Upload Your Dataset</h2>
              
              {/* Dataset Format Information */}
              <div className="mb-6 bg-blue-50 border border-blue-200 rounded-xl p-6">
                <div className="flex items-start space-x-3">
                  <FolderTree className="h-6 w-6 text-blue-500 mt-1 flex-shrink-0" />
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-blue-700 mb-2">
                      Required Dataset Format
                    </h3>
                    <div className="space-y-3 text-blue-600">
                      <p>Your dataset folder should follow this structure:</p>
                      <div className="bg-blue-100 rounded-lg p-4 font-mono text-sm whitespace-pre">
                        DatasetFolderName/<br/>
                        ├── class1/<br/>
                        │   ├── img1.jpg<br/>
                        │   ├── img2.jpg<br/>
                        │   └── ...<br/>
                        ├── class2/<br/>
                        │   ├── img1.jpg<br/>
                        │   ├── img2.jpg<br/>
                        │   └── ...<br/>
                        └── ...<br/>
                      </div>
                      <div className="text-sm space-y-1">
                        <p>• Each class should be in a separate folder</p>
                        <p>• Images should be in JPG/JPEG format</p>
                        <p>• Ensure consistent image dimensions within the dataset</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-2 border-dashed border-emerald-200 rounded-xl p-8 text-center mb-6">
                <input
                  type="file"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="dataset-upload"
                />
                <label
                  htmlFor="dataset-upload"
                  className="flex flex-col items-center cursor-pointer"
                >
                  <Camera className="w-12 h-12 text-emerald-500 mb-4" />
                  {fileName ? (
                    <span className="text-emerald-500 font-medium">{fileName}</span>
                  ) : (
                    <span className="text-gray-500">Drop your dataset here or click to browse</span>
                  )}
                </label>
              </div>
              <button
                onClick={handleFileUpload}
                disabled={isLoading || !file}
                className="w-full py-4 px-6 bg-emerald-500 text-white rounded-xl font-medium
                  hover:bg-emerald-600 transition-colors duration-300 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader className="animate-spin" />
                    <span>Uploading Dataset...</span>
                  </>
                ) : (
                  <>
                    <span>Upload Dataset</span>
                    <ArrowRight size={20} />
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Model Selection Section */}
          <div className={`transition-all duration-500 ease-in-out transform absolute w-full
            ${currentSlide === 1 ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-full'}`}>
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-6">Choose Your Model</h2>
              {isTraining ? (
                <div className="text-center py-8">
                  <Loader className="w-12 h-12 text-emerald-500 animate-spin mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">Training in Progress</h3>
                  <p className="text-gray-600">Please wait while we train the model with your dataset</p>
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="relative">
                    <select
                      value={selectedModel}
                      onChange={handleModelSelection}
                      className="w-full p-4 bg-gray-50 border border-gray-200 rounded-xl appearance-none focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                      <option value="">Select a model</option>
                      {availableModels.map(model => (
                        <option key={model} value={model}>{model}</option>
                      ))}
                    </select>
                  </div>
                  <button
                    onClick={handleTrainModel}
                    disabled={isLoading || !selectedModel}
                    className="w-full py-4 px-6 bg-emerald-500 text-white rounded-xl font-medium
                      hover:bg-emerald-600 transition-colors duration-300 disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <Loader className="animate-spin" />
                        <span>Training Model...</span>
                      </>
                    ) : (
                      <>
                        <span>Train Model</span>
                        <ArrowRight size={20} />
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Test Model Section */}
          <div className={`transition-all duration-500 ease-in-out transform absolute w-full
            ${currentSlide === 2 ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-full'}`}>
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-6">Test Your Model</h2>
              
              {/* Dataset and Model Info */}
              <div className="bg-gray-50 rounded-xl p-4 mb-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">Dataset</p>
                    <p className="font-medium text-gray-800">{fileName}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Selected Model</p>
                    <p className="font-medium text-gray-800">{selectedModel}</p>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="border-2 border-dashed border-emerald-200 rounded-xl p-8 text-center">
                  <input
                    type="file"
                    onChange={(e) => {
                      const file = e.target.files[0];
                      if (file) {
                        setTestFile(file);
                        setTestFileName(file.name);
                      }
                    }}
                    className="hidden"
                    id="test-upload"
                  />
                  <label
                    htmlFor="test-upload"
                    className="flex flex-col items-center cursor-pointer"
                  >
                    <Camera className="w-12 h-12 text-emerald-500 mb-4" />
                    {testFileName ? (
                      <span className="text-emerald-500 font-medium">{testFileName}</span>
                    ) : (
                      <span className="text-gray-500">Upload test image</span>
                    )}
                  </label>
                </div>

                <button
                  onClick={handlePrediction}
                  disabled={isLoading || !testFile}
                  className="w-full py-4 px-6 bg-emerald-500 text-white rounded-xl font-medium
                    hover:bg-emerald-600 transition-colors duration-300 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <Loader className="animate-spin" />
                      <span>Making Prediction...</span>
                    </>
                  ) : (
                    'Make Prediction'
                  )}
                </button>

                {prediction && (
                  <div className="bg-gray-50 rounded-xl p-6 space-y-4">
                    <div className="flex items-center space-x-3 text-emerald-600">
                      <CheckCircle size={24} />
                      <span className="text-lg font-medium">
                        {prediction}
                      </span>
                    </div>
                    {probability && (
                      <div className="ml-9">
                        <div className="text-gray-600">
                          Confidence: {(probability * 100).toFixed(2)}%
                        </div>
                        <div className="mt-2 h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                            style={{ width: `${probability * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}

              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4 flex items-center space-x-3">
            <XCircle className="text-red-500 flex-shrink-0" />
            <p className="text-red-600">{error}</p>
          </div>
        )}
      </div>
    
    </div>
  );
};

export default MachineLearningApp;