// ./src/pages/ML_UI.js

import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Camera,
  CheckCircle,
  XCircle,
  Loader,
  Upload,
  Brain,
  FileQuestion,
  ArrowRight,
  FolderTree,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

const MachineLearningApp = () => {
  // State variables
  const [file, setFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [classDistribution, setClassDistribution] = useState(null);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const [extractFolderName, setExtractFolderName] = useState("");
  const [datasetFolderName, setDatasetFolderName] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState("");
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [testFileName, setTestFileName] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [diseaseCategories, setDiseaseCategories] = useState({});
  const [selectedCategory, setSelectedCategory] = useState("");
  const [existingClasses, setExistingClasses] = useState([]);
  const [newClasses, setNewClasses] = useState([]);
  const [updatedClasses, setUpdatedClasses] = useState([]);
  const [retrainingNeeded, setRetrainingNeeded] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    // Fetch disease categories from the backend
    axios
      .get("http://127.0.0.1:5000/get-disease-categories")
      .then((response) => {
        setDiseaseCategories(response.data);
      })
      .catch((error) => {
        console.error("Error fetching disease categories:", error);
        setError("Failed to fetch disease categories");
      });
  }, []);

  useEffect(() => {
    if (selectedCategory) {
      // Fetch existing classes for the selected disease category
      axios
        .post("http://127.0.0.1:5000/get-model-classes", {
          disease_category: selectedCategory,
        })
        .then((response) => {
          setExistingClasses(response.data.classes || []);
        })
        .catch((error) => {
          console.error("Error fetching existing classes:", error);
          setError("Failed to fetch existing classes");
        });
    } else {
      setExistingClasses([]);
    }
  }, [selectedCategory]);

  const handleFileSelect = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;
    setFile(uploadedFile);
    setFileName(uploadedFile.name);
    setError(null);
  };

  const handleFileUpload = async () => {
    if (!file || !selectedCategory) {
      setError("Please select a dataset and disease category first");
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuccessMessage("");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("disease_category", selectedCategory);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/upload",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      const data = response.data;

      if (data.retraining_needed === false) {
        // No retraining needed, inform the user and proceed to testing
        setRetrainingNeeded(false);
        setExistingClasses(data.existing_classes || []);
        setUpdatedClasses(data.existing_classes || []);
        setNewClasses([]);
        setCurrentSlide(2); // Skip to Test Model slide
        setSuccessMessage(
          "Your desired features are already available in the model and ready to be tested. No need to retrain."
        );
      } else {
        // Retraining is needed
        setRetrainingNeeded(true);
        setClassDistribution(data.class_distribution);
        setExtractFolderName(data.extract_folder_name);
        setDatasetFolderName(data.dataset_folder_name);
        setUpdatedClasses(data.updated_classes || []);
        setNewClasses(
          (data.updated_classes || []).filter(
            (cls) => !existingClasses.includes(cls)
          )
        );
        setCurrentSlide(1);
      }
    } catch (error) {
      setError(
        `Upload failed: ${error.response?.data?.error || error.message}`
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelSelection = (e) => {
    setSelectedModel(e.target.value);
    setError(null);
  };

  const handleTrainModel = async () => {
    if (!selectedModel || !selectedCategory) {
      setError("Please select a model and disease category first");
      return;
    }

    setIsLoading(true);
    setIsTraining(true);
    setError(null);

    try {
      await axios.post(
        `http://127.0.0.1:5000/select-model/${datasetFolderName}`,
        {
          model: selectedModel,
          disease_category: selectedCategory,
          updated_classes: updatedClasses,
        },
        { headers: { "Content-Type": "application/json" } }
      );
      setCurrentSlide(2);
    } catch (error) {
      setError(
        `Training failed: ${error.response?.data?.error || error.message}`
      );
    } finally {
      setIsLoading(false);
      setIsTraining(false);
    }
  };

  const handlePrediction = async () => {
    if (!testFile || !selectedCategory) {
      setError("Please upload a test image and select a disease category");
      return;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);
    setProbability(null);

    const formData = new FormData();
    formData.append("file", testFile);
    formData.append("disease_category", selectedCategory);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setPrediction(response.data.predicted_class);
      // If confidence is provided, set it
      if (response.data.confidence) {
        setProbability(response.data.confidence);
      }
    } catch (error) {
      setError(
        `Prediction failed: ${error.response?.data?.error || error.message}`
      );
    } finally {
      setIsLoading(false);
    }
  };

  const availableModels = ["MobileNetV2", "ResNet50", "InceptionV3"];

  const steps = [
    { title: "Upload Dataset", icon: Upload, color: "emerald" },
    { title: "Select Model", icon: Brain, color: "emerald" },
    { title: "Test Model", icon: FileQuestion, color: "emerald" },
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-16 px-4">
      {/* Navigation */}
      <nav className="bg-white shadow-md fixed top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex justify-between items-center h-20">
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <Brain className="h-8 w-8 text-emerald-500" />
              <span className="text-xl font-bold text-gray-800">
                Medical Imaging
              </span>
            </div>

            {/* Main Nav */}
            <div className="hidden md:flex space-x-10">
              <button className="px-5 py-2 text-gray-600 hover:text-gray-800 focus:outline-none">
                Home
              </button>
              <button
                onClick={() => navigate("/pretrainedModels")}
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
              <button
                onClick={() => setShowMobileMenu(!showMobileMenu)}
                className="text-gray-600 hover:text-gray-800 focus:outline-none"
              >
                {/* Menu Icon */}
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        {showMobileMenu && (
          <div className="md:hidden px-4">
            <div className="px-2 pt-2 pb-3 space-y-1">
              <button className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-800 focus:outline-none">
                Home
              </button>
              <button
                onClick={() => navigate("/pretrainedModels")}
                className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-gray-800 focus:outline-none"
              >
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
              <div
                className={`flex items-center justify-center w-12 h-12 rounded-full mb-4 transition-colors duration-300
                    ${
                      currentSlide >= index
                        ? "bg-emerald-500 text-white"
                        : "bg-gray-200 text-gray-400"
                    }`}
              >
                <step.icon size={24} />
              </div>
              <span
                className={`text-sm font-medium transition-colors duration-300
                    ${
                      currentSlide >= index
                        ? "text-emerald-500"
                        : "text-gray-400"
                    }`}
              >
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
        <div className="relative min-h-[500px]">
          {/* Upload Dataset Section */}
          <div
            className={`transition-all duration-500 ease-in-out transform absolute w-full
                ${
                  currentSlide === 0
                    ? "opacity-100 translate-x-0"
                    : "opacity-0 -translate-x-full"
                }`}
          >
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-6">
                Upload Your Dataset
              </h2>

              {/* Disease Category Selection */}
              <div className="mb-6">
                <label className="block text-gray-700 text-sm font-bold mb-2">
                  Select Disease Category:
                </label>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="w-full p-4 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500"
                >
                  <option value="">Select a category</option>
                  {Object.keys(diseaseCategories).map((category) => (
                    <option key={category} value={category}>
                      {category}
                    </option>
                  ))}
                </select>
              </div>

              {/* Existing Classes */}
              {existingClasses.length > 0 && (
                <div className="mb-6">
                  <p className="text-gray-700 font-medium">Existing Classes:</p>
                  <ul className="list-disc list-inside text-gray-600 mt-2">
                    {existingClasses.map((cls) => (
                      <li key={cls}>{cls}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Instruction */}
              <div className="mb-6">
                <p className="text-gray-700">
                  If your desired class is not listed above, please upload a
                  dataset for the new class.
                </p>
              </div>

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
                        DatasetFolderName/
                        <br />
                        ├── class1/
                        <br />
                        │ &nbsp;&nbsp;&nbsp;├── img1.jpg
                        <br />
                        │ &nbsp;&nbsp;&nbsp;├── img2.jpg
                        <br />
                        │ &nbsp;&nbsp;&nbsp;└── ...
                        <br />
                        ├── class2/
                        <br />
                        │ &nbsp;&nbsp;&nbsp;├── img1.jpg
                        <br />
                        │ &nbsp;&nbsp;&nbsp;├── img2.jpg
                        <br />
                        │ &nbsp;&nbsp;&nbsp;└── ...
                        <br />
                        └── ...
                        <br />
                      </div>
                      <div className="text-sm space-y-1">
                        <p>• Each class should be in a separate folder</p>
                        <p>• Images should be in JPG/JPEG/PNG format</p>
                        <p>• Ensure at least 10 images per class</p>
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
                  accept=".zip"
                />
                <label
                  htmlFor="dataset-upload"
                  className="flex flex-col items-center cursor-pointer"
                >
                  <Camera className="w-12 h-12 text-emerald-500 mb-4" />
                  {fileName ? (
                    <span className="text-emerald-500 font-medium">
                      {fileName}
                    </span>
                  ) : (
                    <span className="text-gray-500">
                      Drop your dataset here or click to browse
                    </span>
                  )}
                </label>
              </div>
              <button
                onClick={handleFileUpload}
                disabled={isLoading || !file || !selectedCategory}
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
          {retrainingNeeded && (
            <div
              className={`transition-all duration-500 ease-in-out transform absolute w-full
                ${
                  currentSlide === 1
                    ? "opacity-100 translate-x-0"
                    : "opacity-0 translate-x-full"
                }`}
            >
              <div className="bg-white rounded-2xl shadow-lg p-8">
                <h2 className="text-3xl font-bold text-gray-800 mb-6">
                  Choose Your Model
                </h2>
                {isTraining ? (
                  <div className="text-center py-8">
                    <Loader className="w-12 h-12 text-emerald-500 animate-spin mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">
                      Training in Progress
                    </h3>
                    <p className="text-gray-600">
                      Please wait while we train the model with your dataset
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Disease Category Display */}
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Disease Category</p>
                      <p className="font-medium text-gray-800">
                        {selectedCategory}
                      </p>
                    </div>

                    {/* Updated Classes */}
                    {updatedClasses && updatedClasses.length > 0 && (
                      <div className="mb-6">
                        <p className="text-gray-700 font-medium">
                          Updated Classes:
                        </p>
                        <ul className="list-disc list-inside text-gray-600 mt-2">
                          {updatedClasses.map((cls) => (
                            <li key={cls}>{cls}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* New Classes */}
                    {newClasses && newClasses.length > 0 && (
                      <div className="mb-6">
                        <p className="text-gray-700 font-medium">
                          New Classes Added:
                        </p>
                        <ul className="list-disc list-inside text-gray-600 mt-2">
                          {newClasses.map((cls) => (
                            <li key={cls}>{cls}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="relative">
                      <select
                        value={selectedModel}
                        onChange={handleModelSelection}
                        className="w-full p-4 bg-gray-50 border border-gray-200 rounded-xl appearance-none focus:outline-none focus:ring-2 focus:ring-emerald-500"
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
          )}

          {/* Test Model Section */}
          <div
            className={`transition-all duration-500 ease-in-out transform absolute w-full
                ${
                  currentSlide === 2
                    ? "opacity-100 translate-x-0"
                    : "opacity-0 translate-x-full"
                }`}
          >
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-6">
                Test Your Model
              </h2>

              {/* Success Message Display */}
              {successMessage && (
                <div
                  className={`mt-6 mb-6 bg-green-50 border border-green-200 rounded-xl p-4 flex items-center space-x-3`}
                >
                  <CheckCircle className="text-green-500 flex-shrink-0" />
                  <p className="text-green-600">{successMessage}</p>
                </div>
              )}

              {/* Dataset and Model Info */}
              <div className="bg-gray-50 rounded-xl p-4 mb-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">Disease Category</p>
                    <p className="font-medium text-gray-800">
                      {selectedCategory}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Selected Model</p>
                    <p className="font-medium text-gray-800">
                      {selectedModel || "Pre-trained Model"}
                    </p>
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
                    accept="image/*"
                  />
                  <label
                    htmlFor="test-upload"
                    className="flex flex-col items-center cursor-pointer"
                  >
                    <Camera className="w-12 h-12 text-emerald-500 mb-4" />
                    {testFileName ? (
                      <span className="text-emerald-500 font-medium">
                        {testFileName}
                      </span>
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
                    "Make Prediction"
                  )}
                </button>

                {prediction && (
                  <div className="bg-gray-50 rounded-xl p-6 space-y-4">
                    <div className="flex items-center space-x-3 text-emerald-600">
                      <CheckCircle size={24} />
                      <span className="text-lg font-medium">{prediction}</span>
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
