// ./src/pages/Trained_model.js

import React, { useState, useEffect } from "react";
import { Camera, CheckCircle, XCircle, Loader, Brain } from "lucide-react";
import axios from "axios";

const Trained_model = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const [diseaseCategories, setDiseaseCategories] = useState({});
  const [selectedCategory, setSelectedCategory] = useState("");
  const [existingClasses, setExistingClasses] = useState([]);
  const [modelExists, setModelExists] = useState(false);

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
      // Check if model exists and fetch existing classes
      axios
        .post("http://127.0.0.1:5000/check-existing-model", {
          disease_category: selectedCategory,
        })
        .then((response) => {
          setModelExists(true);
          setExistingClasses(response.data.classes);
          setError(null);
        })
        .catch((error) => {
          if (error.response && error.response.status === 404) {
            setModelExists(false);
            setExistingClasses([]);
            setError("No trained model exists for the selected category.");
          } else {
            console.error("Error checking model availability:", error);
            setError("Failed to check model availability");
          }
        });
    } else {
      setModelExists(false);
      setExistingClasses([]);
    }
  }, [selectedCategory]);

  const handleFileSelect = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;
    setFile(uploadedFile);
    setFileName(uploadedFile.name);
    setError(null);
    setPrediction(null);
    setProbability(null);
  };

  const handlePrediction = async () => {
    if (!file || !selectedCategory) {
      setError("Please select an image and disease category first");
      return;
    }

    if (!modelExists) {
      setError("No trained model available for the selected category.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);
    setProbability(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("disease_category", selectedCategory);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/pretrainedmodels",
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

  return (
    <div className="min-h-screen bg-gray-50 py-16 px-4">
      {/* Navigation */}
      <nav className="bg-white shadow-md fixed top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center space-x-3">
              <Brain className="h-8 w-8 text-emerald-500" />
              <span className="text-xl font-bold text-gray-800">
                Medical Imaging
              </span>
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
          <h2 className="text-3xl font-bold text-gray-800 mb-6">
            Medical Image Classification
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
          {modelExists && existingClasses.length > 0 && (
            <div className="mb-6">
              <p className="text-gray-700 font-medium">Available Classes:</p>
              <ul className="list-disc list-inside text-gray-600 mt-2">
                {existingClasses.map((cls) => (
                  <li key={cls}>{cls}</li>
                ))}
              </ul>
            </div>
          )}

          {!modelExists && selectedCategory && (
            <div className="mb-6 bg-yellow-50 border border-yellow-200 rounded-xl p-4">
              <div className="flex items-center space-x-3">
                <XCircle className="text-yellow-500 flex-shrink-0" />
                <p className="text-yellow-600">
                  No trained model exists for the selected category. Please
                  train a model first.
                </p>
              </div>
            </div>
          )}

          {/* Upload Requirements */}
          <div className="mb-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
            <div className="flex items-start space-x-3">
              <Camera className="h-6 w-6 text-blue-500 mt-1 flex-shrink-0" />
              <div>
                <h3 className="text-lg font-semibold text-blue-700 mb-2">
                  Upload Requirements
                </h3>
                <ul className="space-y-2 text-blue-600 text-sm">
                  <li>
                    • Upload a clear image related to the selected disease
                    category and class
                  </li>
                  <li>• Ensure proper lighting and focus</li>
                  <li>• Supported formats: JPG, JPEG, PNG</li>
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
                <span className="text-gray-500">
                  Click to upload or drag and drop
                </span>
              )}
              <span className="text-sm text-gray-400 mt-2">
                JPG or PNG up to 5MB
              </span>
            </label>
          </div>

          {/* Predict Button */}
          <button
            onClick={handlePrediction}
            disabled={isLoading || !file || !selectedCategory || !modelExists}
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
              "Analyze Image"
            )}
          </button>

          {/* Prediction Result */}
          {prediction && (
            <div className="mt-6 bg-gray-50 rounded-xl p-6">
              <div className="flex items-center space-x-3 text-emerald-600">
                <CheckCircle size={24} />
                <div>
                  <span className="text-lg font-medium">
                    Prediction Result:
                  </span>
                  <p className="text-xl font-bold mt-1">{prediction}</p>
                  {probability && (
                    <p className="text-gray-600">
                      Confidence: {(probability * 100).toFixed(2)}%
                    </p>
                  )}
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
