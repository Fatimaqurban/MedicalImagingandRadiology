import React, { useState, useEffect } from "react";
import { Camera, CheckCircle, XCircle, Loader, Brain, ChevronRight } from "lucide-react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const Trained_model = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [diseaseCategories, setDiseaseCategories] = useState({});
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedCategoryDetails, setSelectedCategoryDetails] = useState(null);
  const [existingClasses, setExistingClasses] = useState([]);
  const [modelExists, setModelExists] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // Fetch disease categories from the backend
    axios
      .get("http://127.0.0.1:5000/get-disease-categories")
      .then((response) => {
        // Transform the data to include descriptions
        const categoriesWithDetails = Object.entries(response.data).reduce((acc, [category, classes]) => {
          acc[category] = {
            description: getCategoryDescription(category),
            classes: classes.reduce((classAcc, className) => {
              classAcc[className] = {
                description: getClassDescription(category, className),
                symptoms: getSymptoms(category, className)
              };
              return classAcc;
            }, {})
          };
          return acc;
        }, {});
        setDiseaseCategories(categoriesWithDetails);
      })
      .catch((error) => {
        console.error("Error fetching disease categories:", error);
        setError("Failed to fetch disease categories");
      });
  }, []);

  useEffect(() => {
    if (selectedCategory) {
      // Fetch detailed category information and check model
      axios
        .post("http://127.0.0.1:5000/check-existing-model", {
          disease_category: selectedCategory,
        })
        .then((response) => {
          setModelExists(true);
          setExistingClasses(response.data.classes);
          // Fetch detailed category information
          setSelectedCategoryDetails(diseaseCategories[selectedCategory]);
          setError(null);
        })
        .catch((error) => {
          if (error.response && error.response.status === 404) {
            setModelExists(false);
            setExistingClasses([]);
            setSelectedCategoryDetails(diseaseCategories[selectedCategory]);
            setError("No trained model exists for the selected category.");
          } else {
            console.error("Error checking model availability:", error);
            setError("Failed to check model availability");
          }
        });
    } else {
      setModelExists(false);
      setExistingClasses([]);
      setSelectedCategoryDetails(null);
    }
  }, [selectedCategory, diseaseCategories]);

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

  // Helper functions to provide descriptions
  const getCategoryDescription = (category) => {
    const descriptions = {
      "Eye Disease": "Comprehensive analysis of various eye conditions including cataracts, diabetic retinopathy, and glaucoma.",
      "Lung Disease": "Detection of respiratory conditions including COVID-19, tuberculosis, and viral pneumonia.",
      "Brain Tumor": "Identification of different types of brain tumors through medical imaging.",
      "Skin Disease": "Analysis of skin conditions and potential malignancies."
    };
    return descriptions[category] || `Analysis of ${category} conditions`;
  };

  const getClassDescription = (category, className) => {
    const descriptions = {
      "Eye Disease": {
        "Normal": "Healthy eye with no detected abnormalities",
        "cataract": "Clouding of the eye's natural lens that affects vision",
        "diabetic_retinopathy": "Diabetes-related damage to blood vessels in the retina",
        "glaucoma": "Eye condition that damages the optic nerve, often due to high pressure"
      },
      "Lung Disease": {
        "Normal": "Healthy lung tissue with no detected abnormalities",
        "Corona_Virus_Disease": "COVID-19 infection affecting the respiratory system",
        "Tuberculosis": "Bacterial infection primarily affecting the lungs",
        "Viral_Pneumonia": "Viral infection causing inflammation of the air sacs in the lungs"
      },
      "Brain Tumor": {
        "No Tumor": "Normal brain tissue with no detected tumors",
        "glioma": "Tumor that starts in the glial cells of the brain",
        "meningioma": "Tumor that forms in the meninges, the brain's protective membranes",
        "pituitary": "Tumor that develops in the pituitary gland"
      },
      "Skin Disease": {
        "melanoma": "Serious form of skin cancer that develops in melanocytes",
        "nevus": "Common mole or birthmark (usually benign)",
        "pigmented benign keratosis": "Non-cancerous growth on the skin's surface"
      }
    };
    return descriptions[category]?.[className] || `${className} in ${category}`;
  };

  const getSymptoms = (category, className) => {
    const symptoms = {
      "Eye Disease": {
        "cataract": ["Blurred vision", "Light sensitivity", "Poor night vision", "Fading colors"],
        "diabetic_retinopathy": ["Blurred vision", "Dark spots", "Vision loss", "Eye pain"],
        "glaucoma": ["Vision loss", "Eye pain", "Headaches", "Rainbow halos around lights"]
      },
      // Add symptoms for other categories as needed
    };
    return symptoms[category]?.[className] || [];
  };

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-md fixed top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex justify-between items-center h-20">
            {/* Logo */}
            <div 
              onClick={() => navigate("/")} 
              className="flex items-center space-x-3 cursor-pointer hover:opacity-80 transition-opacity"
            >
              <Brain className="h-8 w-8 text-emerald-500" />
              <span className="text-xl font-bold text-gray-800">
              Health<span className="highlight">Care</span>
              </span>
            </div>

            {/* Main Nav */}
            <div className="hidden md:flex space-x-10">
              <button
                className="px-5 py-2 text-gray-600 hover:text-gray-800 focus:outline-none"
                onClick={() => navigate("/")}
              >
                Home
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

      {/* Sidebar */}
      <div className="w-72 mt-28 ml-8 rounded-2xl bg-white shadow-lg border-r border-gray-200 p-6 overflow-y-auto">
        <div className="flex items-center mb-8">
          <h2 className="text-xl font-bold text-gray-800">Categories</h2>
        </div>
        <nav>
          {Object.keys(diseaseCategories).map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`w-full text-left px-4 py-3 rounded-lg mb-2 flex justify-between items-center ${selectedCategory === category
                ? 'bg-emerald-50 text-emerald-600 font-semibold'
                : 'text-gray-600 hover:bg-gray-100'
                }`}
            >
              {category}
              <ChevronRight className="h-5 w-5 opacity-50" />
            </button>
          ))}
        </nav>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 mt-16 p-10">
        {!selectedCategory ? (
          // Welcome message when no category is selected
          <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto">
            <Brain className="w-16 h-16 text-emerald-500 mb-6" />
            <h1 className="text-3xl font-bold text-gray-800 mb-4">
              Use Our Trained Models
            </h1>
            <p className="text-gray-600 text-lg mb-8">
              Select a disease category from the sidebar to get started with AI-powered medical image analysis.
            </p>
            <div className="flex items-center text-emerald-600">
              <ChevronRight className="w-5 h-5 animate-bounce" />
              <span className="text-sm font-medium">Choose a category to begin</span>
            </div>
          </div>
        ) : (
          // Existing category details and upload sections
          <>
            {/* Category Details Section */}
            {selectedCategoryDetails && (
              <div className="bg-white rounded-xl shadow-md p-6 mb-6">
                <h3 className="text-2xl font-bold text-gray-800 mb-4">
                  {selectedCategory} Details
                </h3>
                <p className="text-gray-600 mb-4">
                  {selectedCategoryDetails.description}
                </p>

                {/* Existing Classes */}
                {modelExists && existingClasses.length > 0 && (
                  <div>
                    <h4 className="text-lg font-semibold text-gray-700 mb-3">
                      Available Classes:
                    </h4>
                    <div className="grid md:grid-cols-2 gap-4">
                      {existingClasses.map((cls) => (
                        <div
                          key={cls}
                          className="bg-gray-50 p-4 rounded-lg border border-gray-200"
                        >
                          <h5 className="font-medium text-gray-800 mb-2">{cls}</h5>
                          <p className="text-gray-600 text-sm">
                            {selectedCategoryDetails.classes[cls].description}
                          </p>
                          {selectedCategoryDetails.classes[cls].symptoms?.length > 0 && (
                            <div className="mt-2">
                              <p className="text-sm font-medium text-gray-700">Common Symptoms:</p>
                              <ul className="list-disc list-inside text-sm text-gray-600 mt-1">
                                {selectedCategoryDetails.classes[cls].symptoms.map((symptom, index) => (
                                  <li key={index}>{symptom}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}    

                {!modelExists && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mt-4">
                    <div className="flex items-center space-x-3">
                      <XCircle className="text-yellow-500 flex-shrink-0" />
                      <p className="text-yellow-600">
                        No trained model exists for this category. Please train a model first.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Image Upload and Prediction Section */}
            {selectedCategoryDetails && (
              <div className="bg-white rounded-2xl shadow-lg p-8">
                {/* Upload Requirements */}
                <div className="mb-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
                  <div className="flex items-start space-x-3">
                    <Camera className="h-6 w-6 text-blue-500 mt-1 flex-shrink-0" />
                    <div>
                      <h3 className="text-lg font-semibold text-blue-700 mb-2">
                        Upload Requirements
                      </h3>
                      <ul className="space-y-2 text-blue-600 text-sm">
                        <li>• Upload a clear image related to the selected disease category</li>
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
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default Trained_model;
