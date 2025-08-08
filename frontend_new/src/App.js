import React, { useState, useEffect } from 'react';
import { 
  Upload, User, LogIn, LogOut, Camera, Activity, AlertCircle, Heart, Shield, Star, 
  Clock, TrendingUp, Eye, Brain, Stethoscope, FolderOpen, X, FileText, Download,
  Moon, Sun, FileSpreadsheet
} from 'lucide-react';
import './App.css';

const App = () => {
  const [user, setUser] = useState(null);
  const [activeTab, setActiveTab] = useState('login');
  const [loginData, setLoginData] = useState({ email: '', password: '' });
  const [signupData, setSignupData] = useState({ name: '', email: '', password: '' });
  const [selectedImages, setSelectedImages] = useState([]);
  const [selectedExcel, setSelectedExcel] = useState(null);
  const [imagePreviews, setImagePreviews] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [processingIndex, setProcessingIndex] = useState(-1);
  const [history, setHistory] = useState([]);
  const [darkMode, setDarkMode] = useState(false);

  const API_BASE = 'http://localhost:5000/api';
  // const API_BASE = 'https://disease-prediction-backend-v7.onrender.com/api';

  // Load dark mode preference from localStorage
  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    setDarkMode(savedDarkMode);
    document.documentElement.classList.toggle('dark', savedDarkMode);
  }, []);

  // Toggle dark mode
  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', newDarkMode);
    document.documentElement.classList.toggle('dark', newDarkMode);
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(loginData)
      });

      const data = await response.json();
      if (response.ok) {
        setUser(data.user);
        setActiveTab('dashboard');
        loadHistory(data.user.id);
      } else {
        alert(data.error || 'Login failed');
      }
    } catch (error) {
      alert('Connection error');
    }
    setLoading(false);
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(signupData)
      });

      const data = await response.json();
      if (response.ok) {
        alert('Account created! Please login.');
        setActiveTab('login');
      } else {
        alert(data.error || 'Signup failed');
      }
    } catch (error) {
      alert('Connection error');
    }
    setLoading(false);
  };

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      setSelectedImages(files);
      setPredictions([]); // Clear previous predictions
      
      const previews = [];
      let loadedCount = 0;
      
      files.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          previews[index] = {
            id: index,
            file: file,
            preview: e.target.result,
            name: file.name
          };
          loadedCount++;
          
          if (loadedCount === files.length) {
            setImagePreviews(previews);
          }
        };
        reader.readAsDataURL(file);
      });
    }
  };

  const handleExcelUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedExcel(file);
    }
  };

  const handleBatchPredict = async () => {
    if (selectedImages.length === 0 || !user?.id) return;

    setLoading(true);
    
    const formData = new FormData();
    
    // Add images
    selectedImages.forEach((image) => {
      formData.append('images', image);
    });
    
    // Add Excel file if selected
    if (selectedExcel) {
      formData.append('excel_file', selectedExcel);
    }
    
    formData.append('user_id', user.id);

    try {
      const response = await fetch(`${API_BASE}/batch-predict`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (response.ok) {
        const formattedResults = data.results.map((result, index) => ({
          id: result.id || index,
          fileName: result.fileName,
          prediction: result.prediction,
          patient_data: result.patient_data,
          error: result.error,
          success: result.success,
          preview: imagePreviews.find(p => p.name === result.fileName)?.preview
        }));
        setPredictions(formattedResults);
        loadHistory(user.id);
      } else {
        alert(data.error || 'Batch prediction failed');
      }
    } catch (error) {
      alert('Connection error');
    }
    
    setLoading(false);
    setProcessingIndex(-1);
  };

  const removeImage = (indexToRemove) => {
    const newImages = selectedImages.filter((_, index) => index !== indexToRemove);
    const newPreviews = imagePreviews.filter((_, index) => index !== indexToRemove);
    const newPredictions = predictions.filter((_, index) => index !== indexToRemove);
    
    setSelectedImages(newImages);
    setImagePreviews(newPreviews);
    setPredictions(newPredictions);
  };

  const downloadReport = async (predictionId) => {
    try {
      const response = await fetch(`${API_BASE}/download-report/${predictionId}?user_id=${user.id}`, {
        method: 'GET',
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        
        // Get filename from response headers or use default
        const contentDisposition = response.headers.get('content-disposition');
        let filename = 'Medical_Report.docx';
        if (contentDisposition) {
          const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
          if (filenameMatch && filenameMatch[1]) {
            filename = filenameMatch[1].replace(/['"]/g, '');
          }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('Failed to download report');
      }
    } catch (error) {
      alert('Error downloading report');
    }
  };

  const loadHistory = async (userId) => {
    try {
      const response = await fetch(`${API_BASE}/history?user_id=${userId}`);
      if (response.ok) {
        const data = await response.json();
        setHistory(data.history || []);
      }
    } catch (error) {
      console.error('Failed to load history');
    }
  };

  const handleLogout = () => {
    setUser(null);
    setPredictions([]);
    setSelectedImages([]);
    setImagePreviews([]);
    setSelectedExcel(null);
    setHistory([]);
  };

  if (!user) {
    return (
      <div className="login-page">
        <div className="login-background">
          <div className="bg-circle circle-1"></div>
          <div className="bg-circle circle-2"></div>
          <div className="bg-circle circle-3"></div>
        </div>

        <div className="login-card">
          <div className="login-header">
            <div className="logo-container">
              <Stethoscope className="logo-icon" />
            </div>
            <h1 className="app-title">NeuralSight</h1>
            <p className="app-subtitle">Advanced medical image analysis powered by AI</p>
          </div>

          <div className="tab-container">
            <button
              onClick={() => setActiveTab('login')}
              className={`tab-button ${activeTab === 'login' ? 'active' : ''}`}
            >
              <LogIn className="tab-icon" />
              Login
            </button>
            <button
              onClick={() => setActiveTab('signup')}
              className={`tab-button ${activeTab === 'signup' ? 'active' : ''}`}
            >
              <User className="tab-icon" />
              Sign Up
            </button>
          </div>

          {activeTab === 'login' ? (
            <div className="form-container">
              <div className="input-group">
                <input
                  type="email"
                  placeholder="Email address"
                  value={loginData.email}
                  onChange={(e) => setLoginData({ ...loginData, email: e.target.value })}
                  className="form-input"
                  required
                />
              </div>
              <div className="input-group">
                <input
                  type="password"
                  placeholder="Password"
                  value={loginData.password}
                  onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                  className="form-input"
                  required
                />
              </div>
              <button onClick={handleLogin} disabled={loading} className="submit-button">
                {loading ? 'Signing in...' : 'Sign In'}
              </button>
            </div>
          ) : (
            <div className="form-container">
              <div className="input-group">
                <input
                  type="text"
                  placeholder="Full Name"
                  value={signupData.name}
                  onChange={(e) => setSignupData({ ...signupData, name: e.target.value })}
                  className="form-input"
                  required
                />
              </div>
              <div className="input-group">
                <input
                  type="email"
                  placeholder="Email address"
                  value={signupData.email}
                  onChange={(e) => setSignupData({ ...signupData, email: e.target.value })}
                  className="form-input"
                  required
                />
              </div>
              <div className="input-group">
                <input
                  type="password"
                  placeholder="Password"
                  value={signupData.password}
                  onChange={(e) => setSignupData({ ...signupData, password: e.target.value })}
                  className="form-input"
                  required
                />
              </div>
              <button onClick={handleSignup} disabled={loading} className="submit-button">
                {loading ? 'Creating Account...' : 'Create Account'}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-content">
          <div className="nav-brand">
            <div className="nav-logo">
              <Stethoscope className="nav-icon" />
            </div>
            <div className="nav-title">
              <h1>NeuralSight</h1>
              <p>Advanced Medical Diagnostics</p>
            </div>
          </div>
          <div className="nav-user">
            <button 
              className="theme-toggle" 
              onClick={toggleDarkMode}
              title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {darkMode ? <Sun className="theme-icon" /> : <Moon className="theme-icon" />}
            </button>
            <div className="user-info">
              <div className="user-avatar">
                <User className="user-icon" />
              </div>
              <div>
                <p className="welcome-text">Welcome back,</p>
                <p className="user-name">{user.name}</p>
              </div>
            </div>
            <button className="logout-button" onClick={handleLogout}>
              <LogOut className="logout-icon" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </nav>

      <div className="main-content">
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-content">
              <div>
                <p className="stat-label">Total Scans</p>
                <p className="stat-value">{history.length}</p>
              </div>
              <div className="stat-icon-container blue">
                <Eye className="stat-icon" />
              </div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-content">
              <div>
                <p className="stat-label">Batch Size</p>
                <p className="stat-value green">{selectedImages.length}</p>
              </div>
              <div className="stat-icon-container green">
                <FolderOpen className="stat-icon" />
              </div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-content">
              <div>
                <p className="stat-label">AI Models</p>
                <p className="stat-value purple">1</p>
              </div>
              <div className="stat-icon-container purple">
                <Brain className="stat-icon" />
              </div>
            </div>
          </div>
        </div>

        <div className="content-grid">
          <div className="prediction-section">
            <div className="section-header">
              <div className="section-icon-container">
                <Camera className="section-icon" />
              </div>
              <h2 className="section-title">Upload Medical Images & Patient Data</h2>
              <div className="security-badge">
                <Shield className="badge-icon" />
                <span>Secure</span>
              </div>
            </div>

            <div className="upload-area">
              {/* Excel File Upload */}
              <div className="excel-upload-section">
                <div className="excel-upload-header">
                  <FileSpreadsheet className="excel-icon" />
                  <h3>Patient Data (Optional)</h3>
                </div>
                <input 
                  type="file" 
                  accept=".xlsx,.xls" 
                  onChange={handleExcelUpload} 
                  className="file-input" 
                  id="excel-upload" 
                />
                <label htmlFor="excel-upload" className="excel-upload-label">
                  {selectedExcel ? (
                    <div className="excel-selected">
                      <FileSpreadsheet className="excel-icon-selected" />
                      <span>{selectedExcel.name}</span>
                      <button 
                        className="remove-excel-btn" 
                        onClick={(e) => {
                          e.preventDefault();
                          setSelectedExcel(null);
                        }}
                      >
                        <X size={16} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <FileSpreadsheet className="excel-upload-icon" />
                      <span>Upload Excel file with patient data</span>
                    </>
                  )}
                </label>
                <p className="excel-note">
                  Excel should contain: id, full_name, image_name, age, symptoms columns
                </p>
              </div>

              {/* Image Upload */}
              <input 
                type="file" 
                accept="image/*" 
                multiple 
                onChange={handleImageUpload} 
                className="file-input" 
                id="image-upload" 
              />
              <label htmlFor="image-upload" className="upload-label">
                <div className="upload-icon-container">
                  <FolderOpen className="upload-icon" />
                </div>
                <p className="upload-title">Drop your images here</p>
                <p className="upload-subtitle">or click to browse multiple files</p>
                <p className="upload-note">PNG, JPG, JPEG up to 10MB each</p>
              </label>

              {imagePreviews.length > 0 && (
                <div className="preview-container">
                  <div className="images-grid">
                    {imagePreviews.map((imageObj, index) => (
                      <div key={index} className="image-preview-item">
                        <div className="image-preview-wrapper">
                          <img src={imageObj.preview} alt={`Preview ${index}`} className="preview-image-small" />
                          <button 
                            className="remove-image-btn" 
                            onClick={() => removeImage(index)}
                            disabled={loading}
                          >
                            <X size={16} />
                          </button>
                          {processingIndex === index && (
                            <div className="processing-overlay">
                              <div className="loading-spinner"></div>
                            </div>
                          )}
                        </div>
                        <p className="image-name">{imageObj.name}</p>
                      </div>
                    ))}
                  </div>
                  
                  <button 
                    onClick={handleBatchPredict} 
                    disabled={loading} 
                    className="predict-button batch-predict"
                  >
                    {loading ? (
                      <>
                        <div className="loading-spinner"></div>
                        <span>Analyzing Images...</span>
                      </>
                    ) : (
                      <>
                        <Brain className="button-icon" />
                        <span>Analyze All Images ({selectedImages.length})</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="prediction-section">
            <div className="section-header">
              <div className="section-icon-container" style={{ background: 'linear-gradient(135deg, #10b981, #059669)' }}>
                <AlertCircle className="section-icon" />
              </div>
              <h2 className="section-title">Batch Analysis Results</h2>
              {predictions.length > 0 && (
                <div className="status-badge">
                  <Star className="badge-icon" />
                  <span>{predictions.length} Analyzed</span>
                </div>
              )}
            </div>

            {predictions.length > 0 ? (
              <div className="results-container batch-results">
                {predictions.map((result, index) => (
                  <div key={index} className="batch-result-item">
                    <div className="result-image">
                      <img src={result.preview} alt={result.fileName} className="result-thumbnail" />
                    </div>
                    <div className="result-details">
                      <h4 className="result-filename">{result.fileName}</h4>
                      
                      {/* Patient Data Section */}
                      {result.patient_data && Object.keys(result.patient_data).length > 0 && (
                        <div className="patient-data-section">
                          <h5 className="patient-data-title">Patient Information</h5>
                          <div className="patient-data-grid">
                            {result.patient_data.id && (
                              <div className="patient-data-item">
                                <span className="data-label">ID:</span>
                                <span className="data-value">{result.patient_data.id}</span>
                              </div>
                            )}
                            {result.patient_data.full_name && (
                              <div className="patient-data-item">
                                <span className="data-label">Name:</span>
                                <span className="data-value">{result.patient_data.full_name}</span>
                              </div>
                            )}
                            {result.patient_data.age && (
                              <div className="patient-data-item">
                                <span className="data-label">Age:</span>
                                <span className="data-value">{result.patient_data.age} years</span>
                              </div>
                            )}
                            {result.patient_data.symptoms && (
                              <div className="patient-data-item symptoms">
                                <span className="data-label">Symptoms:</span>
                                <span className="data-value">{result.patient_data.symptoms}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {result.error ? (
                        <div className="result-error">
                          <AlertCircle className="error-icon" />
                          <span>{result.error}</span>
                        </div>
                      ) : (
                        <div className="result-success">
                          <div className="result-disease">{result.prediction.disease}</div>
                          <div className="confidence-meter-small">
                            <div className="meter-track">
                              <div 
                                className="meter-progress" 
                                style={{ width: `${result.prediction.confidence * 100}%` }}
                              ></div>
                            </div>
                            <span className="confidence-value">
                              {(result.prediction.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          {result.prediction.recommendations && (
                            <p className="result-recommendations">
                              {result.prediction.recommendations}
                            </p>
                          )}
                          
                          {/* Download Report Button */}
                          {result.id && (
                            <button 
                              className="download-report-btn"
                              onClick={() => downloadReport(result.id)}
                              title="Download detailed medical report"
                            >
                              <Download className="download-icon" />
                              <FileText className="report-icon" />
                              <span>Download Report</span>
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon-container">
                  <FolderOpen className="empty-icon" />
                </div>
                <h3 className="empty-title">Ready for Batch Analysis</h3>
                <p className="empty-subtitle">Upload multiple medical images and optional Excel file to get AI-powered insights with patient data</p>
              </div>
            )}
          </div>
        </div>

        {history.length > 0 && (
          <div className="history-section">
            <div className="section-header">
              <div className="section-icon-container" style={{ background: 'linear-gradient(135deg, #7c3aed, #a855f7)' }}>
                <Clock className="section-icon" />
              </div>
              <h2 className="section-title">Recent Analysis History</h2>
            </div>
            <div className="history-grid">
              {history.slice(0, 6).map((item, index) => (
                <div key={index} className="history-card">
                  <div className="card-header">
                    <div className="card-icon-container">
                      <Eye className="card-icon" />
                    </div>
                    <div className="confidence-badge">
                      <span>{(item.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  <h3 className="card-title">{item.disease}</h3>
                  <p className="card-filename">{item.image_filename}</p>
                  {item.patient_data && item.patient_data.name && (
                    <p className="card-patient">{item.patient_data.name}</p>
                  )}
                  <p className="card-date">
                    {new Date(item.timestamp).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric'
                    })}
                  </p>
                  <button 
                    className="history-download-btn"
                    onClick={() => downloadReport(item.id)}
                    title="Download report"
                  >
                    <Download size={16} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>

  );
};

export default App;