import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import ML_UI from './pages/ML_UI';
import Trained_model from './pages/Trained_model';
const App = () => {
  return (
    <Router>
      <div>
        <Routes>
          <Route path="/" element={<ML_UI />} />
          <Route path="/pretrainedModels" element={<Trained_model/>} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
