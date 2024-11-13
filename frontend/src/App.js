import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import ML_UI from './pages/ML_UI';

const App = () => {
  return (
    <Router>
      <div>
        <Routes>
          <Route path="/" element={<ML_UI />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
