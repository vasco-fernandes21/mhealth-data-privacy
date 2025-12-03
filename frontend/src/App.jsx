import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    document.body.className = darkMode ? 'dark-mode' : 'light-mode';
  }, [darkMode]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>WESAD Dataset Visualizer</h1>
        <button
          className="theme-toggle"
          onClick={() => setDarkMode(!darkMode)}
        >
          {darkMode ? '☀️' : '🌙'}
        </button>
      </header>
      <Dashboard />
    </div>
  );
}

export default App;