import React, { useState, useEffect } from 'react';
import DatasetInfo from './DatasetInfo';
import FeatureVisualizer from './FeatureVisualizer';
import ClassDistribution from './ClassDistribution';
import ChannelAnalysis from './ChannelAnalysis';
import StreamingMonitor from './StreamingMonitor';
import { loadDatasetInfo, loadFeatureData } from '../utils/dataLoader';
import './Dashboard.css';

const Dashboard = () => {
  const [metadata, setMetadata] = useState(null);
  const [featureData, setFeatureData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [streamingEnabled, setStreamingEnabled] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const meta = await loadDatasetInfo();
      const data = await loadFeatureData('train', 1000); // Load first 1000
      setMetadata(meta);
      setFeatureData(data);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <p>Loading WESAD dataset...</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-tabs">
        <button
          className={activeTab === 'overview' ? 'active' : ''}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={activeTab === 'features' ? 'active' : ''}
          onClick={() => setActiveTab('features')}
        >
          Features
        </button>
        <button
          className={activeTab === 'channels' ? 'active' : ''}
          onClick={() => setActiveTab('channels')}
        >
          Channels
        </button>
        <button
          className={activeTab === 'streaming' ? 'active' : ''}
          onClick={() => setActiveTab('streaming')}
        >
          Live Stream
        </button>
      </div>

      <div className="dashboard-content">
        {activeTab === 'overview' && (
          <>
            <DatasetInfo metadata={metadata} />
            <ClassDistribution metadata={metadata} data={featureData} />
          </>
        )}

        {activeTab === 'features' && (
          <FeatureVisualizer data={featureData} metadata={metadata} />
        )}

        {activeTab === 'channels' && (
          <ChannelAnalysis data={featureData} metadata={metadata} />
        )}

        {activeTab === 'streaming' && (
          <StreamingMonitor
            enabled={streamingEnabled}
            onToggle={setStreamingEnabled}
          />
        )}
      </div>
    </div>
  );
};

export default Dashboard;