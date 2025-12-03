import React from 'react';
import { Info, Users, Database, Clock } from 'lucide-react';
import './DatasetInfo.css';

const DatasetInfo = ({ metadata }) => {
  if (!metadata) return null;

  const formatSize = (mb) => {
    if (mb > 1024) return `${(mb / 1024).toFixed(2)} GB`;
    return `${mb.toFixed(2)} MB`;
  };

  const stats = [
    {
      icon: <Database />,
      label: 'Total Samples',
      value: (
        metadata.train_size +
        metadata.val_size +
        metadata.test_size
      ).toLocaleString(),
    },
    {
      icon: <Users />,
      label: 'Subjects',
      value: metadata.total_subjects,
    },
    {
      icon: <Info />,
      label: 'Features',
      value: `${metadata.n_features}D (${metadata.features_per_channel} × ${metadata.n_channels})`,
    },
    {
      icon: <Clock />,
      label: 'Processing Time',
      value: `${metadata.processing_time_s.toFixed(1)}s`,
    },
  ];

  return (
    <div className="dataset-info">
      <h2>Dataset Overview</h2>

      <div className="stats-grid">
        {stats.map((stat, idx) => (
          <div key={idx} className="stat-card">
            <div className="stat-icon">{stat.icon}</div>
            <div className="stat-content">
              <div className="stat-label">{stat.label}</div>
              <div className="stat-value">{stat.value}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="split-info">
        <h3>Data Split</h3>
        <div className="split-bars">
          <div className="split-bar">
            <div className="split-label">
              <span>Train</span>
              <span>{metadata.train_size.toLocaleString()}</span>
            </div>
            <div className="split-progress">
              <div
                className="split-fill train"
                style={{
                  width: `${
                    (metadata.train_size /
                      (metadata.train_size +
                        metadata.val_size +
                        metadata.test_size)) *
                    100
                  }%`,
                }}
              />
            </div>
          </div>

          <div className="split-bar">
            <div className="split-label">
              <span>Validation</span>
              <span>{metadata.val_size.toLocaleString()}</span>
            </div>
            <div className="split-progress">
              <div
                className="split-fill val"
                style={{
                  width: `${
                    (metadata.val_size /
                      (metadata.train_size +
                        metadata.val_size +
                        metadata.test_size)) *
                    100
                  }%`,
                }}
              />
            </div>
          </div>

          <div className="split-bar">
            <div className="split-label">
              <span>Test</span>
              <span>{metadata.test_size.toLocaleString()}</span>
            </div>
            <div className="split-progress">
              <div
                className="split-fill test"
                style={{
                  width: `${
                    (metadata.test_size /
                      (metadata.train_size +
                        metadata.val_size +
                        metadata.test_size)) *
                    100
                  }%`,
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="channel-info">
        <h3>Channel Configuration</h3>
        <div className="channel-grid">
          <div className="channel-section">
            <h4>1D Channels (8)</h4>
            <ul>
              {metadata.channel_description['1D_channels'].map(
                (ch, idx) => (
                  <li key={idx}>{ch}</li>
                )
              )}
            </ul>
          </div>
          <div className="channel-section">
            <h4>3D Channels (6 = 2×3)</h4>
            <ul>
              {metadata.channel_description['3D_channels'].map(
                (ch, idx) => (
                  <li key={idx}>{ch}</li>
                )
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetInfo;