import React, { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import './FeatureVisualizer.css';

const FeatureVisualizer = ({ data, metadata }) => {
  const [selectedSample, setSelectedSample] = useState(0);
  const [selectedChannel, setSelectedChannel] = useState(0);

  const channelNames = [
    'ECG',
    'EDA (Chest)',
    'Temp (Chest)',
    'EMG',
    'Resp',
    'BVP',
    'EDA (Wrist)',
    'Temp (Wrist)',
    'ACC Chest X',
    'ACC Chest Y',
    'ACC Chest Z',
    'ACC Wrist X',
    'ACC Wrist Y',
    'ACC Wrist Z',
  ];

  const featureNames = [
    'Mean',
    'Std',
    'Min',
    'Max',
    'Median',
    'Q25',
    'Q75',
    'Range',
    'RMS',
    'Total Var',
  ];

  const chartData = useMemo(() => {
    if (!data || !data.features) return [];

    const sample = data.features[selectedSample];
    const channelStart = selectedChannel * 10;
    const channelFeatures = sample.slice(
      channelStart,
      channelStart + 10
    );

    return featureNames.map((name, idx) => ({
      name,
      value: channelFeatures[idx],
    }));
  }, [data, selectedSample, selectedChannel]);

  const allChannelData = useMemo(() => {
    if (!data || !data.features) return [];

    const sample = data.features[selectedSample];
    return channelNames.map((name, idx) => {
      const channelStart = idx * 10;
      const channelFeatures = sample.slice(
        channelStart,
        channelStart + 10
      );
      return {
        name,
        mean: channelFeatures[0],
        std: channelFeatures[1],
        range: channelFeatures[7],
      };
    });
  }, [data, selectedSample]);

  if (!data) return null;

  return (
    <div className="feature-visualizer">
      <div className="controls">
        <div className="control-group">
          <label>Sample:</label>
          <input
            type="range"
            min="0"
            max={data.features.length - 1}
            value={selectedSample}
            onChange={(e) => setSelectedSample(Number(e.target.value))}
          />
          <span>
            {selectedSample + 1} / {data.features.length}
          </span>
          <span className="label-badge">
            Label: {data.labels[selectedSample] === 0
              ? 'Non-Stress'
              : 'Stress'}
          </span>
        </div>

        <div className="control-group">
          <label>Channel:</label>
          <select
            value={selectedChannel}
            onChange={(e) => setSelectedChannel(Number(e.target.value))}
          >
            {channelNames.map((name, idx) => (
              <option key={idx} value={idx}>
                {name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="charts-container">
        <div className="chart-card">
          <h3>
            10 Features - {channelNames[selectedChannel]}
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#8884d8"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>All Channels Overview (Mean, Std, Range)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={allChannelData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="mean" stroke="#82ca9d" />
              <Line type="monotone" dataKey="std" stroke="#8884d8" />
              <Line type="monotone" dataKey="range" stroke="#ffc658" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="feature-matrix">
        <h3>Feature Matrix (140D)</h3>
        <div className="matrix-grid">
          {Array.from({ length: 14 }).map((_, chIdx) => (
            <div key={chIdx} className="matrix-row">
              <div className="matrix-label">{channelNames[chIdx]}</div>
              <div className="matrix-cells">
                {Array.from({ length: 10 }).map((_, featIdx) => {
                  const value =
                    data.features[selectedSample][chIdx * 10 + featIdx];
                  return (
                    <div
                      key={featIdx}
                      className="matrix-cell"
                      title={`${featureNames[featIdx]}: ${value.toFixed(4)}`}
                      style={{
                        backgroundColor: `rgba(136, 132, 216, ${Math.abs(value) / 10})`,
                      }}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FeatureVisualizer;