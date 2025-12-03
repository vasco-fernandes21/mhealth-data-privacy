import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import './ChannelAnalysis.css';

const ChannelAnalysis = ({ data, metadata }) => {
  const [featureIndex, setFeatureIndex] = useState(0);

  const featureNames = [
    'Mean',
    'Std Dev',
    'Min',
    'Max',
    'Median',
    'Q25',
    'Q75',
    'Range',
    'RMS',
    'Total Variation',
  ];

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

  const aggregatedData = useMemo(() => {
    if (!data || !data.features) return [];

    return channelNames.map((name, chIdx) => {
      const channelFeatures = data.features.map(
        (sample) => sample[chIdx * 10 + featureIndex]
      );

      const stressFeatures = data.features
        .filter((_, idx) => data.labels[idx] === 1)
        .map((sample) => sample[chIdx * 10 + featureIndex]);

      const normalFeatures = data.features
        .filter((_, idx) => data.labels[idx] === 0)
        .map((sample) => sample[chIdx * 10 + featureIndex]);

      return {
        name,
        overall:
          channelFeatures.reduce((a, b) => a + b, 0) /
          channelFeatures.length,
        stress:
          stressFeatures.length > 0
            ? stressFeatures.reduce((a, b) => a + b, 0) /
              stressFeatures.length
            : 0,
        normal:
          normalFeatures.length > 0
            ? normalFeatures.reduce((a, b) => a + b, 0) /
              normalFeatures.length
            : 0,
      };
    });
  }, [data, featureIndex]);

  if (!data) return null;

  return (
    <div className="channel-analysis">
      <div className="analysis-controls">
        <label>Feature to Analyze:</label>
        <select
          value={featureIndex}
          onChange={(e) => setFeatureIndex(Number(e.target.value))}
        >
          {featureNames.map((name, idx) => (
            <option key={idx} value={idx}>
              {name}
            </option>
          ))}
        </select>
      </div>

      <div className="analysis-chart">
        <h3>
          {featureNames[featureIndex]} Across All Channels
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={aggregatedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={120} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="normal" fill="#82ca9d" name="Non-Stress" />
            <Bar dataKey="stress" fill="#ff8042" name="Stress" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="channel-stats">
        <h3>Channel Statistics</h3>
        <div className="stats-table">
          <table>
            <thead>
              <tr>
                <th>Channel</th>
                <th>Overall</th>
                <th>Non-Stress</th>
                <th>Stress</th>
                <th>Difference</th>
              </tr>
            </thead>
            <tbody>
              {aggregatedData.map((row, idx) => (
                <tr key={idx}>
                  <td>{row.name}</td>
                  <td>{row.overall.toFixed(4)}</td>
                  <td>{row.normal.toFixed(4)}</td>
                  <td>{row.stress.toFixed(4)}</td>
                  <td
                    className={
                      Math.abs(row.stress - row.normal) > 0.1
                        ? 'significant'
                        : ''
                    }
                  >
                    {(row.stress - row.normal).toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ChannelAnalysis;