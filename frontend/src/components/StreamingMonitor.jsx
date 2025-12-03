import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { streamFeatures } from '../utils/streamProcessor';
import './StreamingMonitor.css';

const StreamingMonitor = ({ enabled, onToggle }) => {
  const [streamData, setStreamData] = useState([]);
  const [currentFeatures, setCurrentFeatures] = useState(null);
  const [fps, setFps] = useState(0);
  const streamRef = useRef(null);
  const fpsRef = useRef({ count: 0, lastTime: Date.now() });

  useEffect(() => {
    if (enabled) {
      startStreaming();
    } else {
      stopStreaming();
    }

    return () => stopStreaming();
  }, [enabled]);

  const startStreaming = () => {
    streamRef.current = streamFeatures((features, label) => {
      const now = Date.now();
      fpsRef.current.count++;

      if (now - fpsRef.current.lastTime >= 1000) {
        setFps(fpsRef.current.count);
        fpsRef.current = { count: 0, lastTime: now };
      }

      setCurrentFeatures({ features, label, timestamp: now });

      setStreamData((prev) => {
        const newData = [
          ...prev,
          {
            time: now,
            mean: features[0],
            std: features[1],
            rms: features[8],
          },
        ].slice(-100);
        return newData;
      });
    });
  };

  const stopStreaming = () => {
    if (streamRef.current) {
      streamRef.current();
      streamRef.current = null;
    }
  };

  return (
    <div className="streaming-monitor">
      <div className="stream-controls">
        <button
          className={`stream-toggle ${enabled ? 'active' : ''}`}
          onClick={() => onToggle(!enabled)}
        >
          {enabled ? '⏸ Pause' : '▶ Start'} Streaming
        </button>
        <div className="stream-stats">
          <span className="fps-counter">{fps} FPS</span>
          <span className="buffer-size">
            Buffer: {streamData.length}/100
          </span>
        </div>
      </div>

      {currentFeatures && (
        <div className="current-features">
          <h3>
            Current Sample
            <span
              className={`label-indicator ${currentFeatures.label === 1 ? 'stress' : 'normal'}`}
            >
              {currentFeatures.label === 1 ? 'STRESS' : 'Normal'}
            </span>
          </h3>

          <div className="features-grid">
            {currentFeatures.features.slice(0, 10).map((val, idx) => (
              <div key={idx} className="feature-card">
                <div className="feature-name">F{idx}</div>
                <div className="feature-value">{val.toFixed(4)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="stream-chart">
        <h3>Real-time Feature Stream</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={streamData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="time"
              tickFormatter={(t) =>
                new Date(t).toLocaleTimeString()
              }
            />
            <YAxis />
            <Tooltip
              labelFormatter={(t) =>
                new Date(t).toLocaleTimeString()
              }
            />
            <Line
              type="monotone"
              dataKey="mean"
              stroke="#8884d8"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="std"
              stroke="#82ca9d"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="rms"
              stroke="#ffc658"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default StreamingMonitor;