import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';
import './ClassDistribution.css';

const ClassDistribution = ({ metadata, data }) => {
  if (!metadata || !data) return null;

  const classData = Object.entries(metadata.class_counts_train).map(
    ([name, count]) => ({
      name,
      value: count,
    })
  );

  const COLORS = ['#0088FE', '#FF8042'];

  return (
    <div className="class-distribution">
      <h2>Class Distribution</h2>

      <div className="distribution-grid">
        <div className="chart-section">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={classData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) =>
                  `${name}: ${(percent * 100).toFixed(1)}%`
                }
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {classData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="weights-section">
          <h3>Class Weights</h3>
          <div className="weights-list">
            {Object.entries(metadata.class_weights).map(([cls, weight]) => (
              <div key={cls} className="weight-item">
                <span className="weight-class">
                  {metadata.class_names[cls]}
                </span>
                <span className="weight-value">{weight.toFixed(3)}</span>
                <div className="weight-bar">
                  <div
                    className="weight-fill"
                    style={{ width: `${(weight / 2) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClassDistribution;