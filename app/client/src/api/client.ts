import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Training API
export const startTraining = async (config: {
  dataset: 'wesad' | 'sleep-edf';
  clients: number;
  sigma: number;
  batch_size?: number;
  epochs?: number;
}) => {
  const response = await apiClient.post('/train', config);
  return response.data;
};

export const getJobStatus = async (jobId: string) => {
  const response = await apiClient.get(`/status/${jobId}`);
  return response.data;
};

export const estimatePrivacy = async (config: {
  dataset: 'wesad' | 'sleep-edf';
  clients: number;
  sigma: number;
  epochs: number;
}) => {
  const response = await apiClient.post('/estimate-privacy', {
    dataset_size: config.dataset === 'wesad' ? 5000 : 3000, // Approximate
    batch_size: 128,
    epochs: config.epochs,
    sigma: config.sigma
  });
  return response.data;
};

export const stopTraining = async (jobId: string) => {
  const response = await apiClient.post(`/stop/${jobId}`);
  return response.data;
};
