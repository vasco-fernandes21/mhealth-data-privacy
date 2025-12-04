import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
const API_KEY = import.meta.env.VITE_API_KEY || 'mhealth-secret-2024';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  },
});

// Training API
export const startTraining = async (config: {
  dataset: 'wesad' | 'sleep-edf';
  clients: number;
  sigma: number;
  // Advanced scientific controls (all optional on the frontend side)
  seed?: number | null;
  max_grad_norm?: number;
  use_class_weights?: boolean;
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

export const getHistory = async (limit = 20) => {
  const response = await apiClient.get(`/history?limit=${limit}`);
  return response.data;
};

export const deleteHistoryRun = async (id: string) => {
  const response = await apiClient.delete(`/history/${id}`);
  return response.data;
};

export const exportRun = async (id: string) => {
  const response = await apiClient.get(`/export/${id}`);
  return response.data;
};
