import axios from 'axios';

export const apiClient = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const estimatePrivacy = async (sigma: number, epochs: number) => {
  const response = await apiClient.post('/estimate-privacy', {
    sigma,
    epochs,
    dataset_size: 5000, // Approximation for UI
    batch_size: 32
  });
  return response.data;
};

