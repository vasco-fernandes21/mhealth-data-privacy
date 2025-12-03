import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const loadDatasetInfo = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/metadata`);
    return response.data;
  } catch (error) {
    console.error('Error loading metadata:', error);
    throw error;
  }
};

export const loadFeatureData = async (split = 'train', limit = 1000) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/data/${split}?limit=${limit}`
    );
    return response.data;
  } catch (error) {
    console.error('Error loading feature data:', error);
    throw error;
  }
};

export const loadBatch = async (split, start, end) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/data/${split}/batch?start=${start}&end=${end}`
    );
    return response.data;
  } catch (error) {
    console.error('Error loading batch:', error);
    throw error;
  }
};