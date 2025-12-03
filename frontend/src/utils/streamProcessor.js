export const streamFeatures = (callback, interval = 100) => {
  let streamInterval;
  let sampleIndex = 0;

  const generateRandomFeatures = () => {
    const features = new Array(140).fill(0).map(() => Math.random() * 10);
    const label = Math.random() > 0.7 ? 1 : 0;
    return { features, label };
  };

  streamInterval = setInterval(() => {
    const { features, label } = generateRandomFeatures();
    callback(features, label);
    sampleIndex++;
  }, interval);

  return () => {
    if (streamInterval) {
      clearInterval(streamInterval);
    }
  };
};

export class FeatureBuffer {
  constructor(maxSize = 1000) {
    this.buffer = [];
    this.maxSize = maxSize;
  }

  push(features, label) {
    this.buffer.push({ features, label, timestamp: Date.now() });
    if (this.buffer.length > this.maxSize) {
      this.buffer.shift();
    }
  }

  getRecent(n = 100) {
    return this.buffer.slice(-n);
  }

  clear() {
    this.buffer = [];
  }
}