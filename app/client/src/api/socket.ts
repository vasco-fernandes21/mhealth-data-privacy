type TrainingConfig = {
    dataset: 'wesad' | 'sleep-edf';
    clients: number;
    sigma: number;
};

type SocketListener = (data: any) => void;

export class TrainingSocket {
    private ws: WebSocket | null = null;
    private listeners: SocketListener[] = [];
    private clientId: string;
    private reconnectAttempts: number = 0;
    private maxReconnectAttempts: number = 5;
    private reconnectInterval: number = 2000; // 2 seconds
    private pendingConfig: TrainingConfig | null = null;
    private connectionPromise: Promise<void> | null = null;

    constructor() {
        this.clientId = crypto.randomUUID();
    }

    private setupWebSocketListeners(ws: WebSocket) {
        ws.onopen = () => {
            console.log("WebSocket connected:", ws.url);
            this.reconnectAttempts = 0; // Reset attempts on successful connect
            // Notify listeners of successful connection
            this.listeners.forEach(cb => cb({ 
                type: 'log', 
                message: 'WebSocket connected. Ready for commands.' 
            }));
            // Send pending config if any
            if (this.pendingConfig) {
                console.log('Sending pending training config');
                this.startTraining(this.pendingConfig);
                this.pendingConfig = null;
            }
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log('Received WebSocket message:', data);
                this.listeners.forEach(cb => cb(data));
            } catch (e) {
                console.error('Error parsing WebSocket message:', e, event.data);
            }
        };

        ws.onerror = (event) => {
            console.error("WebSocket error:", event);
            // It's possible for onerror to fire multiple times for a single disconnection
            // Check readyState to prevent multiple reconnect attempts for the same issue
            if (this.ws?.readyState === WebSocket.CLOSED || this.ws?.readyState === WebSocket.CLOSING) {
                console.log("WebSocket state:", this.ws.readyState);
            }
        };

        ws.onclose = (event) => {
            console.log(`WebSocket disconnected. Code: ${event.code} Reason: ${event.reason}`);
            this.ws = null; // Clear reference to closed WebSocket
            this.connectionPromise = null;
            
            if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) { // 1000 is normal closure
                console.log("Attempting to reconnect...");
                this.reconnectAttempts++;
                setTimeout(() => this.connect(), this.reconnectInterval);
            } else if (event.code !== 1000) {
                console.error("Max reconnect attempts reached. Please restart the application or server.");
                // Optionally notify UI
                this.listeners.forEach(cb => cb({ type: 'error', message: 'Max reconnect attempts reached.' }));
            }
        };
    }

    connect(): Promise<void> {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            console.log("WebSocket already open or connecting.");
            return Promise.resolve();
        }

        if (this.connectionPromise) {
            return this.connectionPromise;
        }

        this.connectionPromise = new Promise((resolve, reject) => {
            const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
            const fullUrl = `${wsUrl}/ws/training/${this.clientId}`;
            console.log('Connecting to WebSocket:', fullUrl);
            this.ws = new WebSocket(fullUrl);
            
            this.setupWebSocketListeners(this.ws);
            
            this.ws.onopen = () => {
                console.log("WebSocket connected successfully");
                this.reconnectAttempts = 0;
                this.connectionPromise = null;
                resolve();
                // Send pending config if any
                if (this.pendingConfig) {
                    console.log('Sending pending training config');
                    this.startTraining(this.pendingConfig);
                    this.pendingConfig = null;
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket connection error:', error);
                this.connectionPromise = null;
                reject(error);
            };
        });

        return this.connectionPromise;
    }

    async startTraining(config: TrainingConfig): Promise<void> {
        // Ensure WebSocket is connected before sending
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.log('WebSocket not ready. Current state:', this.ws?.readyState, '- Connecting...');
            this.pendingConfig = config;
            try {
                await this.connect();
                // Wait a bit for the connection to fully establish
                await new Promise(resolve => setTimeout(resolve, 100));
                // After connection, send the config
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    console.log('WebSocket connected, sending start training message:', config);
                    this.ws.send(JSON.stringify({ action: 'start', config }));
                    this.pendingConfig = null;
                } else {
                    throw new Error('WebSocket connection failed');
                }
            } catch (err) {
                console.error('Failed to connect WebSocket:', err);
                this.pendingConfig = null;
                // Notify listeners of error
                this.listeners.forEach(cb => cb({ 
                    type: 'error', 
                    message: 'Failed to connect to server. Please check if backend is running.' 
                }));
                throw err;
            }
            return;
        }
        console.log('Sending start training message:', config);
        this.ws.send(JSON.stringify({ action: 'start', config }));
    }

    stopTraining() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ action: 'stop' }));
        }
    }

    subscribe(callback: SocketListener) {
        this.listeners.push(callback);
        return () => {
            this.listeners = this.listeners.filter(cb => cb !== callback);
        };
    }
}

export const socketService = new TrainingSocket();
