import { useState, useEffect, useCallback } from 'react';
import { socketService } from '../api/socket';

export type Metrics = {
    accuracy: number;
    loss: number;
    epsilon: number;
    minority_recall: number;
};

export type LogMessage = {
    id: string;
    timestamp: string;
    message: string;
};

export type RunHistory = {
    id: string;
    config: any;
    finalMetrics: Metrics;
    date: string;
};

export const useSimulation = () => {
    const [status, setStatus] = useState<'idle' | 'running' | 'finished'>('idle');
    const [currentRound, setCurrentRound] = useState(0);
    const [activeClient, setActiveClient] = useState<number | null>(null);
    const [metricsHistory, setMetricsHistory] = useState<Metrics[]>([]);
    const [logs, setLogs] = useState<LogMessage[]>([]);
    const [history, setHistory] = useState<RunHistory[]>([]);
    const [currentConfig, setCurrentConfig] = useState<any>(null);

    const addLog = useCallback((msg: string) => {
        setLogs(prev => [...prev, {
            id: crypto.randomUUID(),
            timestamp: new Date().toLocaleTimeString(),
            message: msg
        }].slice(-50)); // Keep last 50 logs
    }, []);

    useEffect(() => {
        socketService.connect();

        const unsub = socketService.subscribe((data) => {
            switch (data.type) {
                case 'log':
                    addLog(data.message);
                    break;
                case 'round_start':
                    setCurrentRound(data.round);
                    setActiveClient(null);
                    break;
                case 'client_training':
                    setActiveClient(data.client_id);
                    break;
                case 'round_complete':
                    setMetricsHistory((prev: Metrics[]) => [...prev, data.metrics]);
                    setActiveClient(null);
                    break;
                case 'finished':
                    setStatus('finished');
                    setActiveClient(null);
                    addLog("Simulation finished successfully.");
                    // Save run to history
                    if (currentConfig && metricsHistory.length > 0) {
                        const lastMetric = metricsHistory[metricsHistory.length - 1];
                        setHistory(prev => [{
                            id: crypto.randomUUID(),
                            config: currentConfig,
                            finalMetrics: lastMetric || data.metrics,
                            date: new Date().toLocaleTimeString()
                        }, ...prev]);
                    }
                    break;
                case 'error':
                    addLog(`Error: ${data.message}`);
                    setStatus('idle');
                    break;
                default:
                    console.log('Unknown event type:', data.type);
            }
        });
        
        return unsub;
    }, [addLog, currentConfig, metricsHistory]);

    const start = async (config: any) => {
        setMetricsHistory([]);
        setLogs([]);
        setCurrentRound(0);
        setStatus('running');
        setCurrentConfig(config);
        addLog(`Connecting to server...`);
        try {
            await socketService.startTraining(config);
            addLog(`Server connection established. Initializing simulation...`);
        } catch (error) {
            addLog(`Failed to connect: ${error}`);
            setStatus('idle');
        }
    };

    const stop = () => {
        socketService.stopTraining();
        setStatus('idle');
        addLog("Simulation stopped by user.");
    };

    return { status, currentRound, activeClient, metricsHistory, logs, history, start, stop };
};
