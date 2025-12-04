import { useState, useRef, useEffect, useCallback } from 'react';
import { startTraining, getJobStatus, stopTraining } from '../api/client';

export type TrainingStatus = 'idle' | 'running' | 'completed' | 'failed' | 'queued';

export type TrainingMetrics = {
    accuracy: number;
    loss: number;
    epsilon: number;
    minority_recall: number;
    round?: number;
    epoch?: number;
};

export const useTraining = () => {
    const [status, setStatus] = useState<TrainingStatus>('idle');
    const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
    const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([]);
    const [logs, setLogs] = useState<string[]>([]);
    const [jobId, setJobId] = useState<string | null>(() => {
        // Load from localStorage on mount
        if (typeof window !== 'undefined') {
            return localStorage.getItem('active_job_id');
        }
        return null;
    });
    const [mode, setMode] = useState<string>('');
    const [progress, setProgress] = useState(0);
    const [currentRound, setCurrentRound] = useState(0);
    const [activeClient, setActiveClient] = useState<number | null>(null);
    
    const pollInterval = useRef<number | null>(null);

    // Poll function
    const checkStatus = useCallback(async (id: string) => {
        try {
            const res = await getJobStatus(id);
            const data = res;
            
            setLogs(data.logs || []);
            setProgress(data.progress || 0);
            setMode(data.mode_detected || 'UNKNOWN');
            
            // Derive round from progress or use current_round
            if (data.current_round) {
                setCurrentRound(data.current_round);
            } else if (data.progress > 0) {
                // Approximate: assuming 40 rounds max
                setCurrentRound(Math.ceil(data.progress / (100 / 40)));
            }

            // Parse logs to detect active client
            const latestLog = data.logs && data.logs.length > 0 ? data.logs[data.logs.length - 1] : '';
            if (latestLog.includes('Client') && latestLog.includes('computing')) {
                // Extract client number from log: "Client 3/8 computing..."
                const match = latestLog.match(/Client (\d+)\//);
                if (match) {
                    setActiveClient(parseInt(match[1]) - 1); // Convert to 0-indexed
                }
            } else if (latestLog.includes('Aggregating') || latestLog.includes('Round')) {
                // Clear active client when aggregating or starting new round
                setActiveClient(null);
            }
            
            // Update metrics if changed
            if (data.metrics) {
                setMetrics(data.metrics);
                // In a real app, backend should send full history array. 
                // For MVP polling, we just push latest if different.
                setMetricsHistory(prev => {
                    const last = prev[prev.length - 1];
                    // Only add if accuracy changed or it's a new epoch/round
                    if (!last || 
                        last.accuracy !== data.metrics.accuracy ||
                        (data.metrics.round && last.round !== data.metrics.round) ||
                        (data.metrics.epoch && last.epoch !== data.metrics.epoch)) {
                        return [...prev, data.metrics];
                    }
                    return prev;
                });
            }
            
            if (data.status === 'completed' || data.status === 'failed') {
                setStatus(data.status);
                if (typeof window !== 'undefined') {
                    localStorage.removeItem('active_job_id');
                }
                if (pollInterval.current) {
                    window.clearInterval(pollInterval.current);
                    pollInterval.current = null;
                }
            } else if (data.status === 'running' || data.status === 'pending') {
                setStatus('running');
            }
        } catch (e) {
            console.error("Polling error", e);
            // Don't stop polling immediately on network hiccup
        }
    }, []);

    // Start Polling if Job ID exists on mount
    useEffect(() => {
        if (jobId) {
            setStatus('running');
            pollInterval.current = window.setInterval(() => checkStatus(jobId), 2000);
            // Immediate first check
            checkStatus(jobId);
        }
        return () => {
            if (pollInterval.current) {
                window.clearInterval(pollInterval.current);
                pollInterval.current = null;
            }
        };
    }, [jobId, checkStatus]);

    const start = useCallback(async (config: any) => {
        try {
            // Reset state
            setMetricsHistory([]);
            setLogs([]);
            setProgress(0);
            setCurrentRound(0);
            
            const res = await startTraining(config);
            const newJobId = res.job_id;
            
            setJobId(newJobId);
            setMode(res.mode_detected || 'UNKNOWN');
            setStatus('running');
            
            if (typeof window !== 'undefined') {
                localStorage.setItem('active_job_id', newJobId);
            }
            
            // Start polling
            if (pollInterval.current) {
                window.clearInterval(pollInterval.current);
            }
            pollInterval.current = window.setInterval(() => checkStatus(newJobId), 2000);
            
            // Immediate first check
            checkStatus(newJobId);
            
        } catch (e: any) {
            console.error('Failed to start training:', e);
            setStatus('failed');
            setLogs(prev => [...prev, `Failed to start training: ${e.message || 'Unknown error'}`]);
        }
    }, [checkStatus]);

    const stop = useCallback(async () => {
        if (jobId) {
            try {
                await stopTraining(jobId);
                setLogs(prev => [...prev, "🛑 Stopping requested..."]);
            } catch (e) {
                console.error('Failed to stop training:', e);
            }
        }
        if (typeof window !== 'undefined') {
            localStorage.removeItem('active_job_id');
        }
        if (pollInterval.current) {
            window.clearInterval(pollInterval.current);
            pollInterval.current = null;
        }
        setStatus('idle');
        setJobId(null);
    }, [jobId]);

    return { 
        status, 
        metrics, 
        metricsHistory, 
        logs, 
        mode, 
        progress, 
        currentRound, 
        activeClient,
        jobId, 
        start, 
        stop 
    };
};
