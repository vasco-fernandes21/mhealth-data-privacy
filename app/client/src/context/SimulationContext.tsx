import { createContext, useContext, ReactNode } from 'react';
import { useTraining, TrainingStatus, TrainingMetrics } from '../hooks/useTraining';

type SimContextType = {
    status: TrainingStatus;
    currentRound: number;
    metricsHistory: TrainingMetrics[];
    previousMetricsHistory: TrainingMetrics[];
    logs: string[];
    progress: number;
    mode: string;
    start: (config: any) => Promise<void>;
    stop: () => Promise<void>;
    activeClient: number | null;
    metrics: TrainingMetrics | null;
    jobId: string | null;
};

const SimulationContext = createContext<SimContextType | null>(null);

export const SimulationProvider = ({ children }: { children: ReactNode }) => {
    const training = useTraining();
    
    return (
        <SimulationContext.Provider value={{
            status: training.status,
            currentRound: training.currentRound,
            metricsHistory: training.metricsHistory,
            previousMetricsHistory: training.previousMetricsHistory,
            logs: training.logs,
            progress: training.progress,
            mode: training.mode,
            start: training.start,
            stop: training.stop,
            activeClient: training.activeClient,
            metrics: training.metrics,
            jobId: training.jobId
        }}>
            {children}
        </SimulationContext.Provider>
    );
};

export const useSimulationContext = () => {
    const context = useContext(SimulationContext);
    if (!context) throw new Error("useSimulationContext must be used within SimulationProvider");
    return context;
};
