import { createContext, useContext, ReactNode } from 'react';
import { useSimulation, Metrics } from '../hooks/useSimulation';

type SimContextType = {
    status: 'idle' | 'running' | 'finished';
    currentRound: number;
    activeClient: number | null;
    metricsHistory: Metrics[];
    logs: any[];
    history: any[];
    start: (config: any) => void;
    stop: () => void;
};

const SimulationContext = createContext<SimContextType | null>(null);

export const SimulationProvider = ({ children }: { children: ReactNode }) => {
    const simulation = useSimulation();
    
    return (
        <SimulationContext.Provider value={simulation}>
            {children}
        </SimulationContext.Provider>
    );
};

export const useSimulationContext = () => {
    const context = useContext(SimulationContext);
    if (!context) throw new Error("useSimulationContext must be used within SimulationProvider");
    return context;
};

