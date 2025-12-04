import { useState, useEffect } from 'react';
import clsx from 'clsx';
import { useSimulationContext } from '../../context/SimulationContext';

export const NetworkViz = ({ config }: { config: any }) => {
  const { status, mode } = useSimulationContext();
  const [visualActiveClient, setVisualActiveClient] = useState<number | null>(null);
  
  // Simulation Loop: Auto-rotate active client visualization
  useEffect(() => {
    if (status === 'running' && config.clients > 0) {
      const interval = setInterval(() => {
        setVisualActiveClient(prev => {
          if (prev === null) return 0;
          return (prev + 1) % config.clients;
        });
      }, 300); // Speed of visual rotation (300ms per client)
      
      return () => clearInterval(interval);
    } else {
      setVisualActiveClient(null);
    }
  }, [status, config.clients]);

  const clients = config.clients > 0 ? config.clients : 1; // Show at least 1 node for Centralized
  const isFederated = config.clients > 0;

  return (
    <div className="w-full h-full min-h-[400px] glass-panel rounded-3xl relative overflow-hidden flex items-center justify-center group">
      
      {/* Decorative Grid Background */}
      <div className="absolute inset-0 opacity-20" 
           style={{ backgroundImage: 'radial-gradient(circle, #64748b 1px, transparent 1px)', backgroundSize: '30px 30px' }}>
      </div>

      {/* Central Server Node */}
      <div className="relative z-10">
        <div className={clsx(
          "w-24 h-24 rounded-full flex items-center justify-center transition-all duration-700 backdrop-blur-md border",
          status === 'running' 
            ? "bg-cyan-500/10 border-cyan-500/50 shadow-[0_0_50px_rgba(6,182,212,0.3)]" 
            : "bg-slate-800/50 border-slate-700"
        )}>
           <div className={clsx(
             "w-12 h-12 rounded-full border-2 transition-all duration-500 flex items-center justify-center",
             status === 'running' ? "border-cyan-400 bg-cyan-950/80 animate-pulse" : "border-slate-600 bg-slate-900"
           )}>
              {/* Server Icon/Pulse */}
              <div className={clsx("w-4 h-4 rounded-full", status === 'running' ? "bg-cyan-400" : "bg-slate-600")} />
           </div>
        </div>
        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-3 text-center">
            <p className="text-[10px] font-bold text-slate-300 uppercase tracking-widest whitespace-nowrap">Aggregation Server</p>
            {status === 'running' && (
              <span className="text-[9px] text-cyan-400 font-mono animate-pulse block mt-1">
                {isFederated ? "AGGREGATING WEIGHTS" : "TRAINING MODEL"}
              </span>
            )}
        </div>
      </div>

      {/* Orbiting Clients (Only if Federated) */}
      {isFederated && (
        <div className="absolute inset-0">
           {Array.from({ length: clients }).map((_, i) => {
              // Calculate position in a circle
              const angle = (i / clients) * 2 * Math.PI - Math.PI / 2;
              const radius = 130; 
              const x = 50 + (Math.cos(angle) * 30); // Using % to be responsive
              const y = 50 + (Math.sin(angle) * 30);
              
              const isActive = visualActiveClient === i;

              return (
                <div 
                  key={i}
                  className="absolute w-12 h-12 -ml-6 -mt-6 transition-all duration-500"
                  style={{ left: `${x}%`, top: `${y}%` }}
                >
                   {/* Data Stream Line */}
                   {isActive && (
                      <div 
                          className="absolute top-1/2 left-1/2 h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent origin-left z-0"
                          style={{ 
                              transform: `rotate(${angle + Math.PI}rad)`,
                              width: '120px', // Approx distance
                              filter: 'drop-shadow(0 0 5px cyan)'
                          }}
                      />
                   )}

                   {/* Client Node */}
                   <div className={clsx(
                      "relative z-10 w-full h-full rounded-xl flex items-center justify-center border transition-all duration-300",
                      isActive 
                          ? "bg-slate-900 border-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.5)] scale-110" 
                          : "bg-slate-900/60 border-slate-700 backdrop-blur-sm"
                   )}>
                      <div className={clsx(
                          "w-2 h-2 rounded-full transition-colors duration-300",
                          isActive ? "bg-cyan-400 shadow-[0_0_10px_cyan]" : "bg-slate-500"
                      )} />
                   </div>
                   
                   {/* Label */}
                   <span className={clsx(
                      "absolute -bottom-5 left-1/2 -translate-x-1/2 text-[9px] font-mono whitespace-nowrap transition-colors",
                      isActive ? "text-cyan-300 font-bold" : "text-slate-600"
                   )}>
                      NODE {i+1}
                   </span>
                   
                   {/* Lock Icon for DP */}
                   {isActive && mode.includes('PRIVACY') && (
                      <div className="absolute -top-3 -right-2 text-[10px]">🔒</div>
                   )}
                </div>
              );
           })}
        </div>
      )}
    </div>
  );
};
