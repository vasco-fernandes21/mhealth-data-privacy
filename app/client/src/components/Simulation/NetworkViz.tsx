import clsx from 'clsx';
import { useSimulationContext } from '../../context/SimulationContext';

export const NetworkViz = ({ config }: { config: any }) => {
  const { activeClient, status, currentRound } = useSimulationContext();
  const clients = config.clients;

  return (
    <div className="w-full h-full min-h-[400px] glass-panel rounded-3xl relative overflow-hidden flex items-center justify-center group">
      
      {/* Decorative Grid Background */}
      <div className="absolute inset-0 opacity-20" 
           style={{ backgroundImage: 'radial-gradient(circle, #64748b 1px, transparent 1px)', backgroundSize: '30px 30px' }}>
      </div>

      {/* Animated Data Streams (Original Addition) */}
      {status === 'running' && (
        <>
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-20 bg-gradient-to-b from-cyan-500/50 to-transparent animate-data-stream"
              style={{
                left: `${20 + i * 30}%`,
                animationDelay: `${i * 2.5}s`,
                animationDuration: `${8 + i * 2}s`
              }}
            />
          ))}
        </>
      )}

      {/* Central Server Node */}
      <div className="relative z-10">
        <div className={clsx(
          "w-24 h-24 rounded-full flex items-center justify-center transition-all duration-700",
          status === 'running' ? "bg-cyan-500/10 shadow-[0_0_50px_rgba(6,182,212,0.2)] animate-hologram" : "bg-slate-800/50"
        )}>
           <div className={clsx(
             "w-12 h-12 rounded-full border-2 transition-all duration-500",
             status === 'running' ? "border-cyan-400 bg-cyan-950 animate-pulse-ring" : "border-slate-600 bg-slate-900"
           )} />
        </div>
        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-3 text-center">
            <p className="text-xs font-bold text-slate-300 uppercase tracking-widest">Global Model</p>
            {status === 'running' && (
              <div className="flex items-center gap-2 justify-center mt-1">
                <span className="text-[10px] text-cyan-400 animate-pulse">AGGREGATING</span>
                <span className="text-[8px] text-slate-500 font-mono">R{currentRound}</span>
              </div>
            )}
        </div>
      </div>

      {/* Orbiting Clients */}
      <div className="absolute inset-0">
         {Array.from({ length: clients }).map((_, i) => {
            const angle = (i / clients) * 2 * Math.PI - Math.PI / 2;
            const radius = 140;
            const x = 50 + (Math.cos(angle) * 35);
            const y = 50 + (Math.sin(angle) * 35);
            
            const isActive = activeClient === i;

            return (
              <div 
                key={i}
                className="absolute w-12 h-12 -ml-6 -mt-6 transition-all duration-500"
                style={{ left: `${x}%`, top: `${y}%` }}
              >
                 {/* Connection Line with animated pulse and encryption indicator */}
                 {isActive && (
                    <div className="absolute top-1/2 left-1/2 origin-left z-0">
                      <div 
                        className="h-[1px] bg-gradient-to-r from-transparent via-amber-400 to-transparent animate-pulse"
                        style={{ 
                          transform: `rotate(${angle + Math.PI}rad)`,
                          width: '150px'
                        }}
                      />
                      {/* Pulse effect along the line (encrypted data) */}
                      <div 
                        className="absolute w-2 h-2 bg-amber-400 rounded-full shadow-[0_0_10px_rgba(251,191,36,0.8)] animate-float"
                        style={{ 
                          transform: `rotate(${angle + Math.PI}rad) translateX(75px)`,
                          animationDelay: `${i * 0.2}s`
                        }}
                      />
                      {/* Lock icon indicator */}
                      <div 
                        className="absolute text-amber-400 text-[8px] animate-pulse"
                        style={{ 
                          transform: `rotate(${angle + Math.PI}rad) translateX(60px) translateY(-8px)`
                        }}
                      >
                        🔒
                      </div>
                    </div>
                 )}

                 {/* Client Node with glow effect */}
                 <div className={clsx(
                    "relative z-10 w-full h-full rounded-xl flex items-center justify-center border transition-all duration-300",
                    isActive 
                        ? "bg-slate-900 border-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.4)] scale-110 animate-hologram" 
                        : "bg-slate-900/40 border-slate-700 backdrop-blur-sm"
                 )}>
                    <div className={clsx(
                        "w-2 h-2 rounded-full transition-all duration-300",
                        isActive ? "bg-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.8)]" : "bg-slate-500"
                    )} />
                    {/* Ripple effect when active */}
                    {isActive && (
                      <div className="absolute inset-0 rounded-xl border border-cyan-400/50 animate-pulse-ring" />
                    )}
                 </div>
                 
                 {/* Label with glow */}
                 <span className={clsx(
                    "absolute -bottom-6 left-1/2 -translate-x-1/2 text-[10px] font-mono whitespace-nowrap transition-colors",
                    isActive ? "text-cyan-300 text-glow" : "text-slate-600"
                 )}>
                    ID: {i < 9 ? `0${i+1}` : i+1}
                 </span>
              </div>
            );
         })}
      </div>

      {/* Scan line effect (Original Addition) */}
      {status === 'running' && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="w-full h-[2px] bg-gradient-to-b from-transparent via-cyan-500/30 to-transparent animate-scan-line" />
        </div>
      )}
    </div>
  );
};
