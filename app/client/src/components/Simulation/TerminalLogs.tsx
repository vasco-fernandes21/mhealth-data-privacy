import { useEffect, useRef } from 'react';
import { useSimulationContext } from '../../context/SimulationContext';

export const TerminalLogs = () => {
  const { logs } = useSimulationContext();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex items-center justify-between mb-4 pb-2 border-b border-white/5">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">System Logs</h3>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
          <span className="text-[10px] text-slate-500 font-mono">LIVE</span>
        </div>
      </div>
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto font-mono text-[11px] space-y-2 pr-2 custom-scrollbar"
      >
          {logs.length === 0 && (
            <div className="text-slate-600 italic flex items-center gap-2">
              <span className="animate-pulse">●</span>
              <span>System ready. Waiting for protocol initialization...</span>
            </div>
          )}
          {logs.map((log, idx) => (
              <div key={idx} className="flex gap-3 animate-fade-in group">
                  <span className={
                      log.includes("Error") || log.includes("error") ? "text-red-400" :
                      log.includes("Complete") || log.includes("completed") || log.includes("finished") ? "text-emerald-400" :
                      log.includes("Round") ? "text-cyan-200 font-bold" :
                      log.includes("Client") ? "text-purple-300" :
                      log.includes("Aggregating") ? "text-blue-300" :
                      "text-slate-300"
                  }>
                      {log}
                  </span>
                  {/* Subtle glow on hover */}
                  <div className="absolute left-0 w-1 h-full bg-cyan-500/0 group-hover:bg-cyan-500/20 transition-colors duration-300" />
              </div>
          ))}
      </div>
    </div>
  );
};
