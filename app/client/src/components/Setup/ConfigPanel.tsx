import { Play, Octagon, Cpu, Globe, Lock, ShieldAlert } from 'lucide-react';
import clsx from 'clsx';

type Props = { config: any; setConfig: any; isRunning: boolean; onStart: any; onStop: any };

// Helper to determine mode based on config
const getModeLabel = (clients: number, sigma: number) => {
    if (clients === 0 && sigma === 0) return { label: 'BASELINE (Centralized)', color: 'text-gray-400', icon: Cpu };
    if (clients === 0 && sigma > 0) return { label: 'CENTRALIZED DP', color: 'text-purple-400', icon: Lock };
    if (clients > 0 && sigma === 0) return { label: 'FEDERATED LEARNING', color: 'text-cyan-400', icon: Globe };
    return { label: 'FL + DIFFERENTIAL PRIVACY', color: 'text-emerald-400', icon: ShieldAlert };
};

export const ConfigPanel = ({ config, setConfig, isRunning, onStart, onStop }: Props) => {
    const mode = getModeLabel(config.clients, config.sigma);
    const ModeIcon = mode.icon;

    return (
        <div className="bg-slate-900/60 p-8 h-full flex flex-col justify-between backdrop-blur-md rounded-l-3xl border-r border-white/5">
            <div>
                <h2 className="text-xl font-light text-white mb-6">
                    Simulation <span className="font-bold text-cyan-400">Parameters</span>
                </h2>
                
                {/* Active Mode Indicator */}
                <div className="mb-8 p-3 rounded-xl bg-slate-950 border border-white/10 flex items-center gap-3 animate-fade-in">
                    <div className={`p-2 rounded-lg bg-white/5 ${mode.color}`}>
                        <ModeIcon className="w-5 h-5" />
                    </div>
                    <div>
                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Detected Architecture</p>
                        <p className={`text-xs font-bold ${mode.color}`}>{mode.label}</p>
                    </div>
                </div>
                
                {/* Dataset Toggle */}
                <div className="mb-8">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 block">Target Dataset</label>
                    <div className="grid grid-cols-2 gap-2 bg-slate-950 p-1 rounded-xl border border-white/5">
                        {['wesad', 'sleep-edf'].map((ds) => (
                            <button
                                key={ds}
                                disabled={isRunning}
                                onClick={() => setConfig({...config, dataset: ds})}
                                className={clsx(
                                    "py-3 rounded-lg text-xs font-bold transition-all duration-300 relative overflow-hidden",
                                    config.dataset === ds 
                                    ? "bg-slate-800 text-white shadow-lg border border-white/10" 
                                    : "text-slate-500 hover:text-slate-300"
                                )}
                            >
                                {config.dataset === ds && (
                                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 animate-pulse" />
                                )}
                                <span className="relative z-10">
                                  {ds === 'wesad' ? 'WESAD' : 'Sleep-EDF'}
                                </span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Sliders */}
                <div className="space-y-8">
                    {/* Privacy Slider */}
                    <div>
                         <div className="flex justify-between text-xs mb-2">
                             <label className="font-bold text-slate-500 uppercase">Privacy Multiplier</label>
                             <span className={config.sigma > 0 ? "text-purple-400 font-mono" : "text-gray-500"}>
                                 {config.sigma === 0 ? "OFF" : config.sigma.toFixed(1)}
                             </span>
                         </div>
                         <input 
                            type="range" min="0" max="2.0" step="0.1"
                            value={config.sigma}
                            disabled={isRunning}
                            onChange={(e) => setConfig({...config, sigma: parseFloat(e.target.value)})}
                            className="w-full h-1 bg-slate-800 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(168,85,247,0.5)] hover:[&::-webkit-slider-thumb]:scale-125 transition-all"
                         />
                         <div className="flex justify-between text-[10px] text-slate-600 mt-1 font-mono">
                            <span>0 (None)</span>
                            <span>2.0 (High)</span>
                         </div>
                    </div>

                    {/* Clients Slider */}
                    <div>
                         <div className="flex justify-between text-xs mb-2">
                             <label className="font-bold text-slate-500 uppercase">Federated Clients</label>
                             <span className={config.clients > 0 ? "text-cyan-400 font-mono" : "text-gray-500"}>
                                 {config.clients === 0 ? "Centralized" : config.clients}
                             </span>
                         </div>
                         <input 
                            type="range" min="0" max="10" step="1"
                            value={config.clients}
                            disabled={isRunning}
                            onChange={(e) => setConfig({...config, clients: parseInt(e.target.value)})}
                            className="w-full h-1 bg-slate-800 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-cyan-500 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(6,182,212,0.5)] hover:[&::-webkit-slider-thumb]:scale-125 transition-all"
                        />
                        <div className="flex justify-between text-[10px] text-slate-600 mt-1 font-mono">
                            <span>0 (Server)</span>
                            <span>10 Nodes</span>
                         </div>
                    </div>
                </div>
            </div>

            <button
                onClick={isRunning ? onStop : onStart}
                className={clsx(
                    "w-full mt-8 py-4 rounded-xl flex items-center justify-center gap-3 font-bold tracking-wide transition-all duration-300 group overflow-hidden relative border",
                    isRunning 
                    ? "bg-red-500/10 text-red-400 border-red-500/50 hover:bg-red-500/20" 
                    : "bg-cyan-600 hover:bg-cyan-500 text-white border-cyan-400/50 shadow-[0_0_30px_rgba(8,145,178,0.4)]"
                )}
            >
                {/* Button Glow Effect */}
                {!isRunning && (
                  <>
                    <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                    <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/0 via-cyan-400/20 to-cyan-400/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                  </>
                )}
                
                <span className="relative z-10 flex items-center gap-2">
                  {isRunning ? (
                    <>
                      <Octagon className="w-5 h-5" /> 
                      <span>ABORT MISSION</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 fill-current" /> 
                      <span>INITIATE PROTOCOL</span>
                    </>
                  )}
                </span>
            </button>
        </div>
    );
};
