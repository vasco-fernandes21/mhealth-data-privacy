import { Play, Octagon } from 'lucide-react';
import clsx from 'clsx';

type Props = { config: any; setConfig: any; isRunning: boolean; onStart: any; onStop: any };

export const ConfigPanel = ({ config, setConfig, isRunning, onStart, onStop }: Props) => {
    return (
        <div className="bg-slate-900/60 p-8 h-full flex flex-col justify-between backdrop-blur-md">
            <div>
                <h2 className="text-xl font-light text-white mb-8">
                    Simulation <span className="font-bold text-cyan-400">Parameters</span>
                </h2>
                
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
                                  {ds === 'wesad' ? 'WESAD (Stress)' : 'Sleep-EDF'}
                                </span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Sliders */}
                <div className="space-y-8">
                    <div>
                         <div className="flex justify-between text-xs mb-2">
                             <label className="font-bold text-slate-500 uppercase">Privacy (Sigma)</label>
                             <span className="text-cyan-400 font-mono">{config.sigma.toFixed(1)}</span>
                         </div>
                         <input 
                            type="range" min="0.3" max="2.0" step="0.1"
                            value={config.sigma}
                            disabled={isRunning}
                            onChange={(e) => setConfig({...config, sigma: parseFloat(e.target.value)})}
                            className="w-full h-1 bg-slate-800 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-cyan-500 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(6,182,212,0.5)] [&::-webkit-slider-thumb]:cursor-pointer"
                         />
                         <div className="flex justify-between text-[10px] text-slate-600 mt-1">
                            <span>Less Privacy</span>
                            <span>Strong Privacy</span>
                         </div>
                    </div>

                    <div>
                         <div className="flex justify-between text-xs mb-2">
                             <label className="font-bold text-slate-500 uppercase">Client Nodes</label>
                             <span className="text-purple-400 font-mono">{config.clients}</span>
                         </div>
                         <input 
                            type="range" min="3" max="10"
                            value={config.clients}
                            disabled={isRunning}
                            onChange={(e) => setConfig({...config, clients: parseInt(e.target.value)})}
                            className="w-full h-1 bg-slate-800 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(168,85,247,0.5)] [&::-webkit-slider-thumb]:cursor-pointer"
                        />
                    </div>
                </div>
            </div>

            <button
                onClick={isRunning ? onStop : onStart}
                className={clsx(
                    "w-full mt-8 py-4 rounded-xl flex items-center justify-center gap-3 font-bold tracking-wide transition-all duration-300 group overflow-hidden relative",
                    isRunning 
                    ? "bg-red-500/10 text-red-400 border border-red-500/50 hover:bg-red-500/20" 
                    : "bg-cyan-600 hover:bg-cyan-500 text-white shadow-[0_0_30px_rgba(8,145,178,0.4)] hover:shadow-[0_0_40px_rgba(8,145,178,0.6)]"
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
                      <span>TERMINATE PROTOCOL</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 fill-current" /> 
                      <span>INITIALIZE RUN</span>
                    </>
                  )}
                </span>
            </button>
        </div>
    );
};
