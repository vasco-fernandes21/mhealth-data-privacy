import { Play, Square } from 'lucide-react';

type Props = { config: any; setConfig: any; isRunning: boolean; onStart: any; onStop: any };

export const CompactConfig = ({ config, setConfig, isRunning, onStart, onStop }: Props) => {
    return (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h2 className="text-lg font-bold text-slate-900 mb-6">Experiment Setup</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div>
                    <label className="block text-xs font-semibold text-slate-500 mb-2">TARGET DATASET</label>
                    <div className="flex gap-2">
                        {['wesad', 'sleep-edf'].map((ds) => (
                            <button
                                key={ds}
                                disabled={isRunning}
                                onClick={() => setConfig({...config, dataset: ds})}
                                className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                                    config.dataset === ds 
                                    ? 'bg-slate-900 text-white shadow-md' 
                                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                }`}
                            >
                                {ds === 'wesad' ? 'Stress (WESAD)' : 'Sleep (EDF)'}
                            </button>
                        ))}
                    </div>
                </div>

                <div>
                    <label className="block text-xs font-semibold text-slate-500 mb-2">PRIVACY BUDGET (Sigma: {config.sigma})</label>
                    <input 
                        type="range" min="0.3" max="2.0" step="0.1"
                        value={config.sigma}
                        disabled={isRunning}
                        onChange={(e) => setConfig({...config, sigma: parseFloat(e.target.value)})}
                        className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                        <span>Less Privacy</span>
                        <span>Strong Privacy</span>
                    </div>
                </div>

                <div>
                     <label className="block text-xs font-semibold text-slate-500 mb-2">NETWORK SIZE</label>
                     <div className="flex items-center gap-4">
                        <span className="text-2xl font-bold text-slate-900 w-8">{config.clients}</span>
                        <input 
                            type="range" min="3" max="10"
                            value={config.clients}
                            disabled={isRunning}
                            onChange={(e) => setConfig({...config, clients: parseInt(e.target.value)})}
                            className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-slate-600"
                        />
                     </div>
                </div>
            </div>

            <button
                onClick={isRunning ? onStop : onStart}
                className={`w-full py-4 rounded-xl flex items-center justify-center gap-2 font-bold transition-all ${
                    isRunning 
                    ? 'bg-red-50 text-red-600 hover:bg-red-100 border border-red-200' 
                    : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-200'
                }`}
            >
                {isRunning ? <><Square className="w-5 h-5" /> Stop Simulation</> : <><Play className="w-5 h-5" /> Initialize Federated Protocol</>}
            </button>
        </div>
    );
};

