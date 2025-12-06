import {
  Play,
  Octagon,
  Cpu,
  Globe,
  Lock,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  Dna,
  Scissors,
  Scale,
} from 'lucide-react';
import clsx from 'clsx';
import { useState } from 'react';

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
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleNumChange = (field: string, val: string, isFloat = false) => {
    const parsed = isFloat ? Number(val) : Number.parseInt(val, 10);
    setConfig((prev: any) => {
      const current = prev?.[field];
      const num = Number.isNaN(parsed) ? current ?? 0 : parsed;
      return { ...prev, [field]: num };
    });
  };

  return (
    <div className="bg-slate-900/60 p-8 h-full flex flex-col justify-between backdrop-blur-md rounded-l-3xl border-r border-white/5 overflow-y-auto scrollbar-hide">
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h2 className="text-xl font-light text-white mb-2">
            Simulation <span className="font-bold text-cyan-400">Lab</span>
          </h2>
          <p className="text-[10px] text-slate-500 uppercase tracking-widest">
            Privacy-Utility Tradeoff Analysis
          </p>
        </div>

        {/* Active Mode Card */}
        <div className="p-3 rounded-xl bg-slate-950 border border-white/10 flex items-center gap-3 animate-fade-in shadow-inner">
          <div className={`p-2 rounded-lg bg-white/5 ${mode.color}`}>
            <ModeIcon className="w-5 h-5" />
          </div>
          <div>
            <p className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">Architecture</p>
            <p className={`text-xs font-bold ${mode.color}`}>{mode.label}</p>
          </div>
        </div>

        {/* Main Controls (Dataset, Sigma, Clients) */}
        <div className="space-y-6">
          {/* Dataset */}
          <div className="grid grid-cols-2 gap-2 bg-slate-950 p-1 rounded-xl border border-white/5">
            {['wesad', 'sleep-edf'].map((ds) => (
              <button
                key={ds}
                disabled={isRunning}
                onClick={() => setConfig({ ...config, dataset: ds })}
                className={clsx(
                  'py-2.5 rounded-lg text-[11px] font-bold transition-all relative overflow-hidden',
                  config.dataset === ds
                    ? 'bg-slate-800 text-white shadow-md border border-white/10'
                    : 'text-slate-500 hover:text-slate-300',
                )}
              >
                {config.dataset === ds && (
                  <div className="absolute inset-0 bg-cyan-500/5 animate-pulse" />
                )}
                <span className="relative z-10">{ds.toUpperCase()}</span>
              </button>
            ))}
          </div>

          {/* Sigma Slider */}
          <div>
            <div className="flex justify-between text-xs mb-2">
              <label className="font-bold text-slate-400">Noise Multiplier (σ)</label>
              <span className={config.sigma > 0 ? 'text-purple-400 font-mono' : 'text-slate-600'}>
                {config.sigma.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="2.0"
              step="0.1"
              value={config.sigma}
              disabled={isRunning}
              onChange={(e) => handleNumChange('sigma', e.target.value, true)}
              className="w-full h-1.5 bg-slate-800 rounded-full appearance-none cursor-pointer accent-purple-500"
            />
          </div>

          {/* Clients Slider */}
          <div>
            <div className="flex justify-between text-xs mb-2">
              <label className="font-bold text-slate-400">Federated Clients</label>
              <span className={config.clients > 0 ? 'text-cyan-400 font-mono' : 'text-slate-600'}>
                {config.clients}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="10"
              step="1"
              value={config.clients}
              disabled={isRunning}
              onChange={(e) => handleNumChange('clients', e.target.value)}
              className="w-full h-1.5 bg-slate-800 rounded-full appearance-none cursor-pointer accent-cyan-500"
            />
          </div>
        </div>

        {/* Advanced / Scientific Controls */}
        <div className="pt-4 border-t border-white/5">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest hover:text-cyan-400 transition-colors w-full mb-3"
          >
            {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            Experimental Variables
          </button>

          <div
            className={clsx(
              'space-y-4 overflow-hidden transition-all duration-500 ease-in-out',
              showAdvanced ? 'max-h-60 opacity-100' : 'max-h-0 opacity-0',
            )}
          >
            {/* 1. Clipping Norm (C) */}
            <div className="bg-slate-800/30 p-3 rounded-lg border border-white/5">
              <div className="flex justify-between items-center mb-2">
                <label className="text-[10px] text-slate-400 flex items-center gap-1.5">
                  <Scissors size={12} className="text-amber-500" /> Max Grad Norm (C)
                </label>
                <span className="text-[10px] font-mono text-amber-500">
                  {(config.max_grad_norm ?? 5.0).toFixed(1)}
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="5.0"
                step="0.1"
                value={config.max_grad_norm ?? 5.0}
                disabled={isRunning}
                onChange={(e) => handleNumChange('max_grad_norm', e.target.value, true)}
                className="w-full h-1 bg-slate-700 rounded-full appearance-none cursor-pointer accent-amber-500"
              />
            </div>

            {/* 2. Seed & Class Weights */}
            <div className="grid grid-cols-2 gap-3">
              {/* Seed Input */}
              <div className="bg-slate-800/30 p-3 rounded-lg border border-white/5">
                <label className="text-[10px] text-slate-400 flex items-center gap-1.5 mb-2">
                  <Dna size={12} className="text-pink-500" /> Seed
                </label>
                <input
                  type="number"
                  value={config.seed ?? ''}
                  placeholder="Rand"
                  disabled={isRunning}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      seed: e.target.value ? parseInt(e.target.value, 10) : null,
                    })
                  }
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono focus:border-pink-500 outline-none"
                />
              </div>

              {/* Class Weights Toggle */}
              <div
                onClick={() =>
                  !isRunning &&
                  setConfig({ ...config, use_class_weights: !config.use_class_weights })
                }
                className={clsx(
                  'bg-slate-800/30 p-3 rounded-lg border cursor-pointer transition-all flex flex-col justify-center items-center gap-1',
                  config.use_class_weights
                    ? 'border-emerald-500/30 bg-emerald-500/5'
                    : 'border-white/5 opacity-60',
                )}
              >
                <Scale
                  size={14}
                  className={config.use_class_weights ? 'text-emerald-400' : 'text-slate-500'}
                />
                <span
                  className={clsx(
                    'text-[9px] font-bold uppercase',
                    config.use_class_weights ? 'text-emerald-400' : 'text-slate-500',
                  )}
                >
                  Weights {config.use_class_weights ? 'ON' : 'OFF'}
                </span>
              </div>
            </div>
            <div className="bg-slate-800/30 p-3 rounded-lg border border-white/5">
              <div className="flex justify-between items-center mb-2">
                <label className="text-[10px] text-slate-400 flex items-center gap-1.5">
                  <span className="text-blue-400 font-bold">N</span> Experimental Runs
                </label>
                <span className="text-[10px] font-mono text-blue-400">
                  {(config.runs ?? 1).toString()}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={config.runs ?? 1}
                disabled={isRunning}
                onChange={(e) => handleNumChange('runs', e.target.value)}
                className="w-full h-1 bg-slate-700 rounded-full appearance-none cursor-pointer accent-blue-500"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Action Button */}
      <button
        onClick={isRunning ? onStop : onStart}
        className={clsx(
          'w-full mt-6 py-4 rounded-xl flex items-center justify-center gap-3 font-bold tracking-wide transition-all duration-300 group overflow-hidden relative border shadow-lg',
          isRunning
            ? 'bg-red-500/10 text-red-400 border-red-500/50 hover:bg-red-500/20'
            : 'bg-cyan-600 hover:bg-cyan-500 text-white border-cyan-400/50 shadow-cyan-500/20',
        )}
      >
        {!isRunning && (
          <>
            <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-in-out" />
          </>
        )}

        <span className="relative z-10 flex items-center gap-2">
          {isRunning ? (
            <>
              <Octagon className="w-5 h-5" /> TERMINATE
            </>
          ) : (
            <>
              <Play className="w-5 h-5 fill-current" /> RUN SIMULATION
            </>
          )}
        </span>
      </button>
    </div>
  );
};
