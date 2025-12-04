import { useCallback, useEffect, useState } from 'react';
import {
  Download,
  ChevronLeft,
  ChevronRight,
  History,
  Trash2,
  CheckCircle2,
  Circle,
} from 'lucide-react';
import clsx from 'clsx';
import { getHistory, deleteHistoryRun, exportRun } from '../../api/client';
import { useSimulationContext } from '../../context/SimulationContext';

type HistoryRun = {
  id: string;
  dataset: string;
  mode: string;
  sigma: number;
  clients: number;
  runs?: number;
  final_accuracy: number;
  final_epsilon: number;
  created_at: string;
};

type SidebarProps = {
  onCollapsedChange?: (collapsed: boolean) => void;
};

export const Sidebar = ({ onCollapsedChange }: SidebarProps) => {
  const { status } = useSimulationContext();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const fetchHistory = useCallback(async () => {
    try {
      const data = await getHistory(50);
      setRuns(data.runs || []);
    } catch (e) {
      console.error('Failed to load history', e);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  useEffect(() => {
    if (status === 'completed' || status === 'failed') {
      fetchHistory();
    }
  }, [status, fetchHistory]);

  const toggleSelection = (id: string) => {
    const next = new Set(selectedIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    setSelectedIds(next);
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.size === 0) return;
    if (!confirm(`Delete ${selectedIds.size} run(s)?`)) return;

    for (const id of selectedIds) {
      try {
        await deleteHistoryRun(id);
      } catch (e) {
        console.error(`Failed to delete run ${id}`, e);
      }
    }
    setRuns((prev) => prev.filter((r) => !selectedIds.has(r.id)));
    setSelectedIds(new Set());
  };

  const handleExportSelected = async () => {
    if (selectedIds.size === 0) return;

    for (const id of selectedIds) {
      try {
        const run = runs.find((r) => r.id === id);
        if (!run) continue;

        const data = await exportRun(id);
        const filename = `exp_${run.dataset}_${run.mode}_${new Date(run.created_at)
          .toISOString()
          .split('T')[0]}.json`;
        const blob = new Blob([JSON.stringify(data, null, 2)], {
          type: 'application/json',
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      } catch (e) {
        console.error('Failed to export', e);
      }
    }
  };

  const formatMode = (mode: string) => {
    if (mode.includes('DP') && mode.includes('FEDERATED')) return 'FL+DP';
    if (mode.includes('FEDERATED')) return 'FL';
    if (mode.includes('DP')) return 'DP';
    return 'BASE';
  };

  return (
    <aside
      className={clsx(
        'fixed left-0 top-0 h-screen pt-20 z-20 hidden md:flex flex-col',
        'bg-slate-900/90 backdrop-blur-xl border-r border-white/10 shadow-2xl',
        'transition-all duration-300 ease-in-out',
        isCollapsed ? 'w-20' : 'w-80',
      )}
    >
      <button
        onClick={() => {
          const next = !isCollapsed;
          setIsCollapsed(next);
          if (onCollapsedChange) onCollapsedChange(next);
        }}
        className="absolute -right-3 top-24 bg-slate-800 border border-white/20 text-slate-400 p-1.5 rounded-full hover:text-white hover:bg-cyan-600 hover:border-cyan-400 transition-all shadow-lg z-30 group"
      >
        {isCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>

      <div className="flex-1 overflow-hidden flex flex-col p-4">
        <div
          className={clsx(
            'flex items-center justify-between mb-4 transition-all duration-300 h-8',
            isCollapsed ? 'justify-center' : 'px-1',
          )}
        >
          {isCollapsed ? (
            <History className="text-slate-500 w-5 h-5" />
          ) : (
            <>
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                <History className="w-4 h-4" /> Experiment History
              </h3>
              <span className="text-[10px] text-slate-600 font-mono">
                {runs.length} Runs
              </span>
            </>
          )}
        </div>

        <div className="space-y-3 overflow-y-auto flex-1 custom-scrollbar pr-1">
          {runs.length === 0 ? (
            !isCollapsed && (
              <div className="text-xs text-slate-600 italic px-2 text-center mt-20 flex flex-col items-center gap-2">
                <div className="w-12 h-1 bg-slate-800 rounded-full mb-2"></div>
                No experiments recorded yet.
              </div>
            )
          ) : (
            runs.map((run) => {
              const isSelected = selectedIds.has(run.id);
              const modeShort = formatMode(run.mode);

              return (
                <div
                  key={run.id}
                  onClick={() => toggleSelection(run.id)}
                  className={clsx(
                    'relative rounded-xl border transition-all cursor-pointer group overflow-hidden select-none',
                    isSelected
                      ? 'bg-cyan-900/20 border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.1)]'
                      : 'bg-slate-800/40 border-white/5 hover:bg-slate-800/60 hover:border-white/10',
                    isCollapsed ? 'p-2 flex flex-col items-center gap-1' : 'p-3',
                  )}
                >
                  {!isCollapsed && (
                    <div className="absolute top-3 right-3 text-slate-600 transition-colors">
                      {isSelected ? (
                        <CheckCircle2 className="w-4 h-4 text-cyan-400 fill-cyan-900/20" />
                      ) : (
                        <Circle className="w-4 h-4 group-hover:text-slate-500" />
                      )}
                    </div>
                  )}

                  {isCollapsed ? (
                    <>
                      <div
                        className={clsx(
                          'w-2 h-2 rounded-full mb-1',
                          run.dataset === 'wesad' ? 'bg-blue-500' : 'bg-purple-500',
                        )}
                      />
                      <span
                        className={clsx(
                          'text-[9px] font-mono',
                          isSelected ? 'text-cyan-300' : 'text-slate-400',
                        )}
                      >
                        {(run.final_accuracy * 100).toFixed(0)}%
                      </span>
                      {isSelected && (
                        <div className="w-1 h-1 bg-cyan-400 rounded-full mt-1" />
                      )}
                    </>
                  ) : (
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-2">
                        <div
                          className={clsx(
                            'w-1.5 h-1.5 rounded-full shadow-[0_0_8px_currentColor]',
                            run.dataset === 'wesad'
                              ? 'bg-blue-500 text-blue-500'
                              : 'bg-purple-500 text-purple-500',
                          )}
                        />
                        <span className="text-xs font-bold text-slate-200 tracking-tight">
                          {run.dataset.toUpperCase()}
                        </span>
                        <span
                          className={clsx(
                            'text-[9px] px-1.5 py-0.5 rounded border font-mono font-bold ml-auto mr-6',
                            modeShort === 'FL+DP'
                              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                              : modeShort === 'DP'
                              ? 'bg-purple-500/10 text-purple-400 border-purple-500/20'
                              : modeShort === 'FL'
                              ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
                              : 'bg-slate-700/50 text-slate-400 border-slate-600',
                          )}
                        >
                          {modeShort}
                        </span>
                      </div>

                      <div className="grid grid-cols-3 gap-2 mt-1">
                        <div className="flex flex-col">
                          <span className="text-[9px] text-slate-500 uppercase">
                            Acc
                          </span>
                          <span
                            className={clsx(
                              'text-sm font-mono font-medium',
                              isSelected ? 'text-white' : 'text-slate-300',
                            )}
                          >
                            {(run.final_accuracy * 100).toFixed(1)}
                            <span className="text-[10px]">%</span>
                          </span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[9px] text-slate-500 uppercase">
                            Eps
                          </span>
                          <span className="text-sm font-mono font-medium text-slate-300">
                            {run.final_epsilon > 0
                              ? run.final_epsilon.toFixed(2)
                              : '-'}
                          </span>
                        </div>
                        <div className="flex flex-col items-end">
                          <span className="text-[9px] text-slate-500 uppercase">
                            Sigma
                          </span>
                          <span className="text-sm font-mono font-medium text-slate-300">
                            {run.sigma}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center justify-between pt-2 border-t border-white/5 mt-1">
                        <span className="text-[9px] text-slate-600 font-mono">
                          {new Date(run.created_at).toLocaleDateString()} •{' '}
                          {new Date(run.created_at).toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </span>
                        {run.clients > 0 && (
                          <span className="text-[9px] text-slate-500 flex items-center gap-1 bg-slate-900/50 px-1.5 rounded">
                            {run.clients} Nodes
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {!isCollapsed && (
        <div className="p-4 border-t border-white/10 bg-slate-900/80 space-y-2">
          {selectedIds.size > 0 ? (
            <div className="space-y-2">
              <div className="flex justify-between items-center mb-2">
                <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">
                  {selectedIds.size} Selected
                </span>
                <button
                  onClick={() => setSelectedIds(new Set())}
                  className="text-[10px] text-slate-500 hover:text-slate-300 underline decoration-dotted"
                >
                  Clear
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={handleExportSelected}
                  className="py-2 rounded-lg bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/30 text-cyan-200 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export
                </button>
                <button
                  onClick={handleDeleteSelected}
                  className="py-2 rounded-lg bg-red-600/10 hover:bg-red-600/20 border border-red-500/20 text-red-300 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Delete
                </button>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-10 text-xs text-slate-600 italic">
              Select runs to manage
            </div>
          )}
        </div>
      )}
    </aside>
  );
};
