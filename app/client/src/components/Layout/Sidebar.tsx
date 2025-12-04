import { History, Settings2, Download, Zap } from 'lucide-react';
import { useSimulationContext } from '../../context/SimulationContext';

export const Sidebar = ({ history }: { history: any[] }) => {
  const { metricsHistory } = useSimulationContext();

  const downloadData = () => {
    if (metricsHistory.length === 0) {
      alert('No data to export. Run a simulation first.');
      return;
    }

    const exportData = {
      timestamp: new Date().toISOString(),
      metrics: metricsHistory,
      summary: {
        total_rounds: metricsHistory.length,
        final_accuracy: metricsHistory[metricsHistory.length - 1]?.accuracy || 0,
        final_epsilon: metricsHistory[metricsHistory.length - 1]?.epsilon || 0,
        final_fairness: metricsHistory[metricsHistory.length - 1]?.minority_recall || 0,
      }
    };

    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(exportData, null, 2)
    )}`;
    const link = document.createElement("a");
    link.href = jsonString;
    link.download = `privacyhealth_experiment_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  };

  return (
    <aside className="w-64 bg-slate-900/60 backdrop-blur-xl border-r border-white/10 h-screen flex flex-col fixed left-0 top-0 pt-16 hidden md:flex z-20">
      <div className="p-6 flex-1 overflow-y-auto">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
          Recent Runs
        </h3>
        <div className="space-y-3">
          {history.length === 0 ? (
            <div className="text-sm text-slate-500 italic">No runs yet.</div>
          ) : (
            history.map((run) => (
              <div key={run.id} className="p-3 rounded-lg border border-white/5 bg-slate-800/40 hover:bg-slate-800/60 transition-colors cursor-default animate-fade-in">
                <div className="flex justify-between items-start mb-1">
                  <span className="text-xs font-bold text-slate-200">{run.config.dataset.toUpperCase()}</span>
                  <span className="text-[10px] text-slate-500">{run.date}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                   <span>Acc: {(run.finalMetrics?.accuracy * 100).toFixed(1)}%</span>
                   <span className="text-slate-600">|</span>
                   <span>σ: {run.config.sigma}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
      
      <div className="mt-auto p-6 border-t border-white/10 space-y-4">
        {/* Export Button */}
        <button
          onClick={downloadData}
          disabled={metricsHistory.length === 0}
          className="w-full py-2 px-3 rounded-lg bg-slate-800/60 hover:bg-slate-800/80 border border-white/10 text-slate-300 text-xs font-medium flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Download className="w-4 h-4" />
          Export Experiment Data
        </button>

        <div className="flex items-center gap-2 text-slate-500 text-xs">
          <Settings2 className="w-4 h-4" />
          <span>v1.0.0-MVP</span>
        </div>
      </div>
    </aside>
  );
};
