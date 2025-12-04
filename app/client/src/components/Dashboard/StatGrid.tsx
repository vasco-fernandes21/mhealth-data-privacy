import { ShieldCheck, Target, Users, Zap } from 'lucide-react';

export const StatGrid = ({ metrics, currentRound }: { metrics: any[], currentRound?: number }) => {
    const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
    
    // Helper for formatting
    const Stat = ({ label, value, color, icon: Icon }: any) => (
        <div className="bg-slate-900/50 border border-white/5 p-4 rounded-2xl flex items-center justify-between hover:border-white/10 transition-all duration-300 group">
            <div>
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{label}</p>
                <p className={`text-xl font-mono mt-1 ${color} text-glow group-hover:scale-105 transition-transform duration-300`}>
                  {value || '--'}
                </p>
            </div>
            <div className={`p-2 rounded-lg bg-white/5 ${color} opacity-50 group-hover:opacity-100 transition-opacity duration-300`}>
                <Icon className="w-5 h-5" />
            </div>
        </div>
    );

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {currentRound !== undefined && currentRound > 0 && (
                <Stat 
                    label="Round" 
                    value={currentRound.toString()} 
                    color="text-slate-300" 
                    icon={Zap}
                />
            )}
            <Stat 
                label="Global Acc" 
                value={latest ? (latest.accuracy * 100).toFixed(1) + '%' : null} 
                color="text-cyan-400" 
                icon={Target}
            />
            <Stat 
                label="Privacy Budget" 
                value={latest ? latest.epsilon?.toFixed(2) || '--' : '--'} 
                color="text-purple-400" 
                icon={ShieldCheck}
            />
             <Stat 
                label="Fairness" 
                value={latest ? (latest.minority_recall * 100).toFixed(1) + '%' : null} 
                color="text-emerald-400" 
                icon={Users}
            />
        </div>
    );
};

