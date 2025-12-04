import { Activity } from 'lucide-react';

export const Header = () => (
    <header className="relative z-20 w-full pt-6 px-6 lg:px-10 flex items-center justify-between">
        <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                <Activity className="text-white w-6 h-6" />
            </div>
            <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">PrivacyHealth</h1>
                <p className="text-xs text-slate-400 font-medium tracking-wide uppercase">Federated Protocol Simulator</p>
            </div>
        </div>
        
        <div className="hidden md:flex items-center gap-4">
             <div className="px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-slate-300">
                v1.0.0-MVP
             </div>
        </div>
    </header>
);
