import { LucideIcon } from 'lucide-react';

type Props = {
    title: string;
    value: string | number;
    icon: LucideIcon;
    trend?: 'up' | 'down' | 'neutral';
    color?: string;
};

export const StatCard = ({ title, value, icon: Icon, color = "text-slate-900" }: Props) => (
    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
        <div>
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">{title}</p>
            <h4 className={`text-2xl font-bold mt-1 ${color}`}>{value}</h4>
        </div>
        <div className="p-3 bg-slate-50 rounded-lg">
            <Icon className="w-5 h-5 text-slate-400" />
        </div>
    </div>
);

