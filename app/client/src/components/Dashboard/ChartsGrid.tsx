import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900/95 border border-slate-700/50 p-3 rounded-lg shadow-xl text-xs backdrop-blur-sm z-50">
        <p className="text-slate-400 mb-1 font-mono">Round {label}</p>
        {payload.map((entry: any, idx: number) => (
          <p key={idx} className={`font-mono ${entry.dataKey === 'accuracy' ? 'text-cyan-400' : entry.dataKey === 'epsilon' ? 'text-purple-400' : 'text-emerald-400'}`}>
            {entry.dataKey === 'accuracy' ? 'Acc' : entry.dataKey === 'epsilon' ? 'ε' : 'Recall'}: {
              (entry.dataKey === 'accuracy' || entry.dataKey === 'minority_recall') 
                ? (entry.value * 100).toFixed(1) + '%' 
                : entry.value.toFixed(2)
            }
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export const ChartsGrid = ({ metrics }: { metrics: any[] }) => {
  const data = metrics.map((m, i) => ({ round: i + 1, ...m }));

  return (
    <div className="flex flex-col gap-6 w-full">
       <div className="flex items-center justify-between">
           <h3 className="text-sm font-medium text-slate-400 uppercase tracking-widest">Performance Analytics</h3>
           <div className="flex gap-4 text-[10px] text-slate-500">
               <span className="flex items-center gap-1">
                 <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></div> Acc
               </span>
               <span className="flex items-center gap-1">
                 <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse"></div> ε
               </span>
               <span className="flex items-center gap-1">
                 <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div> Fairness
               </span>
           </div>
       </div>

       {/* Main Accuracy Chart - Fixed Height */}
       <div className="w-full h-[220px] bg-slate-900/30 rounded-xl border border-white/5 p-2">
         <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis 
                  dataKey="round" 
                  stroke="#475569" 
                  fontSize={10} 
                  tickLine={false} 
                  axisLine={false}
                />
                <YAxis 
                  stroke="#475569" 
                  fontSize={10} 
                  tickLine={false} 
                  axisLine={false}
                  domain={[0, 1]}
                  tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                />
                <Tooltip content={<CustomTooltip />} cursor={{stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1}} />
                <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#06b6d4" 
                    strokeWidth={3} 
                    dot={false}
                    activeDot={{ r: 6, fill: "#06b6d4", stroke: "#fff" }}
                    animationDuration={500}
                />
            </LineChart>
         </ResponsiveContainer>
       </div>

       {/* Secondary Charts Row - Fixed Grid */}
       <div className="grid grid-cols-2 gap-4 w-full h-[160px]">
         {/* Privacy Loss Chart */}
         <div className="bg-slate-900/30 rounded-xl border border-white/5 p-2 flex flex-col">
           <p className="text-[10px] text-slate-500 mb-1 ml-2 uppercase tracking-wider">Privacy Budget (ε)</p>
           <div className="flex-1 min-h-0">
             <ResponsiveContainer width="100%" height="100%">
               <LineChart data={data}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                 <XAxis dataKey="round" stroke="#475569" fontSize={8} tickLine={false} axisLine={false} />
                 <YAxis stroke="#475569" fontSize={8} tickLine={false} axisLine={false} />
                 <Tooltip content={<CustomTooltip />} cursor={{stroke: 'rgba(255,255,255,0.1)'}} />
                 <Line 
                   type="monotone" 
                   dataKey="epsilon" 
                   stroke="#a855f7" 
                   strokeWidth={2} 
                   dot={false}
                   animationDuration={500}
                 />
               </LineChart>
             </ResponsiveContainer>
           </div>
         </div>

         {/* Fairness Chart */}
         <div className="bg-slate-900/30 rounded-xl border border-white/5 p-2 flex flex-col">
           <p className="text-[10px] text-slate-500 mb-1 ml-2 uppercase tracking-wider">Minority Recall</p>
           <div className="flex-1 min-h-0">
             <ResponsiveContainer width="100%" height="100%">
               <LineChart data={data}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                 <XAxis dataKey="round" stroke="#475569" fontSize={8} tickLine={false} axisLine={false} />
                 <YAxis stroke="#475569" fontSize={8} tickLine={false} axisLine={false} domain={[0, 1]} />
                 <Tooltip content={<CustomTooltip />} cursor={{stroke: 'rgba(255,255,255,0.1)'}} />
                 <Line 
                   type="monotone" 
                   dataKey="minority_recall" 
                   stroke="#10b981" 
                   strokeWidth={2} 
                   dot={false}
                   animationDuration={500}
                 />
               </LineChart>
             </ResponsiveContainer>
           </div>
         </div>
       </div>
    </div>
  );
};
