import { AlertTriangle } from 'lucide-react';

export const FairnessAlert = ({ recall }: { recall: number }) => {
  if (recall > 0.3 || recall === 0) return null;
  
  return (
    <div className="absolute top-4 right-4 bg-amber-500/10 border border-amber-500/50 p-3 rounded-lg flex items-center gap-3 animate-pulse z-50 backdrop-blur-sm">
       <AlertTriangle className="text-amber-500 w-5 h-5" />
       <div>
         <h4 className="text-amber-500 text-xs font-bold uppercase">Fairness Warning</h4>
         <p className="text-amber-200/70 text-[10px]">
           Minority class recall critical ({(recall * 100).toFixed(1)}%)
         </p>
         <p className="text-amber-300/50 text-[9px] mt-1">
           Gradient clipping may be affecting underrepresented classes
         </p>
       </div>
    </div>
  );
};

