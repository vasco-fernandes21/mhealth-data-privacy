import { Laptop, Smartphone, Tablet } from 'lucide-react';
import clsx from 'clsx';

type ClientGridProps = {
    totalClients: number;
    activeClient: number | null;
};

const getIcon = (index: number) => {
    // Just for visual variety
    if (index % 3 === 0) return <Smartphone className="w-6 h-6" />;
    if (index % 3 === 1) return <Tablet className="w-6 h-6" />;
    return <Laptop className="w-6 h-6" />;
};

export const ClientGrid = ({ totalClients, activeClient }: ClientGridProps) => {
    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <h2 className="text-lg font-semibold mb-4">Federated Network</h2>
            <div className="grid grid-cols-5 gap-4">
                {Array.from({ length: totalClients }).map((_, i) => {
                    const isActive = activeClient === i;
                    return (
                        <div
                            key={i}
                            className={clsx(
                                "relative aspect-square rounded-xl flex flex-col items-center justify-center border-2 transition-all duration-300",
                                isActive
                                    ? "border-blue-500 bg-blue-50 scale-105 shadow-md"
                                    : "border-gray-100 bg-gray-50 text-gray-400"
                            )}
                        >
                            <div className={clsx(
                                "mb-1 transition-colors",
                                isActive ? "text-blue-600" : "text-gray-400"
                            )}>
                                {getIcon(i)}
                            </div>
                            <span className={clsx(
                                "text-xs font-medium",
                                isActive ? "text-blue-700" : "text-gray-400"
                            )}>
                                Client {i + 1}
                            </span>
                            {isActive && (
                                <span className="absolute top-1 right-1 w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                            )}
                        </div>
                    );
                })}
            </div>
            
            {/* Legend */}
            <div className="mt-6 flex items-center justify-center gap-6 text-xs text-gray-500">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-gray-200 bg-gray-50 rounded" />
                    <span>Idle</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-blue-500 bg-blue-50 rounded" />
                    <span>Training Local Model</span>
                </div>
            </div>
        </div>
    );
};

