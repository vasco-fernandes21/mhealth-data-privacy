import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

interface MetricsChartProps {
    data: Array<{ [key: string]: number }>;
    dataKey: string;
    color: string;
    title: string;
}

export const MetricsChart = ({ data, dataKey, color, title }: MetricsChartProps) => {
    // Transform data to include round numbers
    const chartData = data.map((item, index) => ({
        round: index + 1,
        [dataKey]: item[dataKey]
    }));

    return (
        <div className="bg-white p-4 rounded-lg shadow h-64">
            <h3 className="text-gray-500 text-sm mb-2">{title}</h3>
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis domain={dataKey === 'epsilon' ? [0, 'auto'] : [0, 1]} />
                    <Tooltip />
                    <Line 
                        type="monotone" 
                        dataKey={dataKey} 
                        stroke={color} 
                        strokeWidth={2} 
                        dot={false}
                        isAnimationActive={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

