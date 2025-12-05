import { useState } from 'react';
import { SimulationProvider, useSimulationContext } from './context/SimulationContext';
import { Header } from './components/Layout/Header';
import { Sidebar } from './components/Layout/Sidebar';
import { NetworkViz } from './components/Simulation/NetworkViz';
import { ConfigPanel } from './components/Setup/ConfigPanel';
import { StatGrid } from './components/Dashboard/StatGrid';
import { ChartsGrid } from './components/Dashboard/ChartsGrid';
import { FairnessAlert } from './components/Simulation/FairnessAlert';
import clsx from 'clsx';

const DashboardContent = () => {
  const { status, metricsHistory, previousMetricsHistory, metrics, start, stop, currentRound } =
    useSimulationContext();
  const [config, setConfig] = useState({
    dataset: 'wesad',
    clients: 0,
    sigma: 0.0,
    seed: 42,
    max_grad_norm: 5.0,
    use_class_weights: true,
    runs: 1,
  });
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Get latest minority recall for fairness alert
  const latestRecall = metrics?.minority_recall ||
    (metricsHistory.length > 0 ? metricsHistory[metricsHistory.length - 1]?.minority_recall || 0 : 0);

  return (
    <div className="relative min-h-screen w-full overflow-x-hidden bg-slate-950 bg-aurora flex">
      <Sidebar onCollapsedChange={setIsSidebarCollapsed} />

      <div
        className={clsx(
          'flex-1',
          // Alinha conteúdo com a largura atual da sidebar em ecrãs médios+
          isSidebarCollapsed ? 'md:ml-20' : 'md:ml-72',
        )}
      >
        {/* Noise Texture Overlay for texture */}
        <div
          className="fixed inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=%220 0 200 200%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter id=%22noiseFilter%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%220.65%22 numOctaves=%223%22 stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect width=%22100%25%22 height=%22100%25%22 filter=%22url(%23noiseFilter)%22/%3E%3C/svg%3E")'
          }}
        />

        {/* Floating Particles (Original Addition) */}
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="fixed w-1 h-1 bg-cyan-500/30 rounded-full animate-float pointer-events-none"
            style={{
              left: `${10 + i * 12}%`,
              top: `${20 + (i % 3) * 30}%`,
              animationDelay: `${i * 0.5}s`,
              animationDuration: `${15 + i * 2}s`
            }}
          />
        ))}

        <Header />

        <main className="relative z-10 max-w-[1600px] mx-auto p-6 lg:p-10 space-y-8">

          {/* Hero Section: Config + Network Viz */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Control Panel */}
            <div className="lg:col-span-4 flex flex-col gap-6">
              <div className="glass-panel rounded-3xl p-1 overflow-hidden">
                <ConfigPanel
                  config={config}
                  setConfig={setConfig}
                  isRunning={status === 'running'}
                  onStart={() => start(config)}
                  onStop={stop}
                />
              </div>

              {/* Run Stats (Only visible when running/finished) */}
              <div className={`transition-all duration-500 ${(metricsHistory.length > 0 || metrics) ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'}`}>
                <StatGrid metrics={metricsHistory} currentRound={currentRound} />
              </div>
            </div>

            {/* Right Network Visualization */}
            <div className="lg:col-span-8 h-[500px] lg:h-auto relative">
              <NetworkViz config={config} />
              <FairnessAlert recall={latestRecall} />
            </div>
          </div>

          {/* Data Section */}
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
            <div className="xl:col-span-3 glass-panel rounded-3xl p-6 lg:p-8">
              <ChartsGrid metrics={metricsHistory} previousMetrics={previousMetricsHistory} />
            </div>
          </div>

        </main>
      </div>
    </div>
  );
};

export default function App() {
  return (
    <SimulationProvider>
      <DashboardContent />
    </SimulationProvider>
  );
}