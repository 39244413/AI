import React from 'react';
import { Cpu, BookOpen, Activity, CheckCircle2, RefreshCw } from 'lucide-react';

interface HeaderProps {
  step: number;
  currentLoss: number;
  bestLoss: number;
  isTraining: boolean;
  trainingSamplesCount: number;
  onOpenCodeModal: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  step,
  currentLoss,
  bestLoss,
  isTraining,
  trainingSamplesCount,
  onOpenCodeModal,
}) => {
  const isConverged = step > 1000 && currentLoss < 0.05;

  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-30 px-4 lg:px-8 py-3.5 shadow-xs">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        {/* Left: Brand and Architecture badge */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 via-blue-600 to-indigo-700 flex items-center justify-center text-white shadow-sm shadow-indigo-200">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-slate-900 tracking-tight">
                Transformer Math Lab
              </h1>
              <span className="text-[11px] font-semibold uppercase px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200/60 tracking-wider">
                TrainMathFile.java
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Pure self-attention transformer neural network with manual backpropagation & Adam optimizer
            </p>
          </div>
        </div>

        {/* Right: Live stats & Java code launcher */}
        <div className="flex items-center flex-wrap gap-2.5 sm:gap-3">
          {/* Status badge */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-100 border border-slate-200 text-xs font-medium text-slate-700">
            {isTraining ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span>Training</span>
              </>
            ) : isConverged ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-700 font-semibold">Converged</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-slate-400"></span>
                <span>Paused</span>
              </>
            )}
          </div>

          {/* Steps counter */}
          <div className="px-3 py-1 rounded-lg bg-slate-50 border border-slate-200/80 text-xs">
            <span className="text-slate-400 mr-1.5 font-mono">Step:</span>
            <span className="font-mono font-bold text-slate-900">{step.toLocaleString()}</span>
          </div>

          {/* Loss indicator */}
          <div className="px-3 py-1 rounded-lg bg-slate-50 border border-slate-200/80 text-xs">
            <span className="text-slate-400 mr-1.5 font-mono">MSE Loss:</span>
            <span
              className={`font-mono font-bold ${
                currentLoss < 0.05
                  ? 'text-emerald-600'
                  : currentLoss < 0.5
                  ? 'text-amber-600'
                  : 'text-slate-900'
              }`}
            >
              {currentLoss === 0 && step === 0 ? '0.0000' : currentLoss.toFixed(4)}
            </span>
          </div>

          {/* Dataset info */}
          <div className="hidden sm:flex px-3 py-1 rounded-lg bg-slate-50 border border-slate-200/80 text-xs text-slate-600">
            <span className="text-slate-400 mr-1.5">Train Set:</span>
            <span className="font-semibold text-slate-800">{trainingSamplesCount} eqns</span>
          </div>

          {/* View Java Source button */}
          <button
            id="view-java-code-btn"
            onClick={onOpenCodeModal}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold shadow-xs transition-colors cursor-pointer"
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Java Source & Math</span>
          </button>
        </div>
      </div>
    </header>
  );
};
