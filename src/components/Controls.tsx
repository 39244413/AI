import React from 'react';
import { Play, Pause, StepForward, RotateCcw, FastForward, Sliders, Zap } from 'lucide-react';

interface ControlsProps {
  isTraining: boolean;
  onToggleTraining: () => void;
  onStep: (count: number) => void;
  onReset: () => void;
  stepsPerFrame: number;
  onChangeStepsPerFrame: (val: number) => void;
  learningRate: number;
  onChangeLearningRate: (val: number) => void;
  isBusy: boolean;
}

export const Controls: React.FC<ControlsProps> = ({
  isTraining,
  onToggleTraining,
  onStep,
  onReset,
  stepsPerFrame,
  onChangeStepsPerFrame,
  learningRate,
  onChangeLearningRate,
  isBusy,
}) => {
  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 p-4 sm:p-5 shadow-xs">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Left: Primary actions */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {/* Main Play/Pause */}
          <button
            id="toggle-train-btn"
            onClick={onToggleTraining}
            disabled={isBusy}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm shadow-xs transition-all cursor-pointer ${
              isTraining
                ? 'bg-amber-600 hover:bg-amber-700 text-white shadow-amber-200'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200'
            } disabled:opacity-50`}
          >
            {isTraining ? (
              <>
                <Pause className="w-4 h-4 fill-current" />
                <span>Pause Training</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Start Training</span>
              </>
            )}
          </button>

          {/* Single Step */}
          <button
            id="step-1-btn"
            onClick={() => onStep(1)}
            disabled={isTraining || isBusy}
            className="flex items-center gap-1.5 px-3.5 py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-800 font-medium text-xs border border-slate-200 transition-colors cursor-pointer disabled:opacity-40"
            title="Train exactly 1 step on a random sample"
          >
            <StepForward className="w-4 h-4" />
            <span>Step (1)</span>
          </button>

          {/* Fast Train Batches */}
          <div className="flex items-center rounded-xl bg-slate-100 border border-slate-200 p-0.5">
            <button
              id="step-100-btn"
              onClick={() => onStep(100)}
              disabled={isTraining || isBusy}
              className="px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:text-slate-950 hover:bg-white rounded-lg transition-all cursor-pointer disabled:opacity-40"
            >
              +100
            </button>
            <button
              id="step-1k-btn"
              onClick={() => onStep(1000)}
              disabled={isTraining || isBusy}
              className="px-2.5 py-1.5 text-xs font-semibold text-indigo-700 hover:text-indigo-900 hover:bg-white rounded-lg transition-all cursor-pointer disabled:opacity-40"
            >
              +1,000
            </button>
            <button
              id="step-10k-btn"
              onClick={() => onStep(10000)}
              disabled={isTraining || isBusy}
              className="px-2.5 py-1.5 text-xs font-semibold text-indigo-700 hover:text-indigo-900 hover:bg-white rounded-lg transition-all cursor-pointer disabled:opacity-40"
            >
              +10,000
            </button>
            <button
              id="step-50k-btn"
              onClick={() => onStep(50000)}
              disabled={isTraining || isBusy}
              className="px-2.5 py-1.5 text-xs font-bold text-indigo-800 hover:bg-indigo-50 rounded-lg transition-all cursor-pointer disabled:opacity-40"
            >
              +50,000
            </button>
          </div>

          {/* Reset button */}
          <button
            id="reset-weights-btn"
            onClick={onReset}
            disabled={isTraining || isBusy}
            className="flex items-center gap-1.5 px-3 py-2.5 rounded-xl text-rose-700 hover:bg-rose-50 border border-rose-200 font-medium text-xs transition-colors cursor-pointer disabled:opacity-40 ml-auto sm:ml-0"
            title="Reset all weights (Wq, Wk, Wv, Wo) and Adam moments"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Weights</span>
          </button>
        </div>

        {/* Right: Sliders (Speed & Learning Rate) */}
        <div className="flex flex-wrap items-center gap-4 sm:gap-6 pt-3 lg:pt-0 border-t lg:border-t-0 border-slate-100">
          {/* Speed / Batch per frame */}
          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-500 flex items-center gap-1 whitespace-nowrap">
              <Zap className="w-3.5 h-3.5 text-amber-500" />
              <span>Speed:</span>
            </span>
            <input
              id="speed-slider"
              type="range"
              min="1"
              max="200"
              step="5"
              value={stepsPerFrame}
              onChange={(e) => onChangeStepsPerFrame(Number(e.target.value))}
              className="w-24 sm:w-28 accent-indigo-600 cursor-pointer"
            />
            <span className="font-mono text-slate-700 font-semibold w-14 text-right">
              {stepsPerFrame}x/f
            </span>
          </div>

          {/* Learning Rate */}
          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-500 flex items-center gap-1 whitespace-nowrap">
              <Sliders className="w-3.5 h-3.5 text-indigo-500" />
              <span>LR (η):</span>
            </span>
            <input
              id="lr-slider"
              type="range"
              min="0.0001"
              max="0.005"
              step="0.0001"
              value={learningRate}
              onChange={(e) => onChangeLearningRate(Number(e.target.value))}
              className="w-24 sm:w-28 accent-indigo-600 cursor-pointer"
            />
            <span className="font-mono text-slate-700 font-semibold w-16 text-right">
              {learningRate.toFixed(4)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
