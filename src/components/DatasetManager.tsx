import React from 'react';
import { Database, Check, X, RefreshCw, Sparkles, Filter } from 'lucide-react';
import { TrainingSample, Cache } from '../types';

interface DatasetManagerProps {
  samples: TrainingSample[];
  onToggleSample: (id: string) => void;
  onApplyPreset: (preset: 'java-default' | 'all' | 'holdout-mul' | 'holdout-diag') => void;
  predictFn: (expr: string, strict: boolean) => {
    predicted: number | null;
    rawPredicted: number;
    isKnown: boolean;
    cache: Cache | null;
  };
  onSelectExpr: (expr: string) => void;
  selectedExpr: string;
  step: number;
}

export const DatasetManager: React.FC<DatasetManagerProps> = ({
  samples,
  onToggleSample,
  onApplyPreset,
  predictFn,
  onSelectExpr,
  selectedExpr,
  step,
}) => {
  const trainCount = samples.filter((s) => s.isTrain).length;
  const holdoutCount = samples.length - trainCount;

  // Compute average errors
  let trainErrSum = 0;
  let trainValidCount = 0;
  let testErrSum = 0;
  let testValidCount = 0;

  samples.forEach((s) => {
    const res = predictFn(s.expression, false);
    if (!isNaN(res.rawPredicted)) {
      const err = Math.abs(res.rawPredicted - s.target);
      if (s.isTrain) {
        trainErrSum += err;
        trainValidCount++;
      } else {
        testErrSum += err;
        testValidCount++;
      }
    }
  });

  const avgTrainErr = trainValidCount > 0 ? trainErrSum / trainValidCount : 0;
  const avgTestErr = testValidCount > 0 ? testErrSum / testValidCount : 0;

  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 p-4 sm:p-5 shadow-xs">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-50 text-indigo-600">
            <Database className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-900 text-sm">
              Training & Evaluation Dataset
            </h3>
            <p className="text-[11px] text-slate-500">
              Configure which arithmetic equations are trained vs held-out
            </p>
          </div>
        </div>

        {/* Preset quick selectors */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-slate-400 mr-1 text-[11px]">Presets:</span>
          <button
            onClick={() => onApplyPreset('java-default')}
            className="px-2.5 py-1 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-semibold border border-indigo-200 text-[11px] transition-colors cursor-pointer"
            title="Mirrors TrainMathFile.java (17 trained, 1*2 held-out)"
          >
            Java Default (Hold out 1*2)
          </button>
          <button
            onClick={() => onApplyPreset('all')}
            className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-[11px] font-medium transition-colors cursor-pointer"
          >
            Train All 18
          </button>
          <button
            onClick={() => onApplyPreset('holdout-mul')}
            className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-[11px] font-medium transition-colors cursor-pointer"
          >
            Hold out Multiply
          </button>
        </div>
      </div>

      {/* Dataset stats header */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200/80">
          <span className="text-[10px] text-slate-400 uppercase font-semibold">Training Set</span>
          <div className="text-base font-bold text-slate-900 mt-0.5">{trainCount} samples</div>
        </div>
        <div className="bg-amber-50/60 p-2.5 rounded-xl border border-amber-200/80">
          <span className="text-[10px] text-amber-700 uppercase font-semibold">Held-Out (Test)</span>
          <div className="text-base font-bold text-amber-900 mt-0.5">{holdoutCount} samples</div>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200/80">
          <span className="text-[10px] text-slate-400 uppercase font-semibold">Train Avg Error</span>
          <div className="text-base font-bold text-indigo-700 font-mono mt-0.5">
            {step > 0 ? avgTrainErr.toFixed(4) : '—'}
          </div>
        </div>
        <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200/80">
          <span className="text-[10px] text-slate-400 uppercase font-semibold">Held-Out Avg Error</span>
          <div className="text-base font-bold text-amber-800 font-mono mt-0.5">
            {step > 0 ? avgTestErr.toFixed(4) : '—'}
          </div>
        </div>
      </div>

      {/* Equations Table / Grid */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead>
            <tr className="border-b border-slate-200 text-slate-400 font-semibold uppercase text-[10px]">
              <th className="py-2 px-3">Status</th>
              <th className="py-2 px-3">Expression</th>
              <th className="py-2 px-3">Target</th>
              <th className="py-2 px-3">Model Output (ŷ)</th>
              <th className="py-2 px-3">Abs Error</th>
              <th className="py-2 px-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-mono">
            {samples.map((sample) => {
              const res = predictFn(sample.expression, false);
              const err = Math.abs(res.rawPredicted - sample.target);
              const isSelected = selectedExpr === sample.expression;

              return (
                <tr
                  key={sample.id}
                  onClick={() => onSelectExpr(sample.expression)}
                  className={`cursor-pointer transition-colors ${
                    isSelected ? 'bg-indigo-50/70 font-bold' : 'hover:bg-slate-50/80'
                  }`}
                >
                  <td className="py-2 px-3 font-sans">
                    {sample.isTrain ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-700 text-[10px] font-semibold border border-emerald-200">
                        <Check className="w-3 h-3" />
                        <span>Train</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-amber-50 text-amber-800 text-[10px] font-semibold border border-amber-200">
                        <X className="w-3 h-3" />
                        <span>Held-out</span>
                      </span>
                    )}
                  </td>
                  <td className="py-2 px-3 font-bold text-slate-900">{sample.expression}</td>
                  <td className="py-2 px-3 text-slate-600">{sample.target}</td>
                  <td className="py-2 px-3 font-bold text-indigo-700">
                    {step > 0 ? res.rawPredicted.toFixed(4) : '—'}
                  </td>
                  <td className="py-2 px-3">
                    {step > 0 ? (
                      <span
                        className={
                          err < 0.05
                            ? 'text-emerald-600 font-semibold'
                            : err < 0.3
                            ? 'text-amber-600'
                            : 'text-rose-600'
                        }
                      >
                        ±{err.toFixed(4)}
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td className="py-2 px-3 text-right font-sans">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleSample(sample.id);
                      }}
                      className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors cursor-pointer ${
                        sample.isTrain
                          ? 'bg-slate-100 hover:bg-rose-50 hover:text-rose-700 text-slate-600'
                          : 'bg-indigo-50 hover:bg-indigo-100 text-indigo-700'
                      }`}
                    >
                      {sample.isTrain ? 'Hold Out' : 'Include in Train'}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
