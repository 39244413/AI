import React, { useState } from 'react';
import { Calculator, AlertCircle, CheckCircle2, ShieldAlert, Sparkles, HelpCircle } from 'lucide-react';
import { TrainingSample, Cache } from '../types';

interface InferenceTesterProps {
  samples: TrainingSample[];
  selectedExpr: string;
  onSelectExpr: (expr: string) => void;
  predictFn: (expr: string, strict: boolean) => {
    predicted: number | null;
    rawPredicted: number;
    isKnown: boolean;
    cache: Cache | null;
  };
  step: number;
}

export const InferenceTester: React.FC<InferenceTesterProps> = ({
  samples,
  selectedExpr,
  onSelectExpr,
  predictFn,
  step,
}) => {
  const [strictJavaMode, setStrictJavaMode] = useState<boolean>(true);
  const [customInput, setCustomInput] = useState<string>('');

  const result = predictFn(selectedExpr, strictJavaMode);

  // Compute actual ground truth for selected expression
  let actualValue = 0;
  if (selectedExpr.includes('+')) {
    const [a, b] = selectedExpr.split('+').map(Number);
    actualValue = a + b;
  } else if (selectedExpr.includes('*')) {
    const [a, b] = selectedExpr.split('*').map(Number);
    actualValue = a * b;
  }

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const clean = customInput.trim();
    if (clean) {
      onSelectExpr(clean);
      setCustomInput('');
    }
  };

  const error = result.predicted !== null ? Math.abs(result.predicted - actualValue) : null;
  const rawError = !isNaN(result.rawPredicted) ? Math.abs(result.rawPredicted - actualValue) : null;

  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 p-4 sm:p-5 shadow-xs flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-50 text-indigo-600">
            <Calculator className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-900 text-sm">Model Inference & Testing</h3>
            <p className="text-[11px] text-slate-500">
              Evaluates expression through forward pass finalOut[2][0]
            </p>
          </div>
        </div>

        {/* Strict Java mode toggle */}
        <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer select-none bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200 hover:bg-slate-100 transition-colors">
          <input
            id="strict-java-mode-toggle"
            type="checkbox"
            checked={strictJavaMode}
            onChange={(e) => setStrictJavaMode(e.target.checked)}
            className="rounded text-indigo-600 focus:ring-indigo-500 w-3.5 h-3.5 cursor-pointer"
          />
          <span className="font-medium text-[11px]">Strict Java Mode (NaN for Unseen)</span>
        </label>
      </div>

      {/* Main Result Showcase Card */}
      <div className="bg-slate-50/80 rounded-2xl border border-slate-200 p-4 mb-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-2xl sm:text-3xl font-mono font-bold text-slate-900">
                {selectedExpr}
              </span>
              <span className="text-xl text-slate-400 font-mono">=</span>
              {result.predicted === null ? (
                <div className="flex items-center gap-1 text-rose-600 font-mono font-bold text-lg sm:text-xl bg-rose-50 px-2 py-0.5 rounded-md border border-rose-200">
                  <ShieldAlert className="w-4 h-4" />
                  <span>Double.NaN</span>
                </div>
              ) : (
                <span className="text-2xl sm:text-3xl font-mono font-bold text-indigo-700">
                  {result.predicted.toFixed(4)}
                </span>
              )}
            </div>

            <div className="flex items-center gap-3 mt-1.5 text-xs">
              <span className="text-slate-500 font-medium">
                Actual Target: <span className="font-mono font-bold text-slate-900">{actualValue}</span>
              </span>
              {result.predicted !== null ? (
                <span
                  className={`font-mono font-semibold ${
                    error! < 0.05
                      ? 'text-emerald-600'
                      : error! < 0.3
                      ? 'text-amber-600'
                      : 'text-rose-600'
                  }`}
                >
                  Error: ±{error!.toFixed(4)}
                </span>
              ) : (
                <span className="text-slate-400 font-mono">
                  Raw Output: {result.rawPredicted.toFixed(4)} (err: ±{rawError?.toFixed(4)})
                </span>
              )}
            </div>
          </div>

          {/* Known vs Unseen status badge */}
          <div className="flex sm:flex-col items-start sm:items-end gap-1.5">
            {result.isKnown ? (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Known Training Sample</span>
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-800 border border-amber-200">
                <AlertCircle className="w-3.5 h-3.5" />
                <span>Unseen (Out of Train Set)</span>
              </span>
            )}

            <span className="text-[10px] text-slate-400 font-mono">
              Output pos: index 2 (scalar outDim=1)
            </span>
          </div>
        </div>

        {/* Java code behavior note if NaN */}
        {result.predicted === null && (
          <div className="mt-3 p-2.5 rounded-xl bg-amber-50/80 border border-amber-200/80 text-xs text-amber-900 flex items-start gap-2">
            <HelpCircle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-amber-950">
                Java TrainMathFile.java Line 246 Logic:
              </p>
              <p className="text-[11px] font-mono mt-0.5 text-amber-800">
                if(!knownSamples.contains(expr)) return Double.NaN;
              </p>
              <p className="text-[11px] mt-1 text-amber-900">
                In Java, unseen expressions return NaN to avoid out-of-domain extrapolation.
                Uncheck "Strict Java Mode" above to observe the neural network's raw generalization!
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Quick Select Preset Grid */}
      <div className="mb-4">
        <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
          Click an Equation to Test:
        </div>

        {/* Addition equations */}
        <div className="mb-2">
          <div className="text-[10px] font-medium text-slate-400 mb-1">Addition (1..3 + 1..3):</div>
          <div className="grid grid-cols-3 sm:grid-cols-9 gap-1.5">
            {samples
              .filter((s) => s.op === '+')
              .map((sample) => {
                const isSel = selectedExpr === sample.expression;
                const sampleResult = predictFn(sample.expression, false);
                const sampleErr = Math.abs(sampleResult.rawPredicted - sample.target);
                const isGood = step > 0 && sampleErr < 0.1;

                return (
                  <button
                    key={sample.id}
                    id={`test-sample-${sample.id}`}
                    onClick={() => onSelectExpr(sample.expression)}
                    className={`py-1.5 px-2 rounded-lg text-xs font-mono font-medium transition-all text-center cursor-pointer border ${
                      isSel
                        ? 'bg-indigo-600 text-white border-indigo-700 shadow-xs'
                        : isGood
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200 hover:bg-emerald-100'
                        : !sample.isTrain
                        ? 'bg-amber-50 text-amber-800 border-amber-200 hover:bg-amber-100'
                        : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    <div className="font-bold">{sample.expression}</div>
                    <div className="text-[9px] opacity-75">
                      ={step > 0 ? sampleResult.rawPredicted.toFixed(1) : sample.target}
                    </div>
                  </button>
                );
              })}
          </div>
        </div>

        {/* Multiplication equations */}
        <div>
          <div className="text-[10px] font-medium text-slate-400 mb-1">
            Multiplication (1..3 * 1..3):
          </div>
          <div className="grid grid-cols-3 sm:grid-cols-9 gap-1.5">
            {samples
              .filter((s) => s.op === '*')
              .map((sample) => {
                const isSel = selectedExpr === sample.expression;
                const sampleResult = predictFn(sample.expression, false);
                const sampleErr = Math.abs(sampleResult.rawPredicted - sample.target);
                const isGood = step > 0 && sampleErr < 0.1;

                return (
                  <button
                    key={sample.id}
                    id={`test-sample-${sample.id}`}
                    onClick={() => onSelectExpr(sample.expression)}
                    className={`py-1.5 px-2 rounded-lg text-xs font-mono font-medium transition-all text-center cursor-pointer border ${
                      isSel
                        ? 'bg-indigo-600 text-white border-indigo-700 shadow-xs'
                        : !sample.isTrain
                        ? 'bg-amber-50 text-amber-800 border-dashed border-amber-300 hover:bg-amber-100'
                        : isGood
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200 hover:bg-emerald-100'
                        : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                    }`}
                  >
                    <div className="font-bold">
                      {sample.expression}
                      {!sample.isTrain && <span className="text-[9px] ml-0.5 text-amber-600">*</span>}
                    </div>
                    <div className="text-[9px] opacity-75">
                      ={step > 0 ? sampleResult.rawPredicted.toFixed(1) : sample.target}
                    </div>
                  </button>
                );
              })}
          </div>
        </div>
      </div>

      {/* Custom Expression Input Form */}
      <form onSubmit={handleCustomSubmit} className="mt-auto pt-3 border-t border-slate-100 flex gap-2">
        <input
          id="custom-expression-input"
          type="text"
          placeholder="Custom expr (e.g. 2+2, 3*3, 1*2)"
          value={customInput}
          onChange={(e) => setCustomInput(e.target.value)}
          className="flex-1 px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-slate-50"
        />
        <button
          id="custom-eval-btn"
          type="submit"
          className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold shadow-xs transition-colors cursor-pointer"
        >
          Evaluate
        </button>
      </form>
    </div>
  );
};
