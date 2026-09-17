import React, { useState, useMemo } from 'react';
import { LossRecord } from '../types';
import { TrendingDown, BarChart2 } from 'lucide-react';

interface LossChartProps {
  history: LossRecord[];
  currentLoss: number;
}

export const LossChart: React.FC<LossChartProps> = ({ history, currentLoss }) => {
  const [isLogScale, setIsLogScale] = useState(true);

  // Downsample to max 120 points for smooth SVG rendering
  const downsampled = useMemo(() => {
    if (history.length <= 120) return history;
    const factor = Math.ceil(history.length / 120);
    const result: LossRecord[] = [];
    for (let i = 0; i < history.length; i += factor) {
      result.push(history[i]);
    }
    // ensure the latest point is included
    if (result[result.length - 1] !== history[history.length - 1]) {
      result.push(history[history.length - 1]);
    }
    return result;
  }, [history]);

  const { minLoss, maxLoss, pathData, currentCoord, minCoord } = useMemo(() => {
    if (downsampled.length === 0) {
      return { minLoss: 0, maxLoss: 1, pathData: '', currentCoord: null, minCoord: null };
    }

    let min = Infinity;
    let max = -Infinity;

    for (const pt of downsampled) {
      const val = Math.max(1e-4, pt.loss);
      if (val < min) min = val;
      if (val > max) max = val;
    }

    if (min === Infinity) min = 0.001;
    if (max <= min) max = min + 1;

    const width = 500;
    const height = 160;
    const padding = { top: 12, right: 16, bottom: 20, left: 35 };

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const getX = (index: number) => {
      if (downsampled.length === 1) return padding.left;
      return padding.left + (index / (downsampled.length - 1)) * chartW;
    };

    const getY = (lossVal: number) => {
      const clamped = Math.max(1e-4, lossVal);
      if (isLogScale) {
        const logMin = Math.log10(min);
        const logMax = Math.log10(max);
        const logVal = Math.log10(clamped);
        const ratio = (logVal - logMin) / (logMax - logMin || 1);
        return padding.top + chartH - ratio * chartH;
      } else {
        const ratio = (clamped - min) / (max - min || 1);
        return padding.top + chartH - ratio * chartH;
      }
    };

    let path = '';
    let minPtIndex = 0;
    let minPtVal = Infinity;

    downsampled.forEach((pt, i) => {
      const x = getX(i);
      const y = getY(pt.loss);
      if (i === 0) {
        path += `M ${x.toFixed(1)} ${y.toFixed(1)}`;
      } else {
        path += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
      }
      if (pt.loss < minPtVal) {
        minPtVal = pt.loss;
        minPtIndex = i;
      }
    });

    const lastIdx = downsampled.length - 1;
    const curX = getX(lastIdx);
    const curY = getY(downsampled[lastIdx].loss);

    const minX = getX(minPtIndex);
    const minY = getY(minPtVal);

    return {
      minLoss: min,
      maxLoss: max,
      pathData: path,
      currentCoord: { x: curX, y: curY },
      minCoord: { x: minX, y: minY, val: minPtVal },
    };
  }, [downsampled, isLogScale]);

  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 p-4 sm:p-5 shadow-xs flex flex-col h-full">
      {/* Title & toggles */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-50 text-indigo-600">
            <TrendingDown className="w-4 h-4" />
          </div>
          <h3 className="font-semibold text-slate-900 text-sm">
            Training Loss Curve <span className="text-xs text-slate-400 font-normal">(MSE)</span>
          </h3>
        </div>

        <div className="flex items-center gap-1.5 text-xs">
          <button
            onClick={() => setIsLogScale(!isLogScale)}
            className={`px-2 py-0.5 rounded-md font-mono text-[11px] font-semibold border transition-all cursor-pointer ${
              isLogScale
                ? 'bg-indigo-50 text-indigo-700 border-indigo-200'
                : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
          >
            {isLogScale ? 'Log10 Scale' : 'Linear Scale'}
          </button>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="relative flex-1 min-h-[160px] w-full bg-slate-50/70 rounded-xl border border-slate-100 p-1 flex items-center justify-center overflow-hidden">
        {downsampled.length < 2 ? (
          <div className="text-center p-6 text-slate-400 text-xs">
            <BarChart2 className="w-6 h-6 mx-auto mb-1 opacity-40" />
            <p>Start training to visualize the loss trajectory</p>
          </div>
        ) : (
          <svg
            className="w-full h-full"
            viewBox="0 0 500 160"
            preserveAspectRatio="none"
          >
            {/* Horizontal Grid lines */}
            <line x1="35" y1="12" x2="484" y2="12" stroke="#e2e8f0" strokeDasharray="3 3" />
            <line x1="35" y1="76" x2="484" y2="76" stroke="#e2e8f0" strokeDasharray="3 3" />
            <line x1="35" y1="140" x2="484" y2="140" stroke="#cbd5e1" />

            {/* Target 0.01 threshold guideline */}
            <line x1="35" y1="130" x2="484" y2="130" stroke="#10b981" strokeDasharray="2 2" strokeOpacity="0.4" />

            {/* Y axis labels */}
            <text x="30" y="16" fill="#94a3b8" fontSize="9" textAnchor="end" fontFamily="monospace">
              {maxLoss >= 10 ? maxLoss.toFixed(0) : maxLoss.toFixed(2)}
            </text>
            <text x="30" y="80" fill="#94a3b8" fontSize="9" textAnchor="end" fontFamily="monospace">
              {isLogScale ? (Math.sqrt(maxLoss * minLoss)).toFixed(2) : ((maxLoss + minLoss) / 2).toFixed(2)}
            </text>
            <text x="30" y="142" fill="#94a3b8" fontSize="9" textAnchor="end" fontFamily="monospace">
              {minLoss.toFixed(3)}
            </text>

            {/* Area fill under curve */}
            <path
              d={`${pathData} L ${currentCoord?.x || 484} 140 L 35 140 Z`}
              fill="url(#loss-gradient)"
              opacity="0.35"
            />

            {/* Gradient definition */}
            <defs>
              <linearGradient id="loss-gradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#6366f1" stopOpacity="0.5" />
                <stop offset="100%" stopColor="#6366f1" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Main Loss line */}
            <path
              d={pathData}
              fill="none"
              stroke="#4f46e5"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Current point */}
            {currentCoord && (
              <circle
                cx={currentCoord.x}
                cy={currentCoord.y}
                r="3.5"
                fill="#4338ca"
                stroke="#ffffff"
                strokeWidth="1.5"
              />
            )}
          </svg>
        )}
      </div>

      {/* Footer stat metrics */}
      <div className="flex items-center justify-between mt-3 text-xs text-slate-500 font-mono">
        <div>
          <span>Min Loss: </span>
          <span className="font-semibold text-emerald-600">
            {minCoord ? minCoord.val.toFixed(5) : '—'}
          </span>
        </div>
        <div>
          <span>History: </span>
          <span className="font-semibold text-slate-700">{history.length} pts</span>
        </div>
      </div>
    </div>
  );
};
