import React, { useState } from 'react';
import { Sparkles, Eye, Info, ArrowRight } from 'lucide-react';
import { Cache } from '../types';

interface AttentionVisualizerProps {
  expression: string;
  cache: Cache | null;
}

export const AttentionVisualizer: React.FC<AttentionVisualizerProps> = ({
  expression,
  cache,
}) => {
  const [hoveredCell, setHoveredCell] = useState<{ qIdx: number; kIdx: number } | null>(null);

  // Extract tokens from expression, e.g. "2+3" -> ["2", "+", "3"]
  let tokens = ['1', '+', '1'];
  if (expression) {
    let opPos = -1;
    for (let i = 0; i < expression.length; i++) {
      if (expression[i] === '+' || expression[i] === '*') {
        opPos = i;
        break;
      }
    }
    if (opPos !== -1) {
      tokens = [
        expression.substring(0, opPos),
        expression.substring(opPos, opPos + 1),
        expression.substring(opPos + 1),
      ];
    }
  }

  const tokenLabels = [
    { label: tokens[0], role: 'Operand 1 (Pos 0)' },
    { label: tokens[1], role: 'Operator (Pos 1)' },
    { label: tokens[2], role: 'Operand 2 (Pos 2)' },
  ];

  const attnMatrix = cache?.A || [
    [0.333, 0.333, 0.334],
    [0.333, 0.333, 0.334],
    [0.333, 0.333, 0.334],
  ];

  const getColorStyle = (val: number) => {
    // Shading from clean light blue/indigo to rich deep indigo
    const pct = Math.max(0, Math.min(1, val));
    if (pct < 0.15) {
      return { backgroundColor: 'rgba(238, 242, 255, 0.6)', color: '#475569' };
    } else if (pct < 0.35) {
      return { backgroundColor: 'rgba(199, 210, 254, 0.8)', color: '#312e81' };
    } else if (pct < 0.6) {
      return { backgroundColor: 'rgba(129, 140, 248, 0.85)', color: '#ffffff' };
    } else {
      return { backgroundColor: 'rgba(79, 70, 229, 0.95)', color: '#ffffff' };
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 p-4 sm:p-5 shadow-xs flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-50 text-indigo-600">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-900 text-sm">
              Self-Attention Matrix <span className="font-mono text-indigo-600">A</span>
            </h3>
            <p className="text-[11px] text-slate-500">
              Softmax(Q · Kᵀ / √{24}) weights for <span className="font-mono font-bold text-slate-700">{expression}</span>
            </p>
          </div>
        </div>

        <div className="text-[11px] text-slate-400 font-mono hidden sm:block">
          3×3 Attention Weights
        </div>
      </div>

      {/* Main Attention Heatmap Grid */}
      <div className="my-auto py-2">
        <div className="max-w-md mx-auto">
          {/* Column Header: Keys (K) */}
          <div className="grid grid-cols-4 gap-2 mb-2">
            <div className="text-[11px] font-semibold text-slate-400 flex items-center justify-center">
              <span className="bg-slate-100 px-1.5 py-0.5 rounded text-[10px] text-slate-500 uppercase tracking-wider">
                Q \ K
              </span>
            </div>
            {tokenLabels.map((tok, kIdx) => (
              <div key={kIdx} className="text-center">
                <div className="w-8 h-8 mx-auto rounded-lg bg-indigo-50 border border-indigo-200/80 flex items-center justify-center font-mono font-bold text-indigo-900 text-sm shadow-xs">
                  {tok.label}
                </div>
                <div className="text-[10px] text-slate-500 font-medium mt-1">k{kIdx}</div>
              </div>
            ))}
          </div>

          {/* Rows: Queries (Q) */}
          {tokenLabels.map((qTok, qIdx) => (
            <div key={qIdx} className="grid grid-cols-4 gap-2 items-center mb-2">
              {/* Row header: Query token */}
              <div className="flex items-center justify-between pr-2">
                <span className="text-[10px] text-slate-500 font-medium">q{qIdx}</span>
                <div className="w-8 h-8 rounded-lg bg-slate-100 border border-slate-200 flex items-center justify-center font-mono font-bold text-slate-800 text-sm">
                  {qTok.label}
                </div>
              </div>

              {/* 3 Cells */}
              {[0, 1, 2].map((kIdx) => {
                const weight = attnMatrix[qIdx]?.[kIdx] ?? 0.333;
                const isHovered = hoveredCell?.qIdx === qIdx && hoveredCell?.kIdx === kIdx;
                const style = getColorStyle(weight);

                return (
                  <div
                    key={kIdx}
                    onMouseEnter={() => setHoveredCell({ qIdx, kIdx })}
                    onMouseLeave={() => setHoveredCell(null)}
                    style={style}
                    className={`h-11 rounded-xl flex flex-col items-center justify-center transition-all cursor-pointer select-none font-mono ${
                      isHovered ? 'ring-2 ring-indigo-600 ring-offset-1 scale-105 z-10' : ''
                    }`}
                  >
                    <span className="text-xs font-bold leading-none">
                      {(weight * 100).toFixed(1)}%
                    </span>
                    <span className="text-[9px] opacity-80 mt-0.5">
                      {weight.toFixed(3)}
                    </span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Explanatory tooltip banner */}
      <div className="mt-3 p-2.5 rounded-xl bg-slate-50 border border-slate-200/70 text-xs">
        {hoveredCell ? (
          <div className="flex items-start gap-2 text-slate-700">
            <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-slate-900">
                Query Token '{tokenLabels[hoveredCell.qIdx].label}' (Pos {hoveredCell.qIdx})
              </span>{' '}
              attends{' '}
              <span className="font-bold text-indigo-700 font-mono">
                {((attnMatrix[hoveredCell.qIdx]?.[hoveredCell.kIdx] ?? 0) * 100).toFixed(1)}%
              </span>{' '}
              to Key Token '{tokenLabels[hoveredCell.kIdx].label}' (Pos {hoveredCell.kIdx}).
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-slate-500">
            <Eye className="w-3.5 h-3.5 text-slate-400 shrink-0" />
            <span>Hover any cell to inspect query-key attention distribution. Notice how position 2 (final output) gathers context from the operator and operands!</span>
          </div>
        )}
      </div>

      {/* Token Flow Diagram */}
      <div className="mt-3 pt-3 border-t border-slate-100">
        <div className="text-[11px] font-semibold text-slate-600 mb-2 flex items-center justify-between">
          <span>Output Position (Pos 2) Attention Flow:</span>
          <span className="font-mono text-indigo-600 text-[10px]">
            Loss evaluated at index 2
          </span>
        </div>
        <div className="flex items-center justify-between gap-1 bg-indigo-50/50 p-2 rounded-xl border border-indigo-100 text-xs">
          {tokenLabels.map((tok, idx) => {
            const attnFromLast = attnMatrix[2]?.[idx] ?? 0.333;
            return (
              <div key={idx} className="flex-1 text-center bg-white p-2 rounded-lg border border-slate-100 shadow-2xs">
                <div className="font-mono font-bold text-slate-900 text-sm">
                  {tok.label}
                </div>
                <div className="text-[10px] text-slate-400">Pos {idx}</div>
                <div className="mt-1 font-mono font-bold text-indigo-600 text-xs">
                  {(attnFromLast * 100).toFixed(0)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
