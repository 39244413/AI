import React, { useState } from 'react';
import { Layers, Network, ArrowDown, ChevronRight, Binary, Sparkles } from 'lucide-react';
import { Cache } from '../types';

interface ForwardPassInspectorProps {
  expression: string;
  cache: Cache | null;
  dk: number;
}

export const ForwardPassInspector: React.FC<ForwardPassInspectorProps> = ({
  expression,
  cache,
  dk,
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'matrices' | 'math'>('overview');

  if (!cache) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs text-center text-slate-400 text-xs">
        No forward pass cache available for {expression}.
      </div>
    );
  }

  // Format matrix slice helper for compact rendering
  const renderMiniMatrix = (m: number[][], maxCols: number = 4) => {
    return (
      <div className="font-mono text-[10px] bg-slate-900 text-slate-200 p-2.5 rounded-lg overflow-x-auto">
        {m.map((row, rIdx) => (
          <div key={rIdx} className="flex gap-2 whitespace-nowrap">
            <span className="text-slate-500 w-6 text-right">[{rIdx}]:</span>
            {row.slice(0, maxCols).map((val, cIdx) => (
              <span
                key={cIdx}
                className={val === 0 ? 'text-slate-600' : val > 0 ? 'text-emerald-400' : 'text-indigo-300'}
              >
                {val.toFixed(3)}
              </span>
            ))}
            {row.length > maxCols && <span className="text-slate-500">... ({row.length}d)</span>}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 p-4 sm:p-5 shadow-xs">
      {/* Header with tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-50 text-indigo-600">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-900 text-sm">
              Forward Pass Architecture Pipeline
            </h3>
            <p className="text-[11px] text-slate-500">
              Tracing tensor transformations for <span className="font-mono font-bold text-slate-700">{expression}</span>
            </p>
          </div>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center rounded-xl bg-slate-100 p-1 border border-slate-200 text-xs">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-3 py-1 rounded-lg font-medium transition-all cursor-pointer ${
              activeTab === 'overview'
                ? 'bg-white text-indigo-700 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Pipeline Flow
          </button>
          <button
            onClick={() => setActiveTab('matrices')}
            className={`px-3 py-1 rounded-lg font-medium transition-all cursor-pointer ${
              activeTab === 'matrices'
                ? 'bg-white text-indigo-700 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Tensors & Dimensions
          </button>
          <button
            onClick={() => setActiveTab('math')}
            className={`px-3 py-1 rounded-lg font-medium transition-all cursor-pointer ${
              activeTab === 'math'
                ? 'bg-white text-indigo-700 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Formulas
          </button>
        </div>
      </div>

      {/* Tab 1: Overview Flow */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {/* Step 1: Input & Pos Encoding */}
          <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-3 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded">
                  Step 1
                </span>
                <span className="text-[10px] font-mono text-slate-400">3 × 5</span>
              </div>
              <h4 className="font-semibold text-slate-900 text-xs">Embedding + Pos</h4>
              <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
                One-hot token encoding + sinusoidal positional encoding (0.15·pe).
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-200/60 font-mono text-[11px] text-slate-700">
              X_pos = X + PE
            </div>
          </div>

          {/* Step 2: Projections Q, K, V */}
          <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-3 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded">
                  Step 2
                </span>
                <span className="text-[10px] font-mono text-slate-400">3 × {dk}</span>
              </div>
              <h4 className="font-semibold text-slate-900 text-xs">Linear Projections</h4>
              <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
                Query, Key, and Value computed via learned weights Wq, Wk, Wv.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-200/60 font-mono text-[11px] text-slate-700">
              Q, K, V ∈ ℝ^(3×{dk})
            </div>
          </div>

          {/* Step 3: Scaled Dot-Product Attention */}
          <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 p-3 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 bg-indigo-100 px-1.5 py-0.5 rounded">
                  Step 3
                </span>
                <span className="text-[10px] font-mono text-indigo-500 font-semibold">3 × 3</span>
              </div>
              <h4 className="font-semibold text-indigo-950 text-xs">Scaled Dot Attention</h4>
              <p className="text-[11px] text-indigo-900/80 mt-1 leading-relaxed">
                Softmax of Q·Kᵀ / √{dk}, then weighted sum with V.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-indigo-200/60 font-mono text-[11px] text-indigo-800">
              A = softmax(QKᵀ/√{dk})
            </div>
          </div>

          {/* Step 4: Final Output */}
          <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-3 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-700 bg-emerald-100 px-1.5 py-0.5 rounded">
                  Step 4
                </span>
                <span className="text-[10px] font-mono text-slate-400">3 × 1</span>
              </div>
              <h4 className="font-semibold text-slate-900 text-xs">Projection to Output</h4>
              <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
                Final linear layer Wo projects AttnOut (3×24) to scalar value at position 2.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-200/60 font-mono text-[11px] font-bold text-emerald-700">
              ŷ = {cache.finalOut[2][0].toFixed(4)}
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Matrices & Numerical Slices */}
      {activeTab === 'matrices' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          <div>
            <div className="text-xs font-semibold text-slate-700 mb-1 flex items-center justify-between">
              <span>X_pos (Input + PosEncoding)</span>
              <span className="text-slate-400 font-mono text-[10px]">3×5</span>
            </div>
            {renderMiniMatrix(cache.Xpos, 5)}
          </div>

          <div>
            <div className="text-xs font-semibold text-slate-700 mb-1 flex items-center justify-between">
              <span>Query Matrix Q (X·Wq)</span>
              <span className="text-slate-400 font-mono text-[10px]">3×24</span>
            </div>
            {renderMiniMatrix(cache.Q, 4)}
          </div>

          <div>
            <div className="text-xs font-semibold text-slate-700 mb-1 flex items-center justify-between">
              <span>Key Matrix K (X·Wk)</span>
              <span className="text-slate-400 font-mono text-[10px]">3×24</span>
            </div>
            {renderMiniMatrix(cache.K, 4)}
          </div>

          <div>
            <div className="text-xs font-semibold text-slate-700 mb-1 flex items-center justify-between">
              <span>Value Matrix V (X·Wv)</span>
              <span className="text-slate-400 font-mono text-[10px]">3×24</span>
            </div>
            {renderMiniMatrix(cache.V, 4)}
          </div>

          <div>
            <div className="text-xs font-semibold text-slate-700 mb-1 flex items-center justify-between">
              <span>Attention Weights A (Softmax)</span>
              <span className="text-slate-400 font-mono text-[10px]">3×3</span>
            </div>
            {renderMiniMatrix(cache.A, 3)}
          </div>

          <div>
            <div className="text-xs font-semibold text-slate-700 mb-1 flex items-center justify-between">
              <span>Final Output Matrix O (Attn·Wo)</span>
              <span className="text-slate-400 font-mono text-[10px]">3×1</span>
            </div>
            {renderMiniMatrix(cache.finalOut, 1)}
          </div>
        </div>
      )}

      {/* Tab 3: Mathematical Formulations */}
      {activeTab === 'math' && (
        <div className="space-y-3 text-xs text-slate-700 bg-slate-50 p-4 rounded-xl border border-slate-200">
          <div>
            <span className="font-bold text-slate-900 block mb-1">1. Positional Encoding:</span>
            <code className="font-mono bg-white px-2 py-1 rounded border border-slate-200 text-slate-800 text-[11px] block">
              pe[pos][2i] = sin(pos / 10000^(2i/d)), pe[pos][2i+1] = cos(pos / 10000^(2i/d))
            </code>
          </div>

          <div>
            <span className="font-bold text-slate-900 block mb-1">2. Scaled Dot-Product Attention:</span>
            <code className="font-mono bg-white px-2 py-1 rounded border border-slate-200 text-slate-800 text-[11px] block">
              Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
            </code>
          </div>

          <div>
            <span className="font-bold text-slate-900 block mb-1">3. Loss and Output Gradient:</span>
            <code className="font-mono bg-white px-2 py-1 rounded border border-slate-200 text-slate-800 text-[11px] block">
              L = (O[seq-1] - y)²,  dO[seq-1] = 2 · (O[seq-1] - y),  dO[0..seq-2] = 0
            </code>
          </div>

          <div>
            <span className="font-bold text-slate-900 block mb-1">4. Softmax Jacobian Backpropagation:</span>
            <code className="font-mono bg-white px-2 py-1 rounded border border-slate-200 text-slate-800 text-[11px] block">
              dS[i][j] = Σ_k [ A[i][j] · (δ_jk - A[i][k]) · dA[i][k] ]
            </code>
          </div>
        </div>
      )}
    </div>
  );
};
