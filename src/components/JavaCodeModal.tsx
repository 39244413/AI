import React, { useState } from 'react';
import { X, Copy, Check, Code, FileText, Lightbulb, BookOpen } from 'lucide-react';
import { JAVA_SOURCE_CODE } from '../engine/defaults';

interface JavaCodeModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const JavaCodeModal: React.FC<JavaCodeModalProps> = ({ isOpen, onClose }) => {
  const [copied, setCopied] = useState(false);
  const [tab, setTab] = useState<'code' | 'breakdown'>('code');

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(JAVA_SOURCE_CODE);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
      <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] flex flex-col shadow-2xl border border-slate-200 overflow-hidden">
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50/80">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center">
              <Code className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <span>TrainMathFile.java</span>
                <span className="text-[11px] font-mono font-normal px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                  Vanilla Java Transformer
                </span>
              </h2>
              <p className="text-xs text-slate-500">
                Self-contained mathematical Transformer implementation from scratch
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-700 bg-white hover:bg-slate-100 border border-slate-200 transition-colors cursor-pointer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy Code'}</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab selector */}
        <div className="px-5 pt-2 border-b border-slate-100 flex gap-4 text-xs font-medium">
          <button
            onClick={() => setTab('code')}
            className={`py-2 border-b-2 transition-colors cursor-pointer ${
              tab === 'code'
                ? 'border-indigo-600 text-indigo-600 font-bold'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Java Source Code
          </button>
          <button
            onClick={() => setTab('breakdown')}
            className={`py-2 border-b-2 transition-colors cursor-pointer ${
              tab === 'breakdown'
                ? 'border-indigo-600 text-indigo-600 font-bold'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Mathematical Architecture Breakdown
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {tab === 'code' ? (
            <div className="relative">
              <pre className="font-mono text-xs text-slate-800 bg-slate-950 text-slate-100 p-4 rounded-xl overflow-x-auto leading-relaxed">
                <code>{JAVA_SOURCE_CODE}</code>
              </pre>
            </div>
          ) : (
            <div className="space-y-4 text-xs text-slate-700">
              <div className="p-4 rounded-xl bg-indigo-50/70 border border-indigo-100">
                <div className="flex items-center gap-2 text-indigo-900 font-bold mb-1 text-sm">
                  <Lightbulb className="w-4 h-4 text-indigo-600" />
                  <span>How the Transformer Solves Arithmetic</span>
                </div>
                <p className="text-slate-700 leading-relaxed">
                  Unlike traditional recurrent or feed-forward networks, this architecture represents equations as a sequence of 3 discrete tokens:
                  <code className="mx-1 px-1.5 py-0.5 rounded bg-white text-indigo-800 font-mono">["operand1", "operator", "operand2"]</code>.
                  Self-attention enables the output position to dynamically weight and combine the operand values conditioned on whether the operator is addition (<code className="font-mono">+</code>) or multiplication (<code className="font-mono">*</code>).
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="p-3.5 rounded-xl border border-slate-200 bg-white">
                  <h4 className="font-bold text-slate-900 mb-1">1. Positional Encoding</h4>
                  <p className="text-slate-600 leading-relaxed">
                    Without positional encoding, self-attention is permutation-invariant. Adding <code className="font-mono">sin/cos</code> vectors ensures the network knows the distinction between the 1st operand (pos 0), operator (pos 1), and 2nd operand (pos 2).
                  </p>
                </div>

                <div className="p-3.5 rounded-xl border border-slate-200 bg-white">
                  <h4 className="font-bold text-slate-900 mb-1">2. Scaled Dot-Product Attention</h4>
                  <p className="text-slate-600 leading-relaxed">
                    Queries and Keys are projected into ℝ^(3×24). The attention weights <code className="font-mono">A = softmax(Q·Kᵀ / √24)</code> compute pairwise relevance before scaling Values (V).
                  </p>
                </div>

                <div className="p-3.5 rounded-xl border border-slate-200 bg-white">
                  <h4 className="font-bold text-slate-900 mb-1">3. Analytical Backpropagation</h4>
                  <p className="text-slate-600 leading-relaxed">
                    Every gradient is computed manually in pure Java without autograd frameworks. The softmax Jacobian-vector product <code className="font-mono">softmaxRowGrad</code> backpropagates through the attention matrix with exact calculus.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl border border-slate-200 bg-white">
                  <h4 className="font-bold text-slate-900 mb-1">4. Adam Optimizer (β₁=0.9, β₂=0.999)</h4>
                  <p className="text-slate-600 leading-relaxed">
                    Maintains exponential moving averages of gradients (m) and squared gradients (v) with bias-correction factors <code className="font-mono">1 - β^t</code> to guarantee stable convergence.
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-xl border border-amber-200 bg-amber-50/60 text-amber-900">
                <h4 className="font-bold text-amber-950 mb-1">Why does `predict("1*2")` return Double.NaN?</h4>
                <p className="leading-relaxed">
                  In <code className="font-mono font-semibold">TrainMathFile.java</code> line 246:
                  <code className="block mt-1 p-2 bg-white/80 rounded font-mono text-amber-950">
                    if(!knownSamples.contains(expr)) return Double.NaN;
                  </code>
                  The author maintained a <code className="font-mono">knownSamples</code> set and intentionally left <code className="font-mono">1*2=2</code> out of <code className="font-mono">train.txt</code> to test whether the system properly guards against evaluating out-of-dataset expressions. In our lab app, you can toggle this strict mode on or off to inspect the raw network response!
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-slate-200 bg-slate-50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-900 text-white font-semibold text-xs hover:bg-slate-800 transition-colors cursor-pointer"
          >
            Close Reference
          </button>
        </div>
      </div>
    </div>
  );
};
