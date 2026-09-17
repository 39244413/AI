import React, { useState, useEffect, useRef, useCallback } from 'react';
import { TrainMathTransformer } from './engine/transformer';
import { DEFAULT_DATASET } from './engine/defaults';
import { TrainingSample, LossRecord, Cache } from './types';
import { Header } from './components/Header';
import { Controls } from './components/Controls';
import { LossChart } from './components/LossChart';
import { AttentionVisualizer } from './components/AttentionVisualizer';
import { InferenceTester } from './components/InferenceTester';
import { ForwardPassInspector } from './components/ForwardPassInspector';
import { DatasetManager } from './components/DatasetManager';
import { JavaCodeModal } from './components/JavaCodeModal';

export default function App() {
  // Master instance of the transformer
  const modelRef = useRef<TrainMathTransformer | null>(null);

  // Training state
  const [step, setStep] = useState<number>(0);
  const [currentLoss, setCurrentLoss] = useState<number>(0);
  const [bestLoss, setBestLoss] = useState<number>(Infinity);
  const [lossHistory, setLossHistory] = useState<LossRecord[]>([]);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [stepsPerFrame, setStepsPerFrame] = useState<number>(25);
  const [learningRate, setLearningRate] = useState<number>(0.0011);
  const [isBusy, setIsBusy] = useState<boolean>(false);

  // Dataset & active inspection state
  const [samples, setSamples] = useState<TrainingSample[]>(DEFAULT_DATASET);
  const [selectedExpr, setSelectedExpr] = useState<string>('2+2');
  const [isJavaModalOpen, setIsJavaModalOpen] = useState<boolean>(false);

  // Initialize model on mount
  useEffect(() => {
    const net = new TrainMathTransformer(24, 24, 1, 0.0011, 42);
    // Sync known training samples
    const trainKeys = DEFAULT_DATASET.filter((s) => s.isTrain).map((s) => s.expression);
    net.setKnownSamples(trainKeys);
    modelRef.current = net;
  }, []);

  // Update known samples when dataset changes
  useEffect(() => {
    if (modelRef.current) {
      const trainKeys = samples.filter((s) => s.isTrain).map((s) => s.expression);
      modelRef.current.setKnownSamples(trainKeys);
    }
  }, [samples]);

  // Update learning rate in model
  const handleChangeLearningRate = (lr: number) => {
    setLearningRate(lr);
    if (modelRef.current) {
      modelRef.current.lr = lr;
    }
  };

  // Step function for training
  const performSteps = useCallback(
    (count: number): { lastLoss: number; records: LossRecord[] } => {
      const net = modelRef.current;
      if (!net) return { lastLoss: 0, records: [] };

      const activeTrainSamples = samples.filter((s) => s.isTrain);
      if (activeTrainSamples.length === 0) {
        return { lastLoss: 0, records: [] };
      }

      // Pre-parse training items for speed
      const parsedItems = activeTrainSamples.map((s) => {
        const p = net.parseLine(`${s.expression}=${s.target}`);
        return { x: p![0], y: p![1] };
      });

      let latestLoss = 0;
      const records: LossRecord[] = [];
      const recordInterval = Math.max(1, Math.floor(count / 25));

      for (let i = 0; i < count; i++) {
        const randIdx = Math.floor(Math.random() * parsedItems.length);
        const item = parsedItems[randIdx];
        latestLoss = net.step(item.x, item.y);

        if (i % recordInterval === 0 || i === count - 1) {
          records.push({
            step: net.t,
            loss: latestLoss,
            timestamp: Date.now(),
          });
        }
      }

      return { lastLoss: latestLoss, records };
    },
    [samples]
  );

  // Manual batch stepping (+1, +100, +1k, etc.)
  const handleStep = (count: number) => {
    if (isTraining || isBusy) return;
    setIsBusy(true);

    // Run in a micro-task so UI can display busy status if needed
    setTimeout(() => {
      const { lastLoss, records } = performSteps(count);
      const net = modelRef.current;
      if (net) {
        setStep(net.t);
        setCurrentLoss(lastLoss);
        setBestLoss((prev) => Math.min(prev, lastLoss));
        setLossHistory((prev) => [...prev, ...records].slice(-300));
      }
      setIsBusy(false);
    }, 10);
  };

  // Reset model
  const handleReset = () => {
    setIsTraining(false);
    if (modelRef.current) {
      modelRef.current.resetWeights(Date.now());
      const trainKeys = samples.filter((s) => s.isTrain).map((s) => s.expression);
      modelRef.current.setKnownSamples(trainKeys);
      modelRef.current.lr = learningRate;
    }
    setStep(0);
    setCurrentLoss(0);
    setBestLoss(Infinity);
    setLossHistory([]);
  };

  // Animation frame loop for continuous training
  useEffect(() => {
    if (!isTraining) return;

    let animFrameId: number;
    let accumulatedSteps = 0;

    const loop = () => {
      const { lastLoss, records } = performSteps(stepsPerFrame);
      const net = modelRef.current;

      if (net) {
        accumulatedSteps += stepsPerFrame;
        setStep(net.t);
        setCurrentLoss(lastLoss);
        setBestLoss((prev) => Math.min(prev, lastLoss));

        // Update chart records every ~50 steps or each frame
        if (records.length > 0) {
          setLossHistory((prev) => [...prev, ...records].slice(-300));
        }
      }

      animFrameId = requestAnimationFrame(loop);
    };

    animFrameId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animFrameId);
  }, [isTraining, stepsPerFrame, performSteps]);

  // Predict wrapper function
  const predictFn = useCallback(
    (expr: string, strict: boolean) => {
      if (!modelRef.current) {
        return { predicted: null, rawPredicted: NaN, isKnown: false, cache: null };
      }
      return modelRef.current.predict(expr, strict);
    },
    // We intentionally trigger re-calc on step changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [step, samples]
  );

  // Toggle single sample in dataset
  const handleToggleSample = (id: string) => {
    setSamples((prev) =>
      prev.map((s) => (s.id === id ? { ...s, isTrain: !s.isTrain } : s))
    );
  };

  // Apply dataset preset
  const handleApplyPreset = (preset: 'java-default' | 'all' | 'holdout-mul' | 'holdout-diag') => {
    setSamples((prev) => {
      switch (preset) {
        case 'java-default':
          return prev.map((s) => ({ ...s, isTrain: s.expression !== '1*2' }));
        case 'all':
          return prev.map((s) => ({ ...s, isTrain: true }));
        case 'holdout-mul':
          return prev.map((s) => ({ ...s, isTrain: s.op === '+' }));
        case 'holdout-diag':
          return prev.map((s) => ({
            ...s,
            isTrain: !['1+1', '2+2', '3+3', '1*1', '2*2', '3*3'].includes(s.expression),
          }));
        default:
          return prev;
      }
    });
  };

  // Get current cache for selected expression
  const currentInspection = predictFn(selectedExpr, false);

  return (
    <div className="min-h-screen bg-slate-100/70 text-slate-800 flex flex-col font-sans selection:bg-indigo-100 selection:text-indigo-900">
      {/* Header */}
      <Header
        step={step}
        currentLoss={currentLoss}
        bestLoss={bestLoss}
        isTraining={isTraining}
        trainingSamplesCount={samples.filter((s) => s.isTrain).length}
        onOpenCodeModal={() => setIsJavaModalOpen(true)}
      />

      {/* Main container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Top: Controls bar */}
        <Controls
          isTraining={isTraining}
          onToggleTraining={() => setIsTraining(!isTraining)}
          onStep={handleStep}
          onReset={handleReset}
          stepsPerFrame={stepsPerFrame}
          onChangeStepsPerFrame={setStepsPerFrame}
          learningRate={learningRate}
          onChangeLearningRate={handleChangeLearningRate}
          isBusy={isBusy}
        />

        {/* Row 1: Loss Chart & Attention Visualizer */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-5 flex flex-col">
            <LossChart history={lossHistory} currentLoss={currentLoss} />
          </div>
          <div className="lg:col-span-7 flex flex-col">
            <AttentionVisualizer
              expression={selectedExpr}
              cache={currentInspection.cache}
            />
          </div>
        </div>

        {/* Row 2: Live Inference Tester & Pipeline Inspector */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-6 flex flex-col">
            <InferenceTester
              samples={samples}
              selectedExpr={selectedExpr}
              onSelectExpr={setSelectedExpr}
              predictFn={predictFn}
              step={step}
            />
          </div>
          <div className="lg:col-span-6 flex flex-col">
            <ForwardPassInspector
              expression={selectedExpr}
              cache={currentInspection.cache}
              dk={24}
            />
          </div>
        </div>

        {/* Row 3: Dataset Manager */}
        <DatasetManager
          samples={samples}
          onToggleSample={handleToggleSample}
          onApplyPreset={handleApplyPreset}
          predictFn={predictFn}
          onSelectExpr={setSelectedExpr}
          selectedExpr={selectedExpr}
          step={step}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white px-4 py-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            Transformer Math Lab &bull; Faithfully running <code className="font-mono font-semibold text-slate-700">TrainMathFile.java</code> architecture in TypeScript
          </span>
          <span className="font-mono text-slate-400">
            d_k=24 &bull; d_v=24 &bull; outDim=1 &bull; Adam β₁=0.9, β₂=0.999
          </span>
        </div>
      </footer>

      {/* Java Code & Mathematical Breakdown Modal */}
      <JavaCodeModal
        isOpen={isJavaModalOpen}
        onClose={() => setIsJavaModalOpen(false)}
      />
    </div>
  );
}
