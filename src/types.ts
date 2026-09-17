export interface Cache {
  XRaw: number[][];
  Xpos: number[][];
  Q: number[][];
  K: number[][];
  V: number[][];
  QKt: number[][];
  QKtScaled: number[][];
  A: number[][];
  attnOut: number[][];
  finalOut: number[][];
}

export interface TrainingSample {
  id: string;
  expression: string; // e.g. "1+1"
  n1: string;
  op: string;
  n2: string;
  target: number;
  isTrain: boolean; // whether included in training or held-out
}

export interface LossRecord {
  step: number;
  loss: number;
  timestamp: number;
}

export interface PredictionResult {
  expression: string;
  predicted: number | null; // null if NaN / disallowed
  rawPredicted: number;
  actual: number;
  isKnown: boolean;
  error: number | null;
  cache: Cache | null;
}

export interface TransformerConfig {
  vocab: string[];
  dk: number;
  dv: number;
  outDim: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
}
