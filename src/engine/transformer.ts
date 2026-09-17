import { Cache, TransformerConfig } from '../types';

export class TrainMathTransformer {
  vocab: string[] = ['1', '2', '3', '+', '*'];
  VOCAB_SIZE: number = 5;

  knownSamples: Set<string> = new Set();

  Wq: number[][];
  Wk: number[][];
  Wv: number[][];
  Wo: number[][];

  mWq: number[][];
  vWq: number[][];
  mWk: number[][];
  vWk: number[][];
  mWv: number[][];
  vWv: number[][];
  mWo: number[][];
  vWo: number[][];

  lr: number;
  dk: number;
  dv: number;
  outDim: number;
  beta1: number = 0.9;
  beta2: number = 0.999;
  eps: number = 1e-8;
  t: number = 0;

  private seed: number;

  constructor(
    dk: number = 24,
    dv: number = 24,
    outDim: number = 1,
    lr: number = 0.0011,
    seed: number = 42,
    customVocab?: string[]
  ) {
    if (customVocab) {
      this.vocab = [...customVocab];
    }
    this.VOCAB_SIZE = this.vocab.length;
    this.dk = dk;
    this.dv = dv;
    this.outDim = outDim;
    this.lr = lr;
    this.seed = seed;
    this.t = 0;

    this.Wq = this.randInit(this.VOCAB_SIZE, dk);
    this.Wk = this.randInit(this.VOCAB_SIZE, dk);
    this.Wv = this.randInit(this.VOCAB_SIZE, dv);
    this.Wo = this.randInit(dv, outDim);

    this.mWq = this.zeroLike(this.Wq);
    this.vWq = this.zeroLike(this.Wq);
    this.mWk = this.zeroLike(this.Wk);
    this.vWk = this.zeroLike(this.Wk);
    this.mWv = this.zeroLike(this.Wv);
    this.vWv = this.zeroLike(this.Wv);
    this.mWo = this.zeroLike(this.Wo);
    this.vWo = this.zeroLike(this.Wo);
  }

  private nextRandom(): number {
    // Simple LCG PRNG for reproducibility
    this.seed = (this.seed * 1664525 + 1013904223) % 4294967296;
    return this.seed / 4294967296;
  }

  private randInit(r: number, c: number): number[][] {
    const m: number[][] = [];
    for (let i = 0; i < r; i++) {
      m[i] = [];
      for (let j = 0; j < c; j++) {
        // Range: -0.1 to 0.1 (matches Java: (Math.random()*0.2)-0.1)
        m[i][j] = this.nextRandom() * 0.2 - 0.1;
      }
    }
    return m;
  }

  private zeroLike(a: number[][]): number[][] {
    const r = a.length;
    const c = a[0].length;
    const res: number[][] = [];
    for (let i = 0; i < r; i++) {
      res[i] = new Array(c).fill(0);
    }
    return res;
  }

  public static matMul(a: number[][], b: number[][]): number[][] {
    const ar = a.length;
    const ac = a[0].length;
    const bc = b[0].length;
    const res: number[][] = [];
    for (let i = 0; i < ar; i++) {
      res[i] = new Array(bc).fill(0);
      for (let k = 0; k < ac; k++) {
        const aik = a[i][k];
        for (let j = 0; j < bc; j++) {
          res[i][j] += aik * b[k][j];
        }
      }
    }
    return res;
  }

  public static transpose(m: number[][]): number[][] {
    const r = m.length;
    const c = m[0].length;
    const t: number[][] = [];
    for (let j = 0; j < c; j++) {
      t[j] = new Array(r);
      for (let i = 0; i < r; i++) {
        t[j][i] = m[i][j];
      }
    }
    return t;
  }

  public addPosEncoding(X: number[][]): number[][] {
    const seqLen = X.length;
    const fDim = X[0].length;
    const out: number[][] = [];
    for (let pos = 0; pos < seqLen; pos++) {
      out[pos] = [];
      for (let d = 0; d < fDim; d++) {
        let pe: number;
        if (d % 2 === 0) {
          pe = Math.sin(pos / Math.pow(10000.0, (2.0 * d) / fDim));
        } else {
          pe = Math.cos(pos / Math.pow(10000.0, (2.0 * d) / fDim));
        }
        out[pos][d] = X[pos][d] + pe * 0.15;
      }
    }
    return out;
  }

  public softmaxRows(mat: number[][]): number[][] {
    const rows = mat.length;
    const cols = mat[0].length;
    const out: number[][] = [];
    for (let i = 0; i < rows; i++) {
      out[i] = [];
      let sum = 0;
      // Numerical stability max subtraction
      let maxVal = -Infinity;
      for (let j = 0; j < cols; j++) {
        if (mat[i][j] > maxVal) maxVal = mat[i][j];
      }
      for (let j = 0; j < cols; j++) {
        out[i][j] = Math.exp(mat[i][j] - maxVal);
        sum += out[i][j];
      }
      for (let j = 0; j < cols; j++) {
        out[i][j] /= sum;
      }
    }
    return out;
  }

  public softmaxRowGrad(A: number[][], dA: number[][]): number[][] {
    const n = A.length;
    const d = A[0].length;
    const dS: number[][] = [];
    for (let i = 0; i < n; i++) {
      dS[i] = [];
      for (let j = 0; j < d; j++) {
        let s = 0;
        for (let k = 0; k < d; k++) {
          if (j === k) {
            s += A[i][j] * (1 - A[i][k]) * dA[i][k];
          } else {
            s += -A[i][j] * A[i][k] * dA[i][k];
          }
        }
        dS[i][j] = s;
      }
    }
    return dS;
  }

  public forward(XRaw: number[][]): Cache {
    const Xpos = this.addPosEncoding(XRaw);
    const Q = TrainMathTransformer.matMul(Xpos, this.Wq);
    const K = TrainMathTransformer.matMul(Xpos, this.Wk);
    const V = TrainMathTransformer.matMul(Xpos, this.Wv);
    const QKt = TrainMathTransformer.matMul(Q, TrainMathTransformer.transpose(K));

    const scale = Math.sqrt(this.dk);
    const QKtScaled: number[][] = [];
    for (let i = 0; i < QKt.length; i++) {
      QKtScaled[i] = [];
      for (let j = 0; j < QKt[0].length; j++) {
        QKtScaled[i][j] = QKt[i][j] / scale;
      }
    }

    const A = this.softmaxRows(QKtScaled);
    const attnOut = TrainMathTransformer.matMul(A, V);
    const finalOut = TrainMathTransformer.matMul(attnOut, this.Wo);

    return {
      XRaw,
      Xpos,
      Q,
      K,
      V,
      QKt,
      QKtScaled,
      A,
      attnOut,
      finalOut,
    };
  }

  private adamUpdate(
    W: number[][],
    dW: number[][],
    m: number[][],
    v: number[][]
  ): void {
    const beta1Pow = Math.pow(this.beta1, this.t);
    const beta2Pow = Math.pow(this.beta2, this.t);

    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[0].length; j++) {
        m[i][j] = this.beta1 * m[i][j] + (1 - this.beta1) * dW[i][j];
        v[i][j] = this.beta2 * v[i][j] + (1 - this.beta2) * dW[i][j] * dW[i][j];
        const mh = m[i][j] / (1 - beta1Pow);
        const vh = v[i][j] / (1 - beta2Pow);
        W[i][j] -= (this.lr * mh) / (Math.sqrt(vh) + this.eps);
      }
    }
  }

  public step(XRaw: number[][], target: number[][]): number {
    const cache = this.forward(XRaw);
    const O = cache.finalOut;
    const seq = O.length;
    const f = O[0].length;
    let loss = 0;
    const dO: number[][] = [];

    for (let i = 0; i < seq; i++) {
      dO[i] = [];
      for (let j = 0; j < f; j++) {
        if (i !== seq - 1) {
          dO[i][j] = 0.0;
          continue;
        }
        const diff = O[i][j] - target[i][j];
        loss += diff * diff;
        dO[i][j] = 2.0 * diff;
      }
    }

    const dWo = TrainMathTransformer.matMul(TrainMathTransformer.transpose(cache.attnOut), dO);
    const dAttnOut = TrainMathTransformer.matMul(dO, TrainMathTransformer.transpose(this.Wo));

    const dA = TrainMathTransformer.matMul(dAttnOut, TrainMathTransformer.transpose(cache.V));
    const dQKtScaled = this.softmaxRowGrad(cache.A, dA);

    const scale = Math.sqrt(this.dk);
    const dQKt: number[][] = [];
    for (let i = 0; i < dQKtScaled.length; i++) {
      dQKt[i] = [];
      for (let j = 0; j < dQKtScaled[0].length; j++) {
        dQKt[i][j] = dQKtScaled[i][j] * scale;
      }
    }

    const dQ = TrainMathTransformer.matMul(dQKt, cache.K);
    const dK = TrainMathTransformer.matMul(TrainMathTransformer.transpose(dQKt), cache.Q);
    const dV = TrainMathTransformer.matMul(TrainMathTransformer.transpose(cache.A), dAttnOut);

    const dWq = TrainMathTransformer.matMul(TrainMathTransformer.transpose(cache.Xpos), dQ);
    const dWk = TrainMathTransformer.matMul(TrainMathTransformer.transpose(cache.Xpos), dK);
    const dWv = TrainMathTransformer.matMul(TrainMathTransformer.transpose(cache.Xpos), dV);

    this.t++;
    this.adamUpdate(this.Wq, dWq, this.mWq, this.vWq);
    this.adamUpdate(this.Wk, dWk, this.mWk, this.vWk);
    this.adamUpdate(this.Wv, dWv, this.mWv, this.vWv);
    this.adamUpdate(this.Wo, dWo, this.mWo, this.vWo);

    return loss;
  }

  public oneHot(token: string): number[] {
    const vec: number[] = new Array(this.VOCAB_SIZE).fill(0);
    const idx = this.vocab.indexOf(token);
    if (idx !== -1) {
      vec[idx] = 1.0;
    }
    return vec;
  }

  public parseLine(line: string): [number[][], number[][], string, number] | null {
    const trimmed = line.trim();
    if (!trimmed) return null;
    const eq = trimmed.split('=');
    const expr = eq[0].trim();
    const targetVal = eq.length > 1 ? parseFloat(eq[1].trim()) : 0;

    let opPos = -1;
    for (let i = 0; i < expr.length; i++) {
      const c = expr.charAt(i);
      if (c === '+' || c === '*') {
        opPos = i;
        break;
      }
    }
    if (opPos === -1) return null;

    const n1 = expr.substring(0, opPos).trim();
    const op = expr.substring(opPos, opPos + 1).trim();
    const n2 = expr.substring(opPos + 1).trim();
    const key = n1 + op + n2;

    const seq: number[][] = [
      this.oneHot(n1),
      this.oneHot(op),
      this.oneHot(n2),
    ];
    const tgt: number[][] = [[0], [0], [targetVal]];
    return [seq, tgt, key, targetVal];
  }

  public setKnownSamples(samples: string[]) {
    this.knownSamples = new Set(samples);
  }

  public predict(expr: string, strictKnown: boolean = true): {
    predicted: number | null;
    rawPredicted: number;
    isKnown: boolean;
    cache: Cache | null;
  } {
    const isKnown = this.knownSamples.has(expr);
    const parsed = this.parseLine(`${expr}=0`);
    if (!parsed) {
      return { predicted: null, rawPredicted: NaN, isKnown, cache: null };
    }

    const x = parsed[0];
    const cache = this.forward(x);
    const rawPredicted = cache.finalOut[2][0];

    if (strictKnown && !isKnown) {
      return { predicted: null, rawPredicted, isKnown, cache };
    }

    return { predicted: rawPredicted, rawPredicted, isKnown, cache };
  }

  public getConfig(): TransformerConfig {
    return {
      vocab: [...this.vocab],
      dk: this.dk,
      dv: this.dv,
      outDim: this.outDim,
      lr: this.lr,
      beta1: this.beta1,
      beta2: this.beta2,
      eps: this.eps,
    };
  }

  public resetWeights(seed?: number) {
    if (seed !== undefined) {
      this.seed = seed;
    }
    this.t = 0;
    this.Wq = this.randInit(this.VOCAB_SIZE, this.dk);
    this.Wk = this.randInit(this.VOCAB_SIZE, this.dk);
    this.Wv = this.randInit(this.VOCAB_SIZE, this.dv);
    this.Wo = this.randInit(this.dv, this.outDim);

    this.mWq = this.zeroLike(this.Wq);
    this.vWq = this.zeroLike(this.Wq);
    this.mWk = this.zeroLike(this.Wk);
    this.vWk = this.zeroLike(this.Wk);
    this.mWv = this.zeroLike(this.Wv);
    this.vWv = this.zeroLike(this.Wv);
    this.mWo = this.zeroLike(this.Wo);
    this.vWo = this.zeroLike(this.Wo);
  }
}
