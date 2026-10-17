/** Small dense linear algebra on row-major Float64Array, enough for m <= 64. */
export type Mat64 = { rows: number; cols: number; d: Float64Array };

export const zeros = (rows: number, cols: number): Mat64 => ({ rows, cols, d: new Float64Array(rows * cols) });

export function matmulT(a: Mat64, b: Mat64): Mat64 {
  // a (r x c) @ b^T where b is (s x c) -> (r x s)
  const out = zeros(a.rows, b.rows);
  for (let i = 0; i < a.rows; i++)
    for (let j = 0; j < b.rows; j++) {
      let s = 0;
      for (let k = 0; k < a.cols; k++) s += a.d[i * a.cols + k] * b.d[j * b.cols + k];
      out.d[i * out.cols + j] = s;
    }
  return out;
}

export function matmul(a: Mat64, b: Mat64): Mat64 {
  const out = zeros(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++)
    for (let k = 0; k < a.cols; k++) {
      const x = a.d[i * a.cols + k];
      if (x === 0) continue;
      for (let j = 0; j < b.cols; j++) out.d[i * out.cols + j] += x * b.d[k * b.cols + j];
    }
  return out;
}

/** Row softmax; entries where mask(i,j) is false get probability 0. */
export function softmaxRows(a: Mat64, mask?: (i: number, j: number) => boolean): Mat64 {
  const out = zeros(a.rows, a.cols);
  for (let i = 0; i < a.rows; i++) {
    let mx = -Infinity;
    for (let j = 0; j < a.cols; j++) if (!mask || mask(i, j)) mx = Math.max(mx, a.d[i * a.cols + j]);
    let s = 0;
    for (let j = 0; j < a.cols; j++) {
      const e = !mask || mask(i, j) ? Math.exp(a.d[i * a.cols + j] - mx) : 0;
      out.d[i * a.cols + j] = e;
      s += e;
    }
    for (let j = 0; j < a.cols; j++) out.d[i * a.cols + j] /= s;
  }
  return out;
}

type LU = { n: number; lu: Float64Array; piv: Int32Array };

export function luFactor(a: Mat64): LU {
  const n = a.rows;
  const lu = Float64Array.from(a.d);
  const piv = new Int32Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;
  for (let k = 0; k < n; k++) {
    let p = k;
    let best = Math.abs(lu[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(lu[i * n + k]);
      if (v > best) (best = v), (p = i);
    }
    if (p !== k) {
      for (let j = 0; j < n; j++) {
        const t = lu[k * n + j];
        lu[k * n + j] = lu[p * n + j];
        lu[p * n + j] = t;
      }
      const t = piv[k];
      piv[k] = piv[p];
      piv[p] = t;
    }
    const d = lu[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = (lu[i * n + k] /= d);
      if (f === 0) continue;
      for (let j = k + 1; j < n; j++) lu[i * n + j] -= f * lu[k * n + j];
    }
  }
  return { n, lu, piv };
}

function luSolveVec(f: LU, b: Float64Array, transpose = false): Float64Array {
  const { n, lu, piv } = f;
  if (!transpose) {
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) x[i] = b[piv[i]];
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) x[i] -= lu[i * n + j] * x[j];
    for (let i = n - 1; i >= 0; i--) {
      for (let j = i + 1; j < n; j++) x[i] -= lu[i * n + j] * x[j];
      x[i] /= lu[i * n + i];
    }
    return x;
  }
  // solve A^T x = b with P A = L U  ->  A^T = U^T L^T P
  const y = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) y[i] -= lu[j * n + i] * y[j];
    y[i] /= lu[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) for (let j = i + 1; j < n; j++) y[i] -= lu[j * n + i] * y[j];
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[piv[i]] = y[i];
  return x;
}

/** Solve A X = B for X, column by column. */
export function solve(a: Mat64, b: Mat64): Mat64 {
  const f = luFactor(a);
  const out = zeros(b.rows, b.cols);
  const col = new Float64Array(b.rows);
  for (let j = 0; j < b.cols; j++) {
    for (let i = 0; i < b.rows; i++) col[i] = b.d[i * b.cols + j];
    const x = luSolveVec(f, col);
    for (let i = 0; i < b.rows; i++) out.d[i * b.cols + j] = x[i];
  }
  return out;
}

function mulVec(a: Mat64, x: Float64Array, transpose = false): Float64Array {
  const out = new Float64Array(transpose ? a.cols : a.rows);
  for (let i = 0; i < a.rows; i++)
    for (let j = 0; j < a.cols; j++)
      if (transpose) out[j] += a.d[i * a.cols + j] * x[i];
      else out[i] += a.d[i * a.cols + j] * x[j];
  return out;
}

const norm = (x: Float64Array) => Math.sqrt(x.reduce((s, v) => s + v * v, 0));

/** 2-norm condition number via power iteration on A^T A and on (A^T A)^{-1}. */
export function cond2(a: Mat64, iters = 40): number {
  const n = a.rows;
  let x = new Float64Array(n).map((_, i) => 1 + 0.01 * i);
  let smax = 0;
  for (let t = 0; t < iters; t++) {
    const y = mulVec(a, mulVec(a, x), true);
    smax = norm(y);
    x = y.map((v) => v / smax);
  }
  const f = luFactor(a);
  let z = new Float64Array(n).map((_, i) => 1 - 0.01 * i);
  let inv = 0;
  for (let t = 0; t < iters; t++) {
    const y = luSolveVec(f, luSolveVec(f, z, true));
    inv = norm(y);
    z = y.map((v) => v / inv);
  }
  return Math.sqrt(smax * inv);
}

export function relResidual(a: Mat64, x: Mat64, b: Mat64): number {
  const ax = matmul(a, x);
  let num = 0;
  let den = 0;
  for (let i = 0; i < b.d.length; i++) {
    num += (ax.d[i] - b.d[i]) ** 2;
    den += b.d[i] ** 2;
  }
  return Math.sqrt(num / den);
}
