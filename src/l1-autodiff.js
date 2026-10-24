/**
 * NEXUS · L1 反向契约
 * ---------------------------------------------------------------------------
 * 把"正确性"从测试环节提升到架构层。
 * 作者：晨星 (CJX0712) · MIT
 *
 * 设计决策（沿袭 tianji-ai L1，并经本项目复核）：
 *
 *   方案                              结论      理由
 *   ─────────────────────────────────────────────────────────────────────
 *   tape 式通用 AD                     ✅ 采用   作为**金标准**：写起来直白、
 *                                               每个算子只需写 forward 的微分直觉，
 *                                               出错面小。它的存在是为了给手写 backward 当尺子。
 *   手写显式 backward（L3/L4）         ✅ 采用   性能路径；递推结构（如 DeltaNet）
 *                                               只能手写，tape 展开会炸内存。
 *   只用手写、不保留 tape              ❌        没人给手写梯度当尺子，等于自证。
 *
 * 两条路径由 G21（梯度交叉验证）强制互证：同一算子的 tape 梯度与手写梯度必须
 * 逐位落在 gradcheck 判据内。这不是"多写一份"，是让错误无处可藏。
 *
 * 判据（PyTorch gradcheck 同款混合判据）：|a − n| ≤ atol + rtol·|n|
 *   atol 用**全局梯度尺度**，不是单元素量级 —— 否则 1e-8 量级的梯度会被差分噪声淹没、
 *   被误判 FAIL（tianji 早期踩过：analytic/numeric 比值 1.0000 却判 FAIL）。
 */

'use strict';
const L0 = require('./l0-numerics.js');

// ── 张量 ────────────────────────────────────────────────────────────────────
function shapeSize(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}
function shapeStrides(shape) {
  const st = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) st[i] = st[i + 1] * shape[i + 1];
  return st;
}
function shapeEq(a, b) { return a.length === b.length && a.every((v, i) => v === b[i]); }

class Tensor {
  constructor(data, shape, opts = {}) {
    this.shape = shape.slice();
    const n = shapeSize(shape);
    this.data = data instanceof Float64Array ? data : Float64Array.from(data);
    if (this.data.length !== n) {
      throw new Error(`形状/数据长度不一致：shape=[${shape}] 需要 ${n}，实得 ${this.data.length}`);
    }
    this.grad = null;
    this.requiresGrad = !!opts.requiresGrad;
    this._op = opts.op || null;
    this._inputs = opts.inputs || [];
    this._backward = opts.backward || null;
    this.name = opts.name || '';
  }
  static zeros(shape, requiresGrad) { return new Tensor(new Float64Array(shapeSize(shape)), shape, { requiresGrad }); }
  static ones(shape, requiresGrad) { return new Tensor(new Float64Array(shapeSize(shape)).fill(1), shape, { requiresGrad }); }
  static randn(shape, rng, requiresGrad) {
    const a = new Float64Array(shapeSize(shape));
    for (let i = 0; i < a.length; i++) a[i] = rng.normal();
    return new Tensor(a, shape, { requiresGrad });
  }
  get size() { return this.data.length; }
  get ndim() { return this.shape.length; }

  zeroGrad() { this.grad = null; return this; }

  /** 沿计算图反向传播。grad 可省略（标量损失默认填 1）。 */
  backward(grad) {
    // 1. 拓扑排序（DFS，只走需要梯度的边）
    const topo = [];
    const seen = new Set();
    (function dfs(v) {
      if (seen.has(v)) return;
      seen.add(v);
      for (const inp of v._inputs) if (inp.requiresGrad || inp._backward) dfs(inp);
      topo.push(v);
    })(this);

    if (grad === undefined) {
      if (this.size !== 1) throw new Error(`非标量 backward 必须显式给 grad（本张量 size=${this.size}）`);
      grad = new Float64Array([1]);
    }
    if (grad instanceof Tensor) grad = grad.data;
    if (!(grad instanceof Float64Array)) grad = Float64Array.from(grad);
    if (grad.length !== this.size) {
      throw new Error(`backward 起始梯度形状不符：需要 ${this.size}，实得 ${grad.length}`);
    }
    this.grad = Float64Array.from(grad);

    // 2. 逆序执行每个节点的 backward 闭包
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      if (!v._backward || v.grad === null) continue;
      v._backward(v.grad);
    }
  }

  /** 累加梯度：一个张量被多处消费时必须累加，不能覆盖。 */
  _accum(g) {
    if (!this.requiresGrad) return;
    if (g instanceof Tensor) g = g.data;
    if (this.grad === null) this.grad = new Float64Array(this.size);
    if (g.length !== this.grad.length) {
      throw new Error(`梯度形状契约破裂 @${this.name || '?'}: 期望 ${this.grad.length}，实得 ${g.length}`);
    }
    for (let i = 0; i < g.length; i++) this.grad[i] += g[i];
  }

  item() { if (this.size !== 1) throw new Error('item() 只适用于单元素张量'); return this.data[0]; }
  clone() { return new Tensor(Float64Array.from(this.data), this.shape); }
  toString() {
    return `Tensor([${this.shape}]${this.grad ? ' grad✓' : ''})`;
  }
}

function T(data, shape, opts) { return new Tensor(data, shape, opts); }

/** 包装一个节点：把 forward 结果 + backward 闭包绑成新张量。 */
function node(outData, outShape, inputs, backward, opts = {}) {
  return new Tensor(outData, outShape, {
    inputs, backward,
    requiresGrad: opts.requiresGrad !== undefined ? opts.requiresGrad : inputs.some((i) => i.requiresGrad),
    op: opts.op, name: opts.name,
  });
}

// ── 算子 ────────────────────────────────────────────────────────────────────

/** 逐元素二元：广播按 NumPy 规则（右对齐）。 */
function _broadcastShapes(a, b) {
  const nd = Math.max(a.length, b.length);
  const out = new Array(nd);
  for (let i = 0; i < nd; i++) {
    const da = i < nd - a.length ? 1 : a[i - (nd - a.length)];
    const db = i < nd - b.length ? 1 : b[i - (nd - b.length)];
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`广播失败：[${a}] vs [${b}]`);
    }
    out[i] = Math.max(da, db);
  }
  return out;
}
function _broadcastIdx(shape, strides, outStrides, nd) {
  // 返回映射：给定 out 的线性索引 → 输入的线性索引步长表（预计算每维 stride）
  const pad = nd - shape.length;
  const st = new Array(nd).fill(0);
  for (let i = 0; i < shape.length; i++) st[pad + i] = shape[i] === 1 ? 0 : strides[i];
  return st;
}

function _mapBinary(a, b, f, da, db, opName) {
  const outShape = _broadcastShapes(a.shape, b.shape);
  const nd = outShape.length;
  const n = shapeSize(outShape);
  const out = new Float64Array(n);
  const sa = _broadcastIdx(a.shape, shapeStrides(a.shape), null, nd);
  const sb = _broadcastIdx(b.shape, shapeStrides(b.shape), null, nd);
  const oStrides = shapeStrides(outShape);
  const coords = new Array(nd).fill(0);
  for (let o = 0; o < n; o++) {
    // 由线性索引还原坐标（增量式，比每层取模快）
    let ia = 0, ib = 0;
    for (let d = 0; d < nd; d++) { ia += coords[d] * sa[d]; ib += coords[d] * sb[d]; }
    out[o] = f(a.data[ia], b.data[ib]);
    // 递增坐标
    for (let d = nd - 1; d >= 0; d--) { coords[d]++; if (coords[d] < outShape[d]) break; coords[d] = 0; }
  }
  return node(out, outShape, [a, b], (g) => {
    // 反向：梯度按 forward 的索引映射累加回去
    const gaT = new Float64Array(a.size), gbT = new Float64Array(b.size);
    const c = new Array(nd).fill(0);
    for (let o = 0; o < n; o++) {
      let ia = 0, ib = 0;
      for (let d = 0; d < nd; d++) { ia += c[d] * sa[d]; ib += c[d] * sb[d]; }
      const av = a.data[ia], bv = b.data[ib];
      const [d1, d2] = da(av, bv, out[o]);
      gaT[ia] += g[o] * d1; gbT[ib] += g[o] * d2;
      for (let d = nd - 1; d >= 0; d--) { c[d]++; if (c[d] < outShape[d]) break; c[d] = 0; }
    }
    if (a.requiresGrad) a._accum(gaT);
    if (b.requiresGrad) b._accum(gbT);
  }, { op: opName });
}

const add = (a, b) => _mapBinary(a, b, (x, y) => x + y, () => [1, 1], null, 'add');
const sub = (a, b) => _mapBinary(a, b, (x, y) => x - y, () => [1, -1], null, 'sub');
const mul = (a, b) => _mapBinary(a, b, (x, y) => x * y, (x, y) => [y, x], null, 'mul');
const div = (a, b) => _mapBinary(a, b, (x, y) => x / y, (x, y) => [1 / y, -x / (y * y)], null, 'div');

/** 逐元素一元。 */
function _mapUnary(a, f, df, opName) {
  const out = new Float64Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = f(a.data[i]);
  return node(out, a.shape, [a], (g) => {
    if (!a.requiresGrad) return;
    const ga = new Float64Array(a.size);
    for (let i = 0; i < a.size; i++) ga[i] = g[i] * df(a.data[i], out[i]);
    a._accum(ga);
  }, { op: opName });
}

const exp = (a) => _mapUnary(a, Math.exp, (x, o) => o, 'exp');
const log = (a) => _mapUnary(a, Math.log, (x) => 1 / x, 'log');
const neg = (a) => _mapUnary(a, (x) => -x, () => -1, 'neg');
const tanh = (a) => _mapUnary(a, Math.tanh, (x, o) => 1 - o * o, 'tanh');
const relu = (a) => _mapUnary(a, (x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0), 'relu');
const sigmoid = (a) => _mapUnary(a, L0.sigmoid, (x, o) => o * (1 - o), 'sigmoid');
const sqrt = (a) => _mapUnary(a, Math.sqrt, (x, o) => 0.5 / o, 'sqrt');
const square = (a) => _mapUnary(a, (x) => x * x, (x) => 2 * x, 'square');
/** GELU（tanh 近似，Hendrycks & Gimpel 2016；与精确版最大差 ~1e-3）。 */
const gelu = (a) => _mapUnary(a,
  (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3))),
  (x) => {
    const t = Math.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3));
    const sech2 = 1 - t * t;
    const inner = 0.7978845608028654 * (1 + 3 * 0.044715 * x * x);
    return 0.5 * (1 + t) + 0.5 * x * sech2 * inner;
  }, 'gelu');
/** SiLU / Swish：x·σ(x)。 */
const silu = (a) => _mapUnary(a, (x) => x * L0.sigmoid(x), (x) => {
  const s = L0.sigmoid(x);
  return s + x * s * (1 - s);
}, 'silu');

/** 矩阵乘（2D，支持批量 3D）。 */
function matmul(a, b) {
  if (a.ndim === 2 && b.ndim === 2) {
    const [M, K] = a.shape, [K2, N] = b.shape;
    if (K !== K2) throw new Error(`matmul 维度不匹配：[${a.shape}] @ [${b.shape}]`);
    const out = new Float64Array(M * N);
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const av = a.data[i * K + k];
        if (av === 0) continue;
        for (let j = 0; j < N; j++) out[i * N + j] += av * b.data[k * N + j];
      }
    }
    return node(out, [M, N], [a, b], (g) => {
      if (a.requiresGrad) {
        const ga = new Float64Array(M * K);
        for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
          let s = 0; for (let j = 0; j < N; j++) s += g[i * N + j] * b.data[k * N + j];
          ga[i * K + k] = s;
        }
        a._accum(ga);
      }
      if (b.requiresGrad) {
        const gb = new Float64Array(K * N);
        for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
          let s = 0; for (let i = 0; i < M; i++) s += a.data[i * K + k] * g[i * N + j];
          gb[k * N + j] = s;
        }
        b._accum(gb);
      }
    }, { op: 'matmul' });
  }
  // 批量 (B,M,K) @ (B,K,N)
  if (a.ndim === 3 && b.ndim === 3) {
    const [B, M, K] = a.shape, [B2, K2, N] = b.shape;
    if (B !== B2 || K !== K2) throw new Error(`批量 matmul 维度不匹配：[${a.shape}] @ [${b.shape}]`);
    const out = new Float64Array(B * M * N);
    for (let bth = 0; bth < B; bth++) {
      const ao = bth * M * K, bo = bth * K * N, oo = bth * M * N;
      for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
        const av = a.data[ao + i * K + k];
        if (av === 0) continue;
        for (let j = 0; j < N; j++) out[oo + i * N + j] += av * b.data[bo + k * N + j];
      }
    }
    return node(out, [B, M, N], [a, b], (g) => {
      if (a.requiresGrad) {
        const ga = new Float64Array(B * M * K);
        for (let bth = 0; bth < B; bth++) {
          const ao = bth * M * K, bo = bth * K * N, go = bth * M * N;
          for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
            let s = 0; for (let j = 0; j < N; j++) s += g[go + i * N + j] * b.data[bo + k * N + j];
            ga[ao + i * K + k] = s;
          }
        }
        a._accum(ga);
      }
      if (b.requiresGrad) {
        const gb = new Float64Array(B * K * N);
        for (let bth = 0; bth < B; bth++) {
          const ao = bth * M * K, bo = bth * K * N, go = bth * M * N;
          for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
            let s = 0; for (let i = 0; i < M; i++) s += a.data[ao + i * K + k] * g[go + i * N + j];
            gb[bo + k * N + j] = s;
          }
        }
        b._accum(gb);
      }
    }, { op: 'matmul_batched' });
  }
  throw new Error(`matmul 仅支持 2D/3D，实得 ${a.ndim}D @ ${b.ndim}D`);
}

/** 转置最后两维。 */
function transpose(a) {
  if (a.ndim < 2) throw new Error('transpose 需要至少 2 维');
  const shape = a.shape.slice();
  const d1 = shape[shape.length - 1], d2 = shape[shape.length - 2];
  shape[shape.length - 1] = d2; shape[shape.length - 2] = d1;
  const batch = a.size / (d1 * d2);
  const out = new Float64Array(a.size);
  for (let bth = 0; bth < batch; bth++) {
    const off = bth * d1 * d2;
    for (let i = 0; i < d1; i++) for (let j = 0; j < d2; j++) {
      out[off + j * d1 + i] = a.data[off + i * d2 + j];
    }
  }
  return node(out, shape, [a], (g) => {
    if (!a.requiresGrad) return;
    const ga = new Float64Array(a.size);
    for (let bth = 0; bth < batch; bth++) {
      const off = bth * d1 * d2;
      for (let i = 0; i < d1; i++) for (let j = 0; j < d2; j++) {
        ga[off + i * d2 + j] = g[off + j * d1 + i];
      }
    }
    a._accum(ga);
  }, { op: 'transpose' });
}

/** 重塑（连续布局，仅重排 shape）。 */
function reshape(a, shape) {
  if (shapeSize(shape) !== a.size) {
    throw new Error(`reshape 元素数不符：${a.size} → [${shape}]=${shapeSize(shape)}`);
  }
  return node(a.data, shape, [a], (g) => {
    if (!a.requiresGrad) return;
    a._accum(g);   // 连续布局下 reshape 的反向就是同数据换形状
  }, { op: 'reshape' });
}

/** 求和（可沿轴）。沿轴归约时输出降维。 */
function sum(a, axis) {
  if (axis === undefined) {
    // 全归约 → 标量
    let s = 0;
    for (let i = 0; i < a.size; i++) s += a.data[i];
    s = L0.sumKahan(a.data);   // 补偿求和：归约是浮点误差的重灾区
    return node(new Float64Array([s]), [1], [a], (g) => {
      if (!a.requiresGrad) return;
      const ga = new Float64Array(a.size);
      for (let i = 0; i < a.size; i++) ga[i] = g[0];
      a._accum(ga);
    }, { op: 'sum' });
  }
  if (a.ndim !== 2) throw new Error('sum(axis) 目前支持 2D');
  const [R, C] = a.shape;
  const out = axis === 0 ? new Float64Array(C) : new Float64Array(R);
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    if (axis === 0) out[j] += a.data[i * C + j]; else out[i] += a.data[i * C + j];
  }
  const outShape = axis === 0 ? [1, C] : [R, 1];
  return node(out, outShape, [a], (g) => {
    if (!a.requiresGrad) return;
    const ga = new Float64Array(a.size);
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      ga[i * C + j] = axis === 0 ? g[j] : g[i];
    }
    a._accum(ga);
  }, { op: 'sum_axis' });
}

/** 均值（全归约）。 */
function mean(a) {
  const s = L0.sumKahan(a.data) / a.size;
  return node(new Float64Array([s]), [1], [a], (g) => {
    if (!a.requiresGrad) return;
    const ga = new Float64Array(a.size);
    const c = g[0] / a.size;
    for (let i = 0; i < a.size; i++) ga[i] = c;
    a._accum(ga);
  }, { op: 'mean' });
}

/** 按行 softmax（2D）。平移稳定 + 行和恒 1。 */
function softmaxRows(a) {
  if (a.ndim !== 2) throw new Error('softmaxRows 需要 2D');
  const [R, C] = a.shape;
  const out = new Float64Array(R * C);
  for (let i = 0; i < R; i++) {
    const row = new Float64Array(C);
    for (let j = 0; j < C; j++) row[j] = a.data[i * C + j];
    const s = L0.softmax(row);
    for (let j = 0; j < C; j++) out[i * C + j] = s[j];
  }
  return node(out, [R, C], [a], (g) => {
    if (!a.requiresGrad) return;
    const ga = new Float64Array(R * C);
    for (let i = 0; i < R; i++) {
      // softmax 的 Jacobian：∂s_j/∂x_k = s_j(δ_jk − s_k)
      // 即 ga_k = s_k (g_k − Σ_j g_j s_j)
      let dot = 0;
      for (let j = 0; j < C; j++) dot += g[i * C + j] * out[i * C + j];
      for (let k = 0; k < C; k++) ga[i * C + k] = out[i * C + k] * (g[i * C + k] - dot);
    }
    a._accum(ga);
  }, { op: 'softmax' });
}

/** 交叉熵（logits + 类别索引）。 */
function crossEntropy(logits, targets) {
  const [R, C] = logits.shape;
  if (targets.length !== R) throw new Error('targets 长度必须等于 batch');
  // log-softmax 稳定化：log p_i = x_i − logsumexp(x)
  const lse = new Float64Array(R);
  for (let i = 0; i < R; i++) {
    const row = new Float64Array(C);
    for (let j = 0; j < C; j++) row[j] = logits.data[i * C + j];
    lse[i] = L0.logsumexp(Array.from(row));
  }
  let total = 0;
  for (let i = 0; i < R; i++) total += lse[i] - logits.data[i * C + targets[i]];
  const loss = total / R;
  const out = new Float64Array([loss]);
  return node(out, [1], [logits], (g) => {
    if (!logits.requiresGrad) return;
    const gl = new Float64Array(R * C);
    const c = g[0] / R;
    for (let i = 0; i < R; i++) {
      for (let j = 0; j < C; j++) {
        const p = Math.exp(logits.data[i * C + j] - lse[i]);
        gl[i * C + j] = c * (p - (j === targets[i] ? 1 : 0));
      }
    }
    logits._accum(gl);
  }, { op: 'cross_entropy' });
}

/** 均方误差。 */
function mse(pred, target) {
  const d = sub(pred, target);
  return mean(square(d));
}

// ── 梯度检验（把 tape 梯度与中心差分对拍）──────────────────────────────────
/**
 * 对任意 (params) => loss 的函数做梯度检验。
 * 这是 L1 的核心门禁工具：所有手写 backward 都必须过这一关。
 */
function gradCheckFn(f, params, opts = {}) {
  const { eps = 1e-5, rtol = 1e-5, atolScale = 1e-6 } = opts;
  // 解析梯度
  const params2 = params.map((p) => {
    const t = p instanceof Tensor ? p : T(p, [p.length]);
    t.requiresGrad = true; t.zeroGrad();
    return t;
  });
  const lossT = f(params2);
  lossT.backward();
  params2.forEach((p, i) => {
    if (p.grad === null) {
      throw new Error(`gradCheckFn: 第 ${i} 个参数反向后梯度为 null —— `
        + `它不在计算图里（常见原因：在 f 内部用 .data 另建了 Tensor，把参数踢出了图）`);
    }
  });
  const analytic = params2.map((p) => Float64Array.from(p.grad));

  // 数值梯度（中心差分）
  const numeric = params2.map((p) => new Float64Array(p.size));
  const saved = params2.map((p) => Float64Array.from(p.data));
  params2.forEach((p, pi) => {
    for (let i = 0; i < p.size; i++) {
      const old = p.data[i];
      p.data[i] = old + eps;
      const fp = f(params2.map((q) => (q.requiresGrad ? q : q))).item();
      p.data[i] = old - eps;
      const fm = f(params2.map((q) => q)).item();
      p.data[i] = old;
      numeric[pi][i] = (fp - fm) / (2 * eps);
    }
  });
  params2.forEach((p, pi) => { p.data.set(saved[pi]); });

  // 逐参数用 gradcheck 混合判据
  const results = [];
  for (let pi = 0; pi < params2.length; pi++) {
    const r = L0.gradcheck(analytic[pi], numeric[pi], { rtol, atolScale });
    results.push({ index: pi, ...r });
  }
  const allPass = results.every((r) => r.pass);
  const maxRel = Math.max(...results.map((r) => r.maxRel));
  return { pass: allPass, maxRel, results, analytic, numeric };
}

module.exports = {
  Tensor, T, node, shapeSize, shapeStrides, shapeEq,
  add, sub, mul, div,
  exp, log, neg, tanh, relu, sigmoid, sqrt, square, gelu, silu,
  matmul, transpose, reshape, sum, mean, softmaxRows, crossEntropy, mse,
  gradCheckFn,
};
