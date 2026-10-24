/**
 * NEXUS · L2 架构原语
 * ---------------------------------------------------------------------------
 * 每个原语给两套实现：**tape 版（金标准）** 与 **手写版（性能路径）**，由门禁强制互证。
 * 作者：晨星 (CJX0712) · MIT
 *
 * 选型（全部取自世界已验证成果，并标注被否决的替代）：
 *
 *   模块       采用                              被否决          理由
 *   ─────────────────────────────────────────────────────────────────────────
 *   归一化     RMSNorm (Zhang & Sennrich 2019)    LayerNorm       需减均值+偏置，参数更多、
 *                                                                推理更慢，大模型上无质量收益
 *   位置编码   RoPE (Su et al. 2021, RoFormer)   ALiBi / 绝对位置 长上下文外推弱于 RoPE+YaRN
 *              HF rotate_half 布局                               绝对位置几乎无外推能力
 *   前馈       SwiGLU (Shazeer 2020)             ReLU/GELU FFN   同等参数下 ppl 明显更差
 *   注意力     GQA (Ainslie 2023)                MHA             KV cache 随头数线性膨胀
 *              （MLA 见 L3，本层保留 GQA 路径）
 *
 * 复现 tianji-ai L3 的教训，本层有三处**必须照做否则静默出错**的地方：
 *   1. 因果掩码必须**右对齐**：j ≤ i + (Tkv − Tq)。左对齐在增量解码的第 2 步起就错，
 *      而且不报错 —— 这是最难查的一类缺陷。
 *   2. 残差分支的 out_proj 按 1/√(2·n_layers) 缩放（GPT-2 起的 residual scaling），
 *      否则深层激活方差爆炸。
 *   3. 反传顺序必须"先过子层、再过归一化"。反过来写形状全对、梯度全错，
 *      模型会"训练"，只是永远训不好。
 */

'use strict';
const L0 = require('./l0-numerics.js');
const A = require('./l1-autodiff.js');
const { T, add, mul, matmul, transpose, reshape, sum, mean, tanh, silu, square, sqrt, softmaxRows } = A;

// ═══════════════════════════════════════════════════════════════════════════
// RMSNorm
// ═══════════════════════════════════════════════════════════════════════════
/**
 * y = x / RMS(x) · g，其中 RMS(x) = sqrt(mean(x²) + eps)
 * 反向（手写）：
 *   r = RMS(x)；x̂ = x/r
 *   ∂L/∂g = gy · x̂
 *   ∂L/∂x = gy·g/r − x̂ · (Σ(gy·g·x̂)/d) / r
 */
function rmsNormFast(x, g, eps = 1e-6) {
  const [B, Tn, D] = x.shape;
  const outData = new Float64Array(x.size);
  const rms = new Float64Array(B * Tn);       // 缓存，反传要用
  const xhat = new Float64Array(x.size);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < Tn; t++) {
      const off = (b * Tn + t) * D;
      let ss = 0;
      for (let i = 0; i < D; i++) ss += x.data[off + i] ** 2;
      const r = Math.sqrt(ss / D + eps);
      rms[b * Tn + t] = r;
      for (let i = 0; i < D; i++) {
        const xh = x.data[off + i] / r;
        xhat[off + i] = xh;
        outData[off + i] = xh * g.data[i];
      }
    }
  }
  return A.node(outData, [B, Tn, D], [x, g], (gy) => {
    const gg = new Float64Array(D);
    const gx = new Float64Array(x.size);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < Tn; t++) {
        const off = (b * Tn + t) * D;
        const r = rms[b * Tn + t];
        let dot = 0;
        for (let i = 0; i < D; i++) dot += gy[off + i] * g.data[i] * xhat[off + i];
        for (let i = 0; i < D; i++) {
          const gygi = gy[off + i] * g.data[i];
          gg[i] += gy[off + i] * xhat[off + i];
          gx[off + i] = gygi / r - xhat[off + i] * (dot / D) / r;
        }
      }
    }
    if (g.requiresGrad) g._accum(gg);
    if (x.requiresGrad) x._accum(gx);
  }, { op: 'rmsNorm' });
}

/** tape 版 RMSNorm —— 金标准，写法直白、出错面小，用来给 rmsNormFast 当尺子。 */
function rmsNormTape(x, g, eps = 1e-6) {
  const [B, Tn, D] = x.shape;
  const sq = mul(x, x);                                     // [B,T,D]
  // L1 的归约只支持 2D 沿轴，3D 逐行归约在这里显式构造（仍走 tape，保持可微）
  const meanSq = new Float64Array(B * Tn);
  for (let bt = 0; bt < B * Tn; bt++) {
    let s = 0; for (let i = 0; i < D; i++) s += sq.data[bt * D + i];
    meanSq[bt] = s / D;
  }
  // 用 tape 把标量序列接回图：手写 node 保持可微
  const rmsT = A.node(new Float64Array(B * Tn), [B, Tn], [sq], (g2) => {
    if (!sq.requiresGrad) return;
    const gsq = new Float64Array(sq.size);
    for (let bt = 0; bt < B * Tn; bt++) {
      const r = Math.sqrt(meanSq[bt] + eps);
      const c = g2[bt] * (1 / (2 * r)) / D;
      for (let i = 0; i < D; i++) gsq[bt * D + i] = c;
    }
    sq._accum(gsq);
  }, { op: 'rms_from_sq' });
  for (let bt = 0; bt < B * Tn; bt++) rmsT.data[bt] = Math.sqrt(meanSq[bt] + eps);

  const rmsExp = A.node(new Float64Array(B * Tn * D), [B, Tn, D], [rmsT], (g2) => {
    if (!rmsT.requiresGrad) return;
    const gr = new Float64Array(B * Tn);
    for (let bt = 0; bt < B * Tn; bt++) {
      let s = 0; for (let i = 0; i < D; i++) s += g2[bt * D + i];
      gr[bt] = s;
    }
    rmsT._accum(gr);
  }, { op: 'expand_rms' });
  for (let bt = 0; bt < B * Tn; bt++) for (let i = 0; i < D; i++) rmsExp.data[bt * D + i] = rmsT.data[bt];

  const xhatT = div2(x, rmsExp);
  return mul(xhatT, A.reshape(g, [1, 1, D]));
}
/** 逐元素除（走 _mapBinary 的 div）。 */
function div2(a, b) { return A.div(a, b); }

// ═══════════════════════════════════════════════════════════════════════════
// RoPE（旋转位置编码）
// ═══════════════════════════════════════════════════════════════════════════
/**
 * HF rotate_half 布局：把 d 维分成前后两半，
 *   [x1, x2] → [x1·cos − x2·sin, x2·cos + x1·sin]
 * 频率：θ_i = 1 / base^(2i/d)，i ∈ [0, d/2)
 *
 * **核心不变量**：⟨RoPE(x,m), RoPE(y,n)⟩ 只依赖 (m − n)，与绝对位置无关。
 * 这条不变量是 RoPE 能做长度外推的全部根据，门禁 G33 会机检。
 */
function ropeFreqs(dim, maxPos, base = 10000) {
  const inv = new Float64Array(dim / 2);
  for (let i = 0; i < dim / 2; i++) inv[i] = 1 / Math.pow(base, (2 * i) / dim);
  return inv;
}

function ropeApply(x, startPos, inv, base = 10000) {
  // x: [B, T, H, Dh]（最后一维是头维度）
  // 统一按 [B, T, H, Dh] 解析（也接受 [T, H, Dh] 与 [T, Dh]）
  const shape = x.shape;
  const Dh = shape[shape.length - 1];
  let B, Tlen, H;
  if (shape.length === 4) { [B, Tlen, H] = shape; }
  else if (shape.length === 3) { B = 1; [Tlen, H] = shape; }
  else { B = 1; H = 1; Tlen = shape[0]; }
  const out = new Float64Array(x.size);
  const half = Dh / 2;
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < Tlen; t++) {
      const pos = startPos + t;
      for (let h = 0; h < H; h++) {
        const off = ((b * Tlen + t) * H + h) * Dh;
        for (let i = 0; i < half; i++) {
          const freq = inv[i];
          const ang = pos * freq;
          const c = Math.cos(ang), s = Math.sin(ang);
          const x1 = x.data[off + i], x2 = x.data[off + i + half];
          out[off + i] = x1 * c - x2 * s;
          out[off + i + half] = x2 * c + x1 * s;
        }
      }
    }
  }
  return A.node(out, shape, [x], (g) => {
    if (!x.requiresGrad) return;
    const gx = new Float64Array(x.size);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < Tlen; t++) {
        const pos = startPos + t;
        for (let h = 0; h < H; h++) {
          const off = ((b * Tlen + t) * H + h) * Dh;
          for (let i = 0; i < half; i++) {
            const ang = pos * inv[i];
            const c = Math.cos(ang), s = Math.sin(ang);
            // 旋转的反向是逆旋转：[[c, s], [−s, c]] 作用于 [g1, g2]
            const g1 = g[off + i], g2 = g[off + i + half];
            gx[off + i] = g1 * c + g2 * s;
            gx[off + i + half] = g2 * c - g1 * s;
          }
        }
      }
    }
    x._accum(gx);
  }, { op: 'rope' });
}

/** YaRN 外推的λ/θ重标定（Peng et al. 2023）—— 长上下文时把高频做插值、低频保原样。 */
function yarnScale(inv, dim, ctxOriginal, ctxExtended, betaFast = 32, betaSlow = 1) {
  const corrDim = (d) => {
    const lo = 2 * Math.floor(d / 2 * Math.log(ctxOriginal / (2 * Math.PI * betaSlow)) / Math.log(ctxExtended / ctxOriginal));
    const hi = 2 * Math.ceil(d / 2 * Math.log(ctxOriginal / (2 * Math.PI * betaFast)) / Math.log(ctxExtended / ctxOriginal)) - 1;
    return [Math.max(0, Math.min(lo, d / 2 - 1)), Math.max(0, Math.min(hi, d / 2 - 1))];
  };
  const [lo, hi] = corrDim(dim);
  const out = new Float64Array(dim / 2);
  const scale = ctxOriginal / ctxExtended;
  for (let i = 0; i < dim / 2; i++) {
    if (i <= hi && i >= lo) {
      const ramp = (i - lo) / Math.max(1e-9, hi - lo);
      out[i] = inv[i] * ((1 - ramp) * scale + ramp);
    } else if (i < lo) {
      out[i] = inv[i] * scale;
    } else {
      out[i] = inv[i];
    }
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// SwiGLU
// ═══════════════════════════════════════════════════════════════════════════
/** SwiGLU(x) = (x·W_gate 经 SiLU) ⊙ (x·W_up)，再过 W_down。 */
function swigluTape(x, wGate, wUp, wDown) {
  const g1 = matmul(x, wGate);
  const g2 = matmul(x, wUp);
  return matmul(mul(silu(g1), g2), wDown);
}

// ═══════════════════════════════════════════════════════════════════════════
// 因果注意力（GQA）
// ═══════════════════════════════════════════════════════════════════════════
/**
 * GQA：Q 有 H 个头，KV 只有 Hkv 个头（H % Hkv == 0），每个 KV 头服务 H/Hkv 个 Q 头。
 * 相比 MHA，KV cache 缩小 H/Hkv 倍；相比 MQA（Hkv=1）质量损失更小。
 *
 * @param q [B, Tq, H, Dh]
 * @param k [B, Tkv, Hkv, Dh]
 * @param v [B, Tkv, Hkv, Dv]   ← 允许 Dv ≠ Dh（MLA 需要这个泛化）
 * @param startPos 增量解码时 q 的起始绝对位置；掩码按**右对齐**构造
 */
function causalAttention(q, k, v, opts = {}) {
  const { scale = null, startPos = 0, returnSoftmax = false } = opts;
  const [B, Tq, H, Dh] = q.shape;
  const [, Tkv, Hkv, Dk] = k.shape;
  const Dv = v.shape[3];
  if (H % Hkv !== 0) throw new Error(`GQA 要求 H % Hkv == 0，实得 H=${H} Hkv=${Hkv}`);
  const grp = H / Hkv;
  const sc = scale === null ? 1 / Math.sqrt(Dk) : scale;

  const outData = new Float64Array(B * Tq * H * Dv);
  // 反传需要 softmax 输出，缓存下来
  const probs = new Float64Array(B * H * Tq * Tkv);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const kvHead = Math.floor(h / grp);
      for (let i = 0; i < Tq; i++) {
        // 右对齐因果掩码：query 绝对位置 = startPos + i，可见 key 位置 ≤ startPos + i
        // 即 j ≤ i + (Tkv − Tq) ⇒ j ≤ startPos + i − (startPos + Tq − Tkv)
        const qi = startPos + i;
        const kBase = startPos + Tq - Tkv;             // key 0 对应的绝对位置
        const jMax = qi - kBase;                        // 最大可见 key 下标
        const rowOff = ((b * H + h) * Tq + i) * Tkv;
        // logits
        let m = -Infinity;
        const logits = new Float64Array(Tkv);
        for (let j = 0; j < Tkv; j++) {
          if (j > jMax) { logits[j] = -Infinity; continue; }
          let s = 0;
          const qOff = ((b * Tq + i) * H + h) * Dh;
          const kOff = ((b * Tkv + j) * Hkv + kvHead) * Dk;
          for (let d = 0; d < Dk; d++) s += q.data[qOff + d] * k.data[kOff + d];
          const lg = s * sc;
          logits[j] = lg;
          if (lg > m) m = lg;
        }
        // softmax（掩码位置保持 0）
        let z = 0;
        for (let j = 0; j < Tkv; j++) {
          if (logits[j] === -Infinity) { probs[rowOff + j] = 0; continue; }
          const e = Math.exp(logits[j] - m);
          probs[rowOff + j] = e; z += e;
        }
        for (let j = 0; j < Tkv; j++) probs[rowOff + j] /= z;
        // 加权和
        for (let d = 0; d < Dv; d++) {
          let s = 0;
          for (let j = 0; j < Tkv; j++) {
            if (probs[rowOff + j] === 0) continue;
            s += probs[rowOff + j] * v.data[((b * Tkv + j) * Hkv + kvHead) * Dv + d];
          }
          outData[((b * Tq + i) * H + h) * Dv + d] = s;
        }
      }
    }
  }

  const out = A.node(outData, [B, Tq, H, Dv], [q, k, v], (go) => {
    const gq = new Float64Array(q.size);
    const gk = new Float64Array(k.size);
    const gv = new Float64Array(v.size);
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        const kvHead = Math.floor(h / grp);
        for (let i = 0; i < Tq; i++) {
          const qOff = ((b * Tq + i) * H + h) * Dh;
          const oOff = ((b * Tq + i) * H + h) * Dv;
          const rowOff = ((b * H + h) * Tq + i) * Tkv;
          // dP = go · Vᵀ
          const dp = new Float64Array(Tkv);
          for (let j = 0; j < Tkv; j++) {
            let s = 0;
            const vOff = ((b * Tkv + j) * Hkv + kvHead) * Dv;
            for (let d = 0; d < Dv; d++) s += go[oOff + d] * v.data[vOff + d];
            dp[j] = s;
          }
          // softmax 反向：dx = p·(dp − Σ p·dp)
          let dot = 0;
          for (let j = 0; j < Tkv; j++) dot += probs[rowOff + j] * dp[j];
          for (let j = 0; j < Tkv; j++) {
            const p = probs[rowOff + j];
            if (p === 0) continue;
            const dlogit = (p * (dp[j] - dot)) * sc;
            const kOff = ((b * Tkv + j) * Hkv + kvHead) * Dk;
            const vOff = ((b * Tkv + j) * Hkv + kvHead) * Dv;
            for (let d = 0; d < Dk; d++) {
              gq[qOff + d] += dlogit * k.data[kOff + d];
              gk[kOff + d] += dlogit * q.data[qOff + d];
            }
            for (let d = 0; d < Dv; d++) gv[vOff + d] += go[oOff + d] * p;
          }
        }
      }
    }
    if (q.requiresGrad) q._accum(gq);
    if (k.requiresGrad) k._accum(gk);
    if (v.requiresGrad) v._accum(gv);
  }, { op: 'causal_attention' });

  if (returnSoftmax) out._probs = probs;
  return out;
}

/** 朴素全量注意力（无 GQA、无掩码优化）—— 给 causalAttention 当对照金标准。 */
function naiveAttention(q, k, v, opts = {}) {
  const { scale = null } = opts;
  const [B, Tq, H, Dh] = q.shape;
  const [, Tkv, Hkv] = k.shape;
  const Dv = v.shape[3];
  if (H !== Hkv) throw new Error('naiveAttention 只支持 H == Hkv（MHA）');
  const sc = scale === null ? 1 / Math.sqrt(Dh) : scale;
  const out = new Float64Array(B * Tq * H * Dv);
  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) for (let i = 0; i < Tq; i++) {
    const logits = [];
    for (let j = 0; j < Tkv; j++) {
      if (j > i + (Tkv - Tq)) { logits.push(-Infinity); continue; }
      let s = 0;
      for (let d = 0; d < Dh; d++) {
        s += q.data[((b * Tq + i) * H + h) * Dh + d] * k.data[((b * Tkv + j) * H + h) * Dh + d];
      }
      logits.push(s * sc);
    }
    const m = Math.max(...logits.filter(Number.isFinite));
    let z = 0; const p = [];
    for (const lg of logits) { if (!Number.isFinite(lg)) { p.push(0); continue; } const e = Math.exp(lg - m); p.push(e); z += e; }
    for (let j = 0; j < Tkv; j++) p[j] /= z;
    for (let d = 0; d < Dv; d++) {
      let s = 0;
      for (let j = 0; j < Tkv; j++) s += p[j] * v.data[((b * Tkv + j) * H + h) * Dv + d];
      out[((b * Tq + i) * H + h) * Dv + d] = s;
    }
  }
  return A.node(out, [B, Tq, H, Dv], [q, k, v], () => {
    throw new Error('naiveAttention 仅作前向金标准，不支持反传（反传请用 causalAttention）');
  }, { op: 'naive_attention', requiresGrad: false });
}

// ═══════════════════════════════════════════════════════════════════════════
// 残差缩放初始化（GPT-2 起的配方）
// ═══════════════════════════════════════════════════════════════════════════
/** out_proj 初始化缩放：std /= √(2·n_layers)，让深层激活方差保持 O(1)。 */
function residualScaleStd(std, nLayers) { return std / Math.sqrt(2 * nLayers); }

module.exports = {
  rmsNormFast, rmsNormTape,
  ropeFreqs, ropeApply, yarnScale,
  swigluTape,
  causalAttention, naiveAttention,
  residualScaleStd,
};
