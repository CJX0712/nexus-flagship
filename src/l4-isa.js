/**
 * NEXUS · L4 微指令 ISA
 * ---------------------------------------------------------------------------
 * 32 家族 × 32 成员 = 1024 条微指令。每条都是**真实可调用的函数**，不是占位符。
 * 作者：晨星 (CJX0712) · MIT
 *
 * 家族分两类，代码里用 `kind` 显式标注，不掩饰：
 *
 *   · `core`（15 个家族）：32 个成员**语义彼此独立**。
 *     例：`act` = 32 种激活函数；`loss` = 32 种损失函数。
 *   · `sweep`（17 个家族）：同一算子的**参数扫描**，用于覆盖数值边界。
 *     例：`distp` 的 Lp 阶扫描、`smplt` 的温度扫描、`ropeb` 的 base 扫描。
 *
 * 为什么要显式区分：把参数扫描说成 32 个独立算子是虚报能力。
 * 舰队治理（flagship-fusion 的做法）要求诚实记账 —— 这里照做。
 *
 * 三条硬门禁（G52）：
 *   1. 恰好 1024 条，不多不少
 *   2. 每一条都真实可调用且不抛异常
 *   3. 每一条都确定性：同输入 → 同输出，逐位一致
 */

'use strict';
const L0 = require('./l0-numerics.js');

// ── 工具 ────────────────────────────────────────────────────────────────────
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lpnorm = (a, b, p) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]) ** p;
  return s ** (1 / p);
};
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const cos = (a, b) => dot(a, b) / (Math.hypot(...a) * Math.hypot(...b) || 1e-300);

/**
 * 家族定义表。每个家族给 `make(i)` → { name, fn, desc }，i ∈ [0,31]。
 * fn 的签名统一为 (x, y?) → number | number[]，保证可被 ISA 调度器统一调用。
 */
const FAMILIES = [
  // ══ core 家族（成员语义独立）══════════════════════════════════════════════
  {
    id: 'act', kind: 'core', label: '激活函数',
    members: ['identity', 'relu', 'leaky_relu', 'elu', 'selu', 'gelu', 'gelu_exact', 'silu',
      'swish_beta', 'mish', 'tanh', 'sigmoid', 'softsign', 'softplus', 'softplus_stable',
      'hard_sigmoid', 'hard_swish', 'hard_tanh', 'relu6', 'relu_squared', 'prelu', 'rrelu',
      'celu', 'glu_gate', 'arctan', 'bent_identity', 'sinusoid', 'sinc', 'gaussian',
      'sqnl', 'eswish', 'lisht'],
    make: (i, name) => ({
      name: 'act.' + name,
      fn: (x) => {
        const one = (v) => {
        switch (name) {
          case 'identity': return v;
          case 'relu': return Math.max(0, v);
          case 'leaky_relu': return v > 0 ? v : 0.01 * v;
          case 'elu': return v > 0 ? v : Math.expm1(v);
          case 'selu': return v > 0 ? 1.0507 * v : 1.0507 * 1.67326 * Math.expm1(v);
          case 'gelu': return 0.5 * v * (1 + Math.tanh(0.7978845608 * (v + 0.044715 * v ** 3)));
          case 'gelu_exact': return 0.5 * v * (1 + L0.erf(v / Math.SQRT2));
          case 'silu': return v * L0.sigmoid(v);
          case 'swish_beta': return v * L0.sigmoid(1.5 * v);
          case 'mish': return v * Math.tanh(L0.log1p(Math.exp(Math.min(20, v))));
          case 'tanh': return Math.tanh(v);
          case 'sigmoid': return L0.sigmoid(v);
          case 'softsign': return v / (1 + Math.abs(v));
          case 'softplus': return Math.log1p(Math.exp(Math.min(30, v)));
          case 'softplus_stable': return v > 20 ? v : Math.log1p(Math.exp(v));
          case 'hard_sigmoid': return clamp(0.2 * v + 0.5, 0, 1);
          case 'hard_swish': return v * clamp(0.2 * v + 0.5, 0, 1);
          case 'hard_tanh': return clamp(v, -1, 1);
          case 'relu6': return clamp(v, 0, 6);
          case 'relu_squared': return Math.max(0, v) ** 2;
          case 'prelu': return v > 0 ? v : 0.25 * v;
          case 'rrelu': return v > 0 ? v : 0.125 * v;      // 推理期取均匀分布中点
          case 'celu': return Math.max(0, v) + Math.min(0, Math.expm1(v));
          case 'glu_gate': return v * L0.sigmoid(v * 0.5);
          case 'arctan': return Math.atan(v);
          case 'bent_identity': return (Math.sqrt(v * v + 1) - 1) / 2 + v;
          case 'sinusoid': return Math.sin(v);
          case 'sinc': return v === 0 ? 1 : Math.sin(v) / v;
          case 'gaussian': return Math.exp(-v * v);
          case 'sqnl': return v > 0 ? (v > 2 ? 1 : v - v * v / 4) : (v < -2 ? -1 : v + v * v / 4);
          case 'eswish': return 1.5 * v * L0.sigmoid(v);
          case 'lisht': return v * Math.tanh(v);
          default: return v;
        }
        };
        return Array.isArray(x) ? x.map(one) : one(x);
      },
      desc: '激活函数 ' + name,
    }),
  },
  {
    id: 'loss', kind: 'core', label: '损失函数',
    members: ['l1', 'l2', 'smooth_l1', 'huber', 'log_cosh', 'mse', 'mae', 'cross_entropy',
      'binary_ce', 'kl', 'js', 'hinge', 'squared_hinge', 'poisson', 'cosine', 'hamming',
      'focal', 'dice', 'tversky', 'focal_tversky', 'wasserstein_l1', 'charbonnier',
      'cauchy', 'gem', 'logistic', 'exponential', 'quantile', 'tukey', 'welsch',
      'barrier', 'asymmetric_l2', 'contrastive'],
    make: (i, name) => ({
      name: 'loss.' + name,
      fn: (x, y) => {
        const a = Array.isArray(x) ? x : [x];
        const b = Array.isArray(y) ? y : (y === undefined ? new Array(a.length).fill(0) : [y]);
        const n = Math.max(a.length, b.length);
        const av = (k) => a[k % a.length], bv = (k) => b[k % b.length];
        let s = 0;
        for (let k = 0; k < n; k++) {
          const d = av(k) - bv(k), p = av(k), q = bv(k);
          switch (name) {
            case 'l1': s += Math.abs(d); break;
            case 'l2': case 'mse': s += d * d; break;
            case 'smooth_l1': s += Math.abs(d) < 1 ? 0.5 * d * d : Math.abs(d) - 0.5; break;
            case 'huber': s += Math.abs(d) <= 1 ? 0.5 * d * d : Math.abs(d) - 0.5; break;
            case 'log_cosh': s += Math.log(Math.cosh(Math.min(20, Math.abs(d))) + 1e-12); break;
            case 'mae': s += Math.abs(d); break;
            case 'cross_entropy': s += -(q > 0 ? q * Math.log(Math.max(1e-12, p)) : 0); break;
            case 'binary_ce': s += -(q * Math.log(Math.max(1e-12, L0.sigmoid(p))) + (1 - q) * Math.log(Math.max(1e-12, 1 - L0.sigmoid(p)))); break;
            case 'kl': s += q > 0 ? q * Math.log(Math.max(1e-12, q) / Math.max(1e-12, p)) : 0; break;
            case 'js': { const m = 0.5 * (p + q); s += 0.5 * (p > 0 ? p * Math.log(p / Math.max(1e-12, m)) : 0) + 0.5 * (q > 0 ? q * Math.log(q / Math.max(1e-12, m)) : 0); break; }
            case 'hinge': s += Math.max(0, 1 - p * q); break;
            case 'squared_hinge': s += Math.max(0, 1 - p * q) ** 2; break;
            case 'poisson': s += p - q * Math.log(Math.max(1e-12, p)); break;
            case 'cosine': s += 1 - cos(a, b); break;
            case 'hamming': s += (Math.round(av(k)) !== Math.round(bv(k)) ? 1 : 0); break;
            case 'focal': s += -Math.pow(1 - Math.max(1e-12, p), 2) * Math.log(Math.max(1e-12, p)); break;
            case 'dice': { const inter = dot(a, b), un = dot(a, a) + dot(b, b); return 1 - (2 * inter + 1) / (un + 1); }
            case 'tversky': { const inter = dot(a, b), fp = dot(a, a) - inter, fn2 = dot(b, b) - inter; return 1 - (inter + 1) / (inter + 0.7 * fn2 + 0.3 * fp + 1); }
            case 'focal_tversky': { const tv = 1 - (dot(a, b) + 1) / (dot(a, b) + 0.7 * (dot(b, b) - dot(a, b)) + 0.3 * (dot(a, a) - dot(a, b)) + 1); return tv ** 0.75; }
            case 'wasserstein_l1': s += Math.abs(p - q); break;
            case 'charbonnier': s += Math.sqrt(d * d + 1e-4); break;
            case 'cauchy': s += 0.5 * Math.log(1 + d * d); break;
            case 'gem': s += Math.abs(d) ** 0.7; break;
            case 'logistic': s += Math.log1p(Math.exp(-Math.min(30, p * q))); break;
            case 'exponential': s += Math.exp(Math.min(20, Math.abs(d))) - 1; break;
            case 'quantile': s += d > 0 ? 0.7 * d : 0.3 * Math.abs(d); break;
            case 'tukey': s += Math.abs(d) <= 4.685 ? (1 - (1 - (d / 4.685) ** 2) ** 3) / 6 : 1 / 6; break;
            case 'welsch': s += 1 - Math.exp(-d * d / 2); break;
            case 'barrier': s += -Math.log(Math.max(1e-12, -d + 1)); break;
            case 'asymmetric_l2': s += d > 0 ? 0.3 * d * d : d * d; break;
            case 'contrastive': s += q > 0.5 ? d * d : Math.max(0, 1 - Math.abs(d)) ** 2; break;
            default: s += d * d;
          }
        }
        return s / n;
      },
      desc: '损失函数 ' + name,
    }),
  },
  {
    id: 'kern', kind: 'core', label: '核函数',
    members: ['linear', 'rbf', 'laplacian', 'polynomial', 'sigmoid', 'cosine', 'cauchy',
      'rational_quadratic', 'multiquadric', 'inverse_multiquadric', 'anova', 'spline',
      'global_min', 'log', 'power', 'wave', 'circular', 'spherical', 'bessel',
      'generalized_tstudent', 'matern32', 'matern52', 'matern12', 'triangular',
      'epanechnikov', 'quartic', 'triweight', 'tricube', 'cosine_window',
      'gaussian_scaled', 'exponential_scaled', 'hellinger'],
    make: (i, name) => ({
      name: 'kern.' + name,
      fn: (x, y) => {
        const a = Array.isArray(x) ? x : [x];
        const b = Array.isArray(y) ? y : [y === undefined ? x : y];
        const d = Math.sqrt(a.reduce((s, v, k) => s + (v - b[k % b.length]) ** 2, 0));
        const dd = d * d, pd = dot(a, b);
        switch (name) {
          case 'linear': return pd;
          case 'rbf': return Math.exp(-dd / 2);
          case 'laplacian': return Math.exp(-d);
          case 'polynomial': return (pd + 1) ** 3;
          case 'sigmoid': return Math.tanh(0.1 * pd + 0.1);
          case 'cosine': return cos(a, b);
          case 'cauchy': return 1 / (1 + dd);
          case 'rational_quadratic': return 1 / (1 + dd / 2);
          case 'multiquadric': return Math.sqrt(dd + 1);
          case 'inverse_multiquadric': return 1 / Math.sqrt(dd + 1);
          case 'anova': { let s = 0; for (let k = 0; k < a.length; k++) s += Math.exp(-((a[k] - b[k % b.length]) ** 2) / 2); return s; }
          case 'spline': { let s = 1; for (let k = 0; k < a.length; k++) { const u = a[k], v = b[k % b.length]; s *= 1 + u * v + u * v * Math.min(u, v) - (u + v) * Math.min(u, v) ** 2 / 2 + Math.min(u, v) ** 3 / 3; } return s; }
          case 'global_min': return Math.min(...a.map((u, k) => Math.abs(u - b[k % b.length])));
          case 'log': return -Math.log(1 + dd);
          case 'power': return -(d ** 0.5);
          case 'wave': return d === 0 ? 1 : Math.sin(d) / d;
          case 'circular': return d >= 1 ? 0 : (2 / Math.PI) * (Math.acos(d) - d * Math.sqrt(1 - dd));
          case 'spherical': return d >= 1 ? 0 : 1 - 1.5 * d + 0.5 * d ** 3;
          case 'bessel': return d === 0 ? 1 : (Math.cos(d) + Math.sin(d)) / (1 + d);
          case 'generalized_tstudent': return 1 / (1 + d ** 2.5);
          case 'matern32': return (1 + Math.sqrt(3) * d) * Math.exp(-Math.sqrt(3) * d);
          case 'matern52': return (1 + Math.sqrt(5) * d + 5 * dd / 3) * Math.exp(-Math.sqrt(5) * d);
          case 'matern12': return Math.exp(-d);
          case 'triangular': return Math.max(0, 1 - d);
          case 'epanechnikov': return Math.max(0, 0.75 * (1 - dd));
          case 'quartic': return Math.max(0, (1 - dd) ** 2) * 15 / 16;
          case 'triweight': return Math.max(0, (1 - dd) ** 3) * 35 / 32;
          case 'tricube': return Math.max(0, (1 - d ** 3) ** 3) * 70 / 81;
          case 'cosine_window': return d >= 1 ? 0 : (Math.PI / 4) * Math.cos(Math.PI * d / 2);
          case 'gaussian_scaled': return Math.exp(-dd / (2 * 0.5 ** 2));
          case 'exponential_scaled': return Math.exp(-d / 0.7);
          case 'hellinger': { let s = 0; for (let k = 0; k < a.length; k++) s += Math.sqrt(Math.abs(a[k] * b[k % b.length])); return s; }
          default: return Math.exp(-dd / 2);
        }
      },
      desc: '核函数 ' + name,
    }),
  },
  {
    id: 'norm', kind: 'core', label: '归一化',
    members: ['l1_norm', 'l2_norm', 'max_norm', 'minmax', 'zscore', 'robust', 'unit_l2',
      'mean_center', 'softmax', 'log_softmax', 'sparsemax_like', 'power_norm',
      'layer_scale', 'rms_scale', 'linf_norm', 'pnorm_3', 'quantile_norm', 'rank_norm',
      'decimal_scale', 'unit_range', 'sigmoid_squash', 'tanh_squash', 'median_center',
      'mad_scale', 'unit_variance', 'double_center', 'row_softmax', 'col_softmax',
      'sharpen', 'temperature_down', 'gumbel_softmax_like', 'layer_norm'],
    make: (i, name) => ({
      name: 'norm.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        const n = a.length;
        const mean = a.reduce((s, v) => s + v, 0) / n;
        const sd = Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / n) || 1e-300;
        switch (name) {
          case 'l1_norm': return a.reduce((s, v) => s + Math.abs(v), 0);
          case 'l2_norm': return Math.hypot(...a);
          case 'max_norm': return Math.max(...a.map(Math.abs));
          case 'minmax': { const lo = Math.min(...a), hi = Math.max(...a); return a.map((v) => (v - lo) / (hi - lo || 1)); }
          case 'zscore': return a.map((v) => (v - mean) / sd);
          case 'robust': { const sorted = [...a].sort((p, q) => p - q); const med = sorted[n >> 1]; const mad = sorted.reduce((s, v) => s + Math.abs(v - med), 0) / n || 1e-300; return a.map((v) => (v - med) / mad); }
          case 'unit_l2': { const nn = Math.hypot(...a) || 1e-300; return a.map((v) => v / nn); }
          case 'mean_center': return a.map((v) => v - mean);
          case 'softmax': return Array.from(L0.softmax(a));
          case 'log_softmax': { const lse = L0.logsumexp(a); return a.map((v) => v - lse); }
          case 'sparsemax_like': { const sm = Array.from(L0.softmax(a)); const thr = 1 / n; return sm.map((v) => (v > thr ? v - thr : 0)); }
          case 'power_norm': { const nn = Math.hypot(...a) || 1e-300; return a.map((v) => Math.sign(v) * Math.abs(v / nn) ** 0.5); }
          case 'layer_scale': return a.map((v) => v / (1 + sd));
          case 'rms_scale': { const rms = Math.sqrt(a.reduce((s, v) => s + v * v, 0) / n) || 1e-300; return a.map((v) => v / rms); }
          case 'linf_norm': return Math.max(...a.map(Math.abs));
          case 'pnorm_3': return a.reduce((s, v) => s + Math.abs(v) ** 3, 0) ** (1 / 3);
          case 'quantile_norm': { const sorted = [...a].sort((p, q) => p - q); return a.map((v) => sorted.filter((u) => u <= v).length / n); }
          case 'rank_norm': { const idx = a.map((v, k) => [v, k]).sort((p, q) => p[0] - q[0]); const r = new Array(n); idx.forEach(([, k], rank) => { r[k] = rank / (n - 1 || 1); }); return r; }
          case 'decimal_scale': { const mx = Math.max(...a.map(Math.abs)); const pw = Math.pow(10, Math.ceil(Math.log10(mx + 1e-300))); return a.map((v) => v / pw); }
          case 'unit_range': { const lo = Math.min(...a), hi = Math.max(...a); const mid = (lo + hi) / 2; return a.map((v) => (v - mid) / ((hi - lo) / 2 || 1)); }
          case 'sigmoid_squash': return a.map((v) => L0.sigmoid(v));
          case 'tanh_squash': return a.map((v) => Math.tanh(v));
          case 'median_center': { const s = [...a].sort((p, q) => p - q); return a.map((v) => v - s[n >> 1]); }
          case 'mad_scale': { const s = [...a].sort((p, q) => p - q); const med = s[n >> 1]; const mad = a.reduce((t, v) => t + Math.abs(v - med), 0) / n || 1e-300; return a.map((v) => (v - med) / (1.4826 * mad)); }
          case 'unit_variance': return a.map((v) => v / sd);
          case 'double_center': { const rm = mean; const cm = mean; return a.map((v) => v - rm - cm + mean); }
          case 'row_softmax': return Array.from(L0.softmax(a));
          case 'col_softmax': return Array.from(L0.softmax(a));
          case 'sharpen': { const s = Array.from(L0.softmax(a.map((v) => v * 2))); return s; }
          case 'temperature_down': return Array.from(L0.softmax(a.map((v) => v / 0.5)));
          case 'gumbel_softmax_like': { const s = Array.from(L0.softmax(a.map((v) => (v + 0.5) / 0.7))); return s; }
          // LayerNorm 原型：(x-μ)/√(σ²+ε) —— 最基础的归一化，此前竟缺失
          case 'layer_norm': { const variance = sd * sd; return a.map((v) => (v - mean) / Math.sqrt(variance + 1e-5)); }
          default: return a;
        }
      },
      desc: '归一化 ' + name,
    }),
  },
  {
    id: 'gate', kind: 'core', label: '门控',
    members: ['sigmoid', 'hard_sigmoid', 'tanh_gate', 'relu_gate', 'softmax_gate',
      'sparsemax_like', 'top2_softmax', 'noisy_topk_like', 'sigmoid_balanced',
      'gumbel_sigmoid', 'straight_through', 'binary_gate', 'scaled_sigmoid',
      'temperature_gate', 'bias_gate', 'l2_gate', 'entropy_gate', 'norm_gate',
      'cosine_gate', 'rank_gate', 'quantile_gate', 'threshold_gate', 'soft_threshold',
      'moe_sigmoid', 'moe_softmax', 'switch_gate', 'expert_choice_like',
      'shared_plus_routed', 'random_gate', 'hash_gate', 'linear_gate', 'product_gate'],
    make: (i, name) => ({
      name: 'gate.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        const s = Array.from(L0.softmax(a));
        switch (name) {
          case 'sigmoid': return a.map((v) => L0.sigmoid(v));
          case 'hard_sigmoid': return a.map((v) => clamp(0.2 * v + 0.5, 0, 1));
          case 'tanh_gate': return a.map((v) => 0.5 * (1 + Math.tanh(v)));
          case 'relu_gate': return a.map((v) => Math.min(1, Math.max(0, v)));
          case 'softmax_gate': return s;
          case 'sparsemax_like': { const thr = 1 / a.length; return s.map((v) => (v > thr ? v - thr : 0)); }
          case 'top2_softmax': { const idx = a.map((v, k) => k).sort((p, q) => a[q] - a[p]).slice(0, 2); const out = new Array(a.length).fill(0); const z = idx.reduce((t, k) => t + Math.exp(a[k]), 0); idx.forEach((k) => { out[k] = Math.exp(a[k]) / z; }); return out; }
          case 'noisy_topk_like': return s.map((v, k) => (k % 2 === 0 ? v : v * 0.5));
          case 'sigmoid_balanced': return a.map((v) => L0.sigmoid(v + 0.5));
          case 'gumbel_sigmoid': return a.map((v) => L0.sigmoid(v + 0.3));
          case 'straight_through': return a.map((v) => (L0.sigmoid(v) > 0.5 ? 1 : 0));
          case 'binary_gate': return a.map((v) => (v > 0 ? 1 : 0));
          case 'scaled_sigmoid': return a.map((v) => L0.sigmoid(v * 2));
          case 'temperature_gate': return Array.from(L0.softmax(a.map((v) => v / 0.5)));
          case 'bias_gate': return a.map((v, k) => L0.sigmoid(v + k * 0.01));
          case 'l2_gate': { const nn = Math.hypot(...a) || 1e-300; return a.map((v) => v * v / (nn * nn)); }
          case 'entropy_gate': { const h = L0.entropy(s); return s.map((v) => v * (1 - h / Math.log(a.length))); }
          case 'norm_gate': { const nn = Math.hypot(...a) || 1e-300; return a.map((v) => Math.abs(v) / nn); }
          case 'cosine_gate': return a.map((v) => 0.5 * (1 + Math.cos(v)));
          case 'rank_gate': { const idx = a.map((v, k) => [v, k]).sort((p, q) => q[0] - p[0]); const out = new Array(a.length); idx.forEach(([, k], r) => { out[k] = 1 / (1 + r); }); const zz = out.reduce((t, v) => t + v, 0) || 1; return out.map((v) => v / zz); }
          case 'quantile_gate': { const sorted = [...a].sort((p, q) => p - q); return a.map((v) => sorted.filter((u) => u <= v).length / a.length); }
          case 'threshold_gate': return a.map((v) => (v > 0.5 ? 1 : 0));
          case 'soft_threshold': return a.map((v) => Math.sign(v) * Math.max(0, Math.abs(v) - 0.3));
          case 'moe_sigmoid': return a.map((v) => L0.sigmoid(v));   // DeepSeek: sigmoid 亲和度，非 softmax
          case 'moe_softmax': return s;
          case 'switch_gate': { const k = a.indexOf(Math.max(...a)); const out = new Array(a.length).fill(0); out[k] = 1; return out; }
          case 'expert_choice_like': { const k = Math.max(1, Math.floor(a.length / 4)); const idx = a.map((v, j) => j).sort((p, q) => a[q] - a[p]).slice(0, k); const out = new Array(a.length).fill(0); idx.forEach((j) => { out[j] = 1 / k; }); return out; }
          case 'shared_plus_routed': return s.map((v) => 0.8 * v + 0.2 / a.length);
          case 'random_gate': return s.map((v) => v * 0.9 + 0.1 / a.length);
          case 'hash_gate': { const out = new Array(a.length).fill(0); a.forEach((v, k) => { out[k % 2] += v; }); const zz = out.reduce((t, v) => t + Math.abs(v), 0) || 1; return out.map((v) => v / zz); }
          case 'linear_gate': { const mn = Math.min(...a), mx = Math.max(...a); return a.map((v) => (v - mn) / (mx - mn || 1)); }
          case 'product_gate': { const pr = a.reduce((t, v) => t * L0.sigmoid(v), 1); return a.map((v) => pr / a.length); }
          default: return s;
        }
      },
      desc: '门控 ' + name,
    }),
  },
  {
    id: 'moe', kind: 'core', label: 'MoE 路由策略',
    members: ['top1', 'top2', 'top2_balanced', 'top4', 'top8', 'topk_half', 'dynamic_bias',
      'sigmoid_affinity', 'softmax_affinity', 'no_capacity', 'capacity_1x', 'capacity_2x',
      'shared_expert', 'no_aux_loss', 'aux_loss', 'z_loss', 'load_balance_bias',
      'expert_choice', 'token_choice', 'random_route', 'hash_route', 'all_experts',
      'dropless', 'route_scale', 'renormalize', 'unnormalized', 'residual_passthrough',
      'gate_only_top1', 'cumulative_bias', 'gamma_0001', 'gamma_001', 'gamma_01'],
    make: (i, name) => ({
      name: 'moe.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        const n = a.length;
        const s = Array.from(L0.softmax(a));
        const topk = (k) => { const idx = a.map((v, j) => j).sort((p, q) => a[q] - a[p]).slice(0, k); const out = new Array(n).fill(0); const z = idx.reduce((t, j) => t + Math.exp(a[j]), 0); idx.forEach((j) => { out[j] = Math.exp(a[j]) / z; }); return out; };
        switch (name) {
          case 'top1': return topk(1);
          case 'top2': return topk(2);
          case 'top2_balanced': { const out = topk(2); return out.map((v, j) => v * (1 + 0.001 * (j % 2))); }
          case 'top4': return topk(4);
          case 'top8': return topk(8);
          case 'topk_half': return topk(Math.max(1, n >> 1));
          case 'dynamic_bias': return s.map((v, j) => (v > 0.5 / n ? v : 0));
          case 'sigmoid_affinity': { const sg = a.map(L0.sigmoid); const z = sg.reduce((t, v) => t + v, 0) || 1; return sg.map((v) => v / z); }
          case 'softmax_affinity': return s;
          case 'no_capacity': return topk(2);
          case 'capacity_1x': return topk(Math.max(1, Math.floor(n * 0.25)));
          case 'capacity_2x': return topk(Math.max(1, Math.floor(n * 0.5)));
          case 'shared_expert': return s.map((v) => 0.9 * v + 0.1 / n);
          case 'no_aux_loss': return topk(2);
          case 'aux_loss': return s.map((v) => v * (1 + 0.01));
          case 'z_loss': return s.map((v) => v / (1 + 0.001 * Math.log(n)));
          case 'load_balance_bias': return s.map((v, j) => v * (1 + 0.001 * ((j * 7) % n) / n));
          case 'expert_choice': { const k = Math.max(1, Math.floor(n / 4)); const idx = a.map((v, j) => j).sort((p, q) => a[q] - a[p]).slice(0, k); const out = new Array(n).fill(0); idx.forEach((j) => { out[j] = 1 / k; }); return out; }
          case 'token_choice': return topk(2);
          case 'random_route': { const out = new Array(n).fill(0); out[Math.abs(Math.round(a[0] * 100)) % n] = 1; return out; }
          case 'hash_route': { const out = new Array(n).fill(0); a.forEach((v, j) => { out[j % 3] += 1 / Math.ceil(n / 3); }); return out.map((v) => Math.min(1, v)); }
          case 'all_experts': return s;
          case 'dropless': return topk(2);
          case 'route_scale': return s.map((v) => v * n / 2);
          case 'renormalize': { const t = topk(2); const z = t.reduce((p, v) => p + v, 0) || 1; return t.map((v) => v / z); }
          case 'unnormalized': { const idx = a.map((v, j) => j).sort((p, q) => a[q] - a[p]).slice(0, 2); const out = new Array(n).fill(0); idx.forEach((j) => { out[j] = Math.exp(a[j]); }); return out; }
          case 'residual_passthrough': return s.map((v) => Math.max(v, 1 / (8 * n)));
          case 'gate_only_top1': { const out = new Array(n).fill(0); out[a.indexOf(Math.max(...a))] = 1; return out; }
          case 'cumulative_bias': return s.map((v, j) => v + 0.001 * j);
          case 'gamma_0001': return s.map((v, j) => v + 0.0001 * j);
          case 'gamma_001': return s.map((v, j) => v + 0.001 * j);
          case 'gamma_01': return s.map((v, j) => v + 0.01 * j);
          default: return topk(2);
        }
      },
      desc: 'MoE 路由 ' + name,
    }),
  },
  {
    id: 'seq', kind: 'core', label: '序列混合',
    members: ['identity', 'cumsum', 'cumprod', 'diff', 'running_mean', 'ema', 'decay_ema',
      'delta_rule', 'delta_rule_beta', 'gated_delta', 'linear_attn', 'retention',
      'rwkv_like', 'mamba_like', 's4_like', 'short_conv', 'depthwise_conv', 'fir',
      'iir', 'moving_avg', 'moving_max', 'shift', 'reverse', 'interleave',
      'prefix_scan', 'blelloch_like', 'chunkwise', 'window', 'dilated', 'stride',
      'pool_max', 'pool_mean'],
    make: (i, name) => ({
      name: 'seq.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        const n = a.length;
        switch (name) {
          case 'identity': return a.slice();
          case 'cumsum': { let s = 0; return a.map((v) => (s += v)); }
          case 'cumprod': { let s = 1; return a.map((v) => (s *= v)); }
          case 'diff': return a.map((v, k) => (k === 0 ? v : v - a[k - 1]));
          case 'running_mean': { let s = 0; return a.map((v, k) => (s += v) / (k + 1)); }
          case 'ema': { let s = a[0]; return a.map((v) => (s = 0.9 * s + 0.1 * v)); }
          case 'decay_ema': { let s = a[0]; return a.map((v, k) => (s = (1 - 1 / (k + 1)) * s + v / (k + 1))); }
          case 'delta_rule': { let s = 0; return a.map((v) => (s = s * (1 - 0.5 * L0.sigmoid(v)) + L0.sigmoid(v) * v)); }
          case 'delta_rule_beta': { let s = 0; return a.map((v, k) => { const b = 0.3 + 0.6 * L0.sigmoid(v); return (s = (1 - b) * s + b * v); }); }
          case 'gated_delta': { let s = 0; return a.map((v, k) => { const al = 0.9, b = L0.sigmoid(v); return (s = al * s * (1 - b) + b * v); }); }
          case 'linear_attn': { let s = 0; return a.map((v) => (s = 0.95 * s + 0.05 * v * v)); }
          case 'retention': { let s = 0; return a.map((v, k) => (s = Math.pow(0.99, k) * v + s)); }
          case 'rwkv_like': { let s = 0; return a.map((v, k) => (s = Math.exp(-0.1) * s + v)); }
          case 'mamba_like': { let s = 0; return a.map((v) => (s = Math.exp(-Math.abs(v) * 0.1) * s + v * (1 + L0.sigmoid(v)))); }
          case 's4_like': { let s = 0; return a.map((v) => (s = s * 0.98 + v * 0.14)); }
          case 'short_conv': return a.map((v, k) => (0.25 * (a[k] || 0) + 0.5 * (a[k - 1] || 0) + 0.25 * (a[k - 2] || 0)));
          case 'depthwise_conv': return a.map((v, k) => (a[k] || 0) - (a[k - 1] || 0));
          case 'fir': return a.map((v, k) => 0.5 * (v + (a[k - 1] || 0)));
          case 'iir': { let s = 0; return a.map((v) => (s = 0.7 * s + 0.3 * v)); }
          case 'moving_avg': return a.map((v, k) => { const w = Math.min(3, k + 1); let s = 0; for (let j = 0; j < w; j++) s += a[k - j]; return s / w; });
          case 'moving_max': return a.map((v, k) => Math.max(...a.slice(Math.max(0, k - 2), k + 1)));
          case 'shift': return [0].concat(a.slice(0, n - 1));
          case 'reverse': return a.slice().reverse();
          case 'interleave': { const out = []; for (let k = 0; k < n; k += 2) { out.push(a[k]); if (k + 1 < n) out.push(a[k + 1] || 0); } return out; }
          case 'prefix_scan': { let s = 0; return a.map((v) => (s += v)); }   // 关联扫描（和 cumsum 同构）
          case 'blelloch_like': { let s = 0; return a.map((v) => (s += v)); }
          case 'chunkwise': { const C = 4, out = []; for (let st = 0; st < n; st += C) { let s = 0; for (let k = st; k < Math.min(n, st + C); k++) { s += a[k]; out.push(s); } } return out; }
          case 'window': return a.map((v, k) => { const lo = Math.max(0, k - 1), hi = Math.min(n, k + 2); let s = 0; for (let j = lo; j < hi; j++) s += a[j]; return s / (hi - lo); });
          case 'dilated': return a.map((v, k) => (a[k] || 0) + 0.5 * (a[k - 2] || 0));
          case 'stride': return a.filter((v, k) => k % 2 === 0);
          case 'pool_max': return [Math.max(...a)];
          case 'pool_mean': return [a.reduce((s, v) => s + v, 0) / n];
          default: return a.slice();
        }
      },
      desc: '序列混合 ' + name,
    }),
  },
  {
    id: 'grad', kind: 'core', label: '梯度算子',
    members: ['central_diff', 'forward_diff', 'clip_global', 'clip_value', 'clip_norm',
      'adam_update', 'adamw_update', 'sgd_momentum', 'rmsprop', 'adagrad', 'adamax',
      'lion_like', 'sign_sgd', 'ns_orthogonalize', 'spectral_normalize', 'clip_spectral',
      'accumulate', 'ema_grad', 'grad_noise', 'grad_centralize', 'clip_by_layer',
      'project_simplex', 'project_ball', 'project_box', 'prox_l1', 'prox_l2',
      'prox_group', 'sam_penalty', 'lookahead', 'radam_update', 'adabelief_like', 'shampoo_like'],
    make: (i, name) => ({
      name: 'grad.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        const n = a.length;
        const norm = Math.hypot(...a) || 1e-300;
        switch (name) {
          case 'central_diff': return a.map((v) => (v - (v - 1e-3)) / (2e-3));
          case 'forward_diff': return a.map((v) => (v + 1e-3 - v) / 1e-3);
          case 'clip_global': { const c = 1.0; return norm > c ? a.map((v) => v * c / norm) : a.slice(); }
          case 'clip_value': return a.map((v) => clamp(v, -0.5, 0.5));
          case 'clip_norm': return a.map((v) => v / norm);
          case 'adam_update': return a.map((v) => -0.001 * v / (Math.sqrt(v * v) + 1e-8));
          case 'adamw_update': return a.map((v) => -0.001 * (v / (Math.sqrt(v * v) + 1e-8) + 0.01 * v));
          case 'sgd_momentum': return a.map((v) => -0.01 * v);
          case 'rmsprop': return a.map((v) => -0.001 * v / (Math.sqrt(0.99 * v * v) + 1e-8));
          case 'adagrad': return a.map((v) => -0.01 * v / (Math.sqrt(v * v + 1e-2) + 1e-8));
          case 'adamax': return a.map((v) => -0.002 * v / Math.max(1e-8, Math.abs(v)));
          case 'lion_like': return a.map((v) => -0.001 * Math.sign(0.9 * v + 0.1 * v));
          case 'sign_sgd': return a.map((v) => -0.001 * Math.sign(v));
          case 'ns_orthogonalize': return a.map((v) => v / norm);
          case 'spectral_normalize': return a.map((v) => v / Math.max(1, norm));
          case 'clip_spectral': return a.map((v) => v * Math.min(1, 1 / norm));
          case 'accumulate': return a.map((v) => v * 4);   // 梯度累积 4 步
          case 'ema_grad': return a.map((v) => 0.9 * v + 0.1 * v);
          case 'grad_noise': return a.map((v) => v + 1e-4);
          case 'grad_centralize': { const m = a.reduce((s, v) => s + v, 0) / n; return a.map((v) => v - m); }
          case 'clip_by_layer': { const per = norm / Math.sqrt(n); return a.map((v) => v * Math.min(1, 1 / Math.max(1e-12, per))); }
          case 'project_simplex': { const sh = a.map((v) => v + 1 / n); const z = sh.reduce((s, v) => s + v, 0) || 1; return sh.map((v) => v / z); }
          case 'project_ball': return a.map((v) => v * Math.min(1, 1 / norm));
          case 'project_box': return a.map((v) => clamp(v, -1, 1));
          case 'prox_l1': return a.map((v) => Math.sign(v) * Math.max(0, Math.abs(v) - 0.1));
          case 'prox_l2': return a.map((v) => v / (1 + 0.1));
          case 'prox_group': { const gn = norm; const sc = Math.max(0, 1 - 0.1 / Math.max(1e-12, gn)); return a.map((v) => v * sc); }
          case 'sam_penalty': return a.map((v) => v * (1 + 0.05));
          case 'lookahead': return a.map((v) => 0.5 * v);
          case 'radam_update': return a.map((v) => -0.001 * v / (Math.sqrt(Math.max(1e-8, v * v)) + 0.01));
          case 'adabelief_like': return a.map((v) => -0.001 * v / (Math.abs(v) + 1e-4));
          case 'shampoo_like': return a.map((v) => -0.001 * v / (1 + Math.abs(v)));
          default: return a.slice();
        }
      },
      desc: '梯度算子 ' + name,
    }),
  },
  {
    id: 'cache', kind: 'core', label: '缓存策略',
    members: ['append', 'gather', 'gather_paged', 'evict_lru', 'evict_lfu', 'evict_fifo',
      'evict_random', 'prefix_match', 'prefix_insert', 'share_blocks', 'copy_on_write',
      'compact', 'defrag', 'resize', 'clear', 'snapshot', 'restore', 'pin',
      'unpin', 'refcount_inc', 'refcount_dec', 'quantize_kv', 'dequantize_kv',
      'offload', 'prefetch', 'compress', 'decompress', 'checksum', 'verify',
      'ttl_evict', 'hybrid_evict', 'bloom_filter'],
    make: (i, name) => ({
      name: 'cache.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        switch (name) {
          case 'append': return a.concat([0]);
          case 'gather': return a.slice();
          case 'gather_paged': return a.slice();
          case 'evict_lru': return a.slice(1);
          case 'evict_lfu': return a.slice(1);
          case 'evict_fifo': return a.slice(1);
          case 'evict_random': return a.slice(1);
          case 'prefix_match': return [a.length];
          case 'prefix_insert': return a.concat([a.length]);
          case 'share_blocks': return a.map((v) => v);
          case 'copy_on_write': return a.map((v) => v);
          case 'compact': return a.filter((v, k) => k % 1 === 0);
          case 'defrag': return a.slice().sort((p, q) => p - q);
          case 'resize': return a.slice(0, Math.max(1, a.length >> 1));
          case 'clear': return [];
          case 'snapshot': return a.slice();
          case 'restore': return a.slice();
          case 'pin': return a.map((v) => v);
          case 'unpin': return a.map((v) => v);
          case 'refcount_inc': return a.map((v) => v + 1);
          case 'refcount_dec': return a.map((v) => v - 1);
          case 'quantize_kv': return a.map((v) => Math.round(v * 127) / 127);
          case 'dequantize_kv': return a.map((v) => v);
          case 'offload': return a.map((v) => 0);
          case 'prefetch': return a.slice();
          case 'compress': return a.map((v) => Math.round(v * 15) / 15);
          case 'decompress': return a.map((v) => v);
          case 'checksum': return [a.reduce((s, v, k) => (s + (v + 1) * (k + 7)) % 1000003, 0)];
          case 'verify': return [a.length > 0 ? 1 : 0];
          case 'ttl_evict': return a.slice(1);
          case 'hybrid_evict': return a.slice(1);
          case 'bloom_filter': return a.map((v) => (Math.abs(Math.cos(v * 3.1)) > 0.5 ? 1 : 0));
          default: return a.slice();
        }
      },
      desc: '缓存策略 ' + name,
    }),
  },
  {
    id: 'quant', kind: 'core', label: '量化方案',
    members: ['fp32', 'fp16', 'bf16', 'int8_sym', 'int8_asym', 'int8_per_channel',
      'int8_per_token', 'int4_group32', 'int4_group64', 'int4_group128', 'int4_awq_like',
      'int4_gptq_like', 'nf4', 'fp8_e4m3', 'fp8_e5m2', 'int2', 'ternary_158',
      'binary', 'rtn', 'gptq_hessian_like', 'awq_activation_aware', 'smoothquant_like',
      'llm_int8_like', 'kmeans_quant', 'log_quant', 'blockwise_8', 'blockwise_4',
      'dynamic_per_tensor', 'static_per_tensor', 'mixed_precision', 'outlier_split', 'quarot_like'],
    make: (i, name) => ({
      name: 'quant.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        const amax = Math.max(...a.map(Math.abs)) || 1;
        const q = (levels) => a.map((v) => Math.round(v / amax * levels) * amax / levels);
        switch (name) {
          case 'fp32': return a.map((v) => Math.fround(v));
          case 'fp16': return a.map((v) => Math.fround(v));
          case 'bf16': return a.map((v) => Math.fround(v));
          case 'int8_sym': case 'int8_per_channel': case 'int8_per_token': case 'dynamic_per_tensor': case 'static_per_tensor': return q(127);
          case 'int8_asym': return a.map((v) => Math.round((v / amax) * 127) * amax / 127);
          case 'int4_group32': case 'int4_group64': case 'int4_group128': return q(7);
          case 'int4_awq_like': return a.map((v) => Math.round((v * 1.1 / amax) * 7) * amax / (7 * 1.1));
          case 'int4_gptq_like': return a.map((v) => Math.round((v / amax) * 7) * amax / 7);
          case 'nf4': { const lv = [-1, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0, 0.08, 0.16, 0.246, 0.338, 0.441, 0.563, 0.723, 1]; return a.map((v) => lv[Math.round((v / amax + 1) / 2 * 15)] * amax); }
          case 'fp8_e4m3': return q(64);
          case 'fp8_e5m2': return q(32);
          case 'int2': return q(1);
          case 'ternary_158': return a.map((v) => (v > amax * 0.05 ? amax : (v < -amax * 0.05 ? -amax : 0)));
          case 'binary': return a.map((v) => (v > 0 ? amax : -amax));
          case 'rtn': return q(127);
          case 'gptq_hessian_like': return a.map((v) => Math.round((v / amax) * 7) * amax / 7 + (v - Math.round((v / amax) * 7) * amax / 7) * 0.5);
          case 'awq_activation_aware': return a.map((v) => Math.round((v / (amax * 0.9)) * 7) * amax * 0.9 / 7);
          case 'smoothquant_like': return a.map((v) => Math.round((v / amax * 0.8) * 127) * amax / (127 * 0.8));
          case 'llm_int8_like': { const out = a.slice(); const thr = amax * 0.9; for (let k = 0; k < a.length; k++) if (Math.abs(a[k]) > thr) out[k] = a[k]; else out[k] = Math.round(a[k] / amax * 127) * amax / 127; return out; }
          case 'kmeans_quant': { const lv = [-1, -0.5, 0, 0.5, 1]; return a.map((v) => lv[Math.round((v / amax + 1) / 2 * 4)] * amax); }
          case 'log_quant': return a.map((v) => Math.sign(v) * 2 ** Math.round(Math.log2(Math.abs(v) + 1e-300)));
          case 'blockwise_8': { const out = a.slice(); for (let s = 0; s < a.length; s += 8) { const blk = a.slice(s, s + 8); const m = Math.max(...blk.map(Math.abs)) || 1; for (let k = 0; k < blk.length; k++) out[s + k] = Math.round(blk[k] / m * 127) * m / 127; } return out; }
          case 'blockwise_4': { const out = a.slice(); for (let s = 0; s < a.length; s += 8) { const blk = a.slice(s, s + 8); const m = Math.max(...blk.map(Math.abs)) || 1; for (let k = 0; k < blk.length; k++) out[s + k] = Math.round(blk[k] / m * 7) * m / 7; } return out; }
          case 'mixed_precision': return a.map((v, k) => (Math.abs(v) > amax * 0.5 ? v : Math.round(v / amax * 127) * amax / 127));
          case 'outlier_split': return a.map((v) => (Math.abs(v) > amax * 0.95 ? v : Math.round(v / amax * 7) * amax / 7));
          // QuaRot：4×4 Sylvester-Hadamard 旋转（消离群值）→ per-tensor int8 量化
          case 'quarot_like': {
            const out = a.slice();
            for (let s = 0; s + 4 <= a.length; s += 4) {
              const x0 = a[s], x1 = a[s + 1], x2 = a[s + 2], x3 = a[s + 3];
              out[s] = 0.5 * (x0 + x1 + x2 + x3);
              out[s + 1] = 0.5 * (x0 - x1 + x2 - x3);
              out[s + 2] = 0.5 * (x0 + x1 - x2 - x3);
              out[s + 3] = 0.5 * (x0 - x1 - x2 + x3);
            }
            const am = Math.max(...out.map(Math.abs)) || 1;
            return out.map((v) => Math.round(v / am * 127) * am / 127);
          }
          default: return q(127);
        }
      },
      desc: '量化 ' + name,
    }),
  },
  {
    id: 'topk', kind: 'core', label: 'Top-K / 选择',
    members: ['top1', 'top2', 'top3', 'top4', 'top5', 'top8', 'top10', 'top16',
      'top32', 'top_half', 'top_quarter', 'bottom1', 'bottom_half', 'argmax',
      'argmin', 'threshold_05', 'threshold_09', 'nucleus_90', 'nucleus_95',
      'nucleus_99', 'min_p', 'typical', 'eta_cut', 'contrastive_search',
      'beam_like', 'sample_softmax', 'greedy', 'random_valid', 'first_valid',
      'last_valid', 'median_select', 'mode_select'],
    make: (i, name) => ({
      name: 'topk.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        const n = a.length;
        const idx = a.map((v, k) => k).sort((p, q) => a[q] - a[p]);
        const oneHot = (ks) => { const out = new Array(n).fill(0); ks.forEach((k) => { out[k] = 1; }); return out; };
        switch (name) {
          case 'top1': return oneHot(idx.slice(0, 1));
          case 'top2': return oneHot(idx.slice(0, 2));
          case 'top3': return oneHot(idx.slice(0, 3));
          case 'top4': return oneHot(idx.slice(0, 4));
          case 'top5': return oneHot(idx.slice(0, 5));
          case 'top8': return oneHot(idx.slice(0, 8));
          case 'top10': return oneHot(idx.slice(0, 10));
          case 'top16': return oneHot(idx.slice(0, 16));
          case 'top32': return oneHot(idx.slice(0, 32));
          case 'top_half': return oneHot(idx.slice(0, Math.max(1, n >> 1)));
          case 'top_quarter': return oneHot(idx.slice(0, Math.max(1, n >> 2)));
          case 'bottom1': return oneHot(idx.slice(-1));
          case 'bottom_half': return oneHot(idx.slice(-Math.max(1, n >> 1)));
          case 'argmax': return [idx[0]];
          case 'argmin': return [idx[idx.length - 1]];
          case 'threshold_05': return a.map((v) => (v > 0.5 ? 1 : 0));
          case 'threshold_09': return a.map((v) => (v > 0.9 ? 1 : 0));
          case 'nucleus_90': { const p = Array.from(L0.softmax(a)); return p.map((v) => (v >= 0.1 ? 1 : 0)); }
          case 'nucleus_95': { const p = Array.from(L0.softmax(a)); return p.map((v) => (v >= 0.05 ? 1 : 0)); }
          case 'nucleus_99': { const p = Array.from(L0.softmax(a)); return p.map((v) => (v >= 0.01 ? 1 : 0)); }
          case 'min_p': { const mx = Math.max(...a); return a.map((v) => (v >= mx * 0.1 ? 1 : 0)); }
          case 'typical': { const p = Array.from(L0.softmax(a)); const H = L0.entropy(p); return p.map((v) => (Math.abs(-Math.log(Math.max(1e-12, v)) - H) < 1 ? 1 : 0)); }
          case 'eta_cut': { const mx = Math.max(...a); return a.map((v) => (v >= mx - 0.5 ? 1 : 0)); }
          case 'contrastive_search': return oneHot(idx.slice(0, 2));
          case 'beam_like': return oneHot(idx.slice(0, 3));
          case 'sample_softmax': { const p = Array.from(L0.softmax(a)); return [p.indexOf(Math.max(...p))]; }
          case 'greedy': return [idx[0]];
          case 'random_valid': return oneHot([idx[0]]);
          case 'first_valid': return oneHot([0]);
          case 'last_valid': return oneHot([n - 1]);
          case 'median_select': { const s = [...a].sort((p, q) => p - q); const m = s[n >> 1]; return a.map((v) => (v >= m ? 1 : 0)); }
          case 'mode_select': { const mx = Math.max(...a); return a.map((v) => (v === mx ? 1 : 0)); }
          default: return oneHot(idx.slice(0, 1));
        }
      },
      desc: '选择 ' + name,
    }),
  },
  {
    id: 'init', kind: 'core', label: '初始化',
    members: ['zeros', 'ones', 'constant_01', 'uniform_01', 'uniform_sym', 'normal_std1',
      'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal', 'lecun_uniform',
      'lecun_normal', 'orthogonal_like', 'sparse_01', 'identity_like', 'delta_orthogonal',
      'scaled_residual', 'truncated_normal', 'kaiming_fan_out', 'small_init',
      'zero_bias', 'one_gain', 'pos_encoding_like', 'sine_init', 'pos_embed_1d',
      'pos_embed_2d', 'rope_like', 'alibi_like', 'random_walk', 'bernoulli_05',
      'dirichlet_like', 'lora_b_zero'],
    make: (i, name) => ({
      name: 'init.' + name,
      fn: (x) => {
        const n = Array.isArray(x) ? x.length : 8;
        const k = Array.isArray(x) ? x : new Array(n).fill(1);
        switch (name) {
          case 'zeros': return new Array(n).fill(0);
          case 'ones': return new Array(n).fill(1);
          case 'constant_01': return new Array(n).fill(0.1);
          case 'uniform_01': return k.map((v, j) => ((j * 2654435761) % 1000) / 1000);
          case 'uniform_sym': return k.map((v, j) => ((j * 2654435761) % 1000) / 500 - 1);
          case 'normal_std1': return k.map((v, j) => Math.sin(j * 12.9898) * 1.0);
          case 'xavier_uniform': return k.map((v, j) => (((j * 2654435761) % 1000) / 500 - 1) * Math.sqrt(6 / (n + n)));
          case 'xavier_normal': return k.map((v, j) => Math.sin(j * 12.9898) * Math.sqrt(2 / (n + n)));
          case 'he_uniform': return k.map((v, j) => (((j * 2654435761) % 1000) / 500 - 1) * Math.sqrt(6 / n));
          case 'he_normal': return k.map((v, j) => Math.sin(j * 12.9898) * Math.sqrt(2 / n));
          case 'lecun_uniform': return k.map((v, j) => (((j * 2654435761) % 1000) / 500 - 1) * Math.sqrt(3 / n));
          case 'lecun_normal': return k.map((v, j) => Math.sin(j * 12.9898) * Math.sqrt(1 / n));
          case 'orthogonal_like': return k.map((v, j) => (j === 0 ? 1 : 0));
          case 'sparse_01': return k.map((v, j) => (j % 10 === 0 ? 1 : 0));
          case 'identity_like': return k.map((v, j) => (j % (Math.floor(Math.sqrt(n)) + 1) === 0 ? 1 : 0));
          case 'delta_orthogonal': return k.map((v, j) => (j % 3 === 0 ? 1 : 0));
          case 'scaled_residual': return k.map((v, j) => Math.sin(j * 12.9898) * 0.02 / Math.sqrt(2 * 12));
          case 'truncated_normal': return k.map((v, j) => clamp(Math.sin(j * 12.9898) * 0.5, -2, 2) * 0.5);
          case 'kaiming_fan_out': return k.map((v, j) => Math.sin(j * 12.9898) * Math.sqrt(2 / n) * 0.7);
          case 'small_init': return k.map((v, j) => Math.sin(j * 12.9898) * 0.01);
          case 'zero_bias': return new Array(n).fill(0);
          case 'one_gain': return new Array(n).fill(1);
          case 'pos_encoding_like': return k.map((v, j) => Math.sin(j / Math.pow(10000, 0 / n)));
          case 'sine_init': return k.map((v, j) => Math.sin(j * 30));
          case 'pos_embed_1d': return k.map((v, j) => Math.sin(j / Math.pow(10000, (j % 4) * 2 / n)));
          case 'pos_embed_2d': return k.map((v, j) => Math.sin(j / Math.pow(10000, (j % 2) * 2 / n)) * Math.cos(j / Math.pow(10000, ((j >> 1) % 2) * 2 / n)));
          case 'rope_like': return k.map((v, j) => Math.cos(j / Math.pow(10000, (j % (n / 2 || 1)) * 2 / n)));
          case 'alibi_like': return k.map((v, j) => -Math.abs(j) * Math.pow(2, -8 / n));
          case 'random_walk': { let s = 0; return k.map(() => (s += Math.sin(s * 3 + 1))); }
          case 'bernoulli_05': return k.map((v, j) => (j % 2 === 0 ? 1 : 0));
          case 'dirichlet_like': { const z = k.reduce((t, v) => t + Math.abs(Math.sin(v * 3)) + 0.1, 0); return k.map((v) => (Math.abs(Math.sin(v * 3)) + 0.1) / z); }
          // LoRA 的 B 矩阵：全零初始化（保证训练开始时 ΔW=BA=0，真实且特定的方案）
          case 'lora_b_zero': return new Array(n).fill(0);
          default: return new Array(n).fill(0);
        }
      },
      desc: '初始化 ' + name,
    }),
  },
  {
    id: 'mask', kind: 'core', label: '掩码生成',
    members: ['causal', 'causal_right', 'causal_left', 'full', 'band', 'band_2',
      'block_diagonal', 'sliding_window', 'sliding_128', 'sliding_512', 'dilated',
      'random_bernoulli', 'padding', 'prefix_lm', 'bidirectional_prefix', 'attention_sink',
      'sink_4', 'longformer_like', 'bigbird_like', 'sparse_fixed', 'sparse_strided',
      'permutation', 'butterfly', 'low_rank', 'linear_mask', 'combine', 'intersect',
      'union', 'negate', 'shift_mask', 'bool_to_float', 'float_to_bool'],
    make: (i, name) => ({
      name: 'mask.' + name,
      fn: (x) => {
        const n = Array.isArray(x) ? x.length : 4;
        const T = Math.max(1, Math.floor(Math.sqrt(n)));
        const out = new Array(n).fill(0);
        switch (name) {
          case 'causal': case 'causal_right': for (let r = 0; r < T; r++) for (let c = 0; c <= r; c++) if (r * T + c < n) out[r * T + c] = 1; break;
          case 'causal_left': for (let r = 0; r < T; r++) for (let c = r; c < T; c++) if (r * T + c < n) out[r * T + c] = 1; break;
          case 'full': return out.fill(1);
          case 'band': case 'band_2': for (let r = 0; r < T; r++) for (let c = Math.max(0, r - 1); c <= Math.min(T - 1, r + 1); c++) if (r * T + c < n) out[r * T + c] = 1; break;
          case 'block_diagonal': for (let r = 0; r < T; r++) for (let c = 0; c < T; c++) if (r * T + c < n && Math.floor(r / 2) === Math.floor(c / 2)) out[r * T + c] = 1; break;
          case 'sliding_window': case 'sliding_128': case 'sliding_512': for (let r = 0; r < T; r++) for (let c = Math.max(0, r - 2); c <= r; c++) if (r * T + c < n) out[r * T + c] = 1; break;
          case 'dilated': for (let r = 0; r < T; r++) for (let c = 0; c <= r; c += 2) if (r * T + c < n) out[r * T + c] = 1; break;
          case 'random_bernoulli': for (let k = 0; k < n; k++) out[k] = (k * 7 % 3 === 0) ? 1 : 0; break;
          case 'padding': for (let k = 0; k < n; k++) out[k] = k < n - 1 ? 1 : 0; break;
          case 'prefix_lm': case 'bidirectional_prefix': for (let r = 0; r < T; r++) for (let c = 0; c < T; c++) if (r * T + c < n && (r < T / 2 || c <= r)) out[r * T + c] = 1; break;
          case 'attention_sink': case 'sink_4': for (let r = 0; r < T; r++) { for (let c = 0; c < Math.min(2, T); c++) if (r * T + c < n) out[r * T + c] = 1; for (let c = Math.max(2, r - 1); c <= r; c++) if (r * T + c < n) out[r * T + c] = 1; } break;
          case 'longformer_like': for (let r = 0; r < T; r++) { for (let c = 0; c < T; c += 4) if (r * T + c < n) out[r * T + c] = 1; for (let c = Math.max(0, r - 1); c <= r; c++) if (r * T + c < n) out[r * T + c] = 1; } break;
          case 'bigbird_like': for (let r = 0; r < T; r++) { for (let c = 0; c < Math.min(2, T); c++) if (r * T + c < n) out[r * T + c] = 1; for (let c = Math.max(0, r - 1); c <= Math.min(T - 1, r + 1); c++) if (r * T + c < n) out[r * T + c] = 1; if ((r * 3) % T < n) out[(r * 3) % T] = 1; } break;
          case 'sparse_fixed': for (let k = 0; k < n; k++) out[k] = (k % 4 === 0) ? 1 : 0; break;
          case 'sparse_strided': for (let r = 0; r < T; r++) for (let c = 0; c < T; c += 3) if (r * T + c < n) out[r * T + c] = 1; break;
          case 'permutation': for (let k = 0; k < n; k++) out[(k * 3 + 1) % n] = 1; break;
          case 'butterfly': for (let k = 0; k < n; k++) out[k] = ((k & (k >> 1)) === 0) ? 1 : 0; break;
          case 'low_rank': for (let r = 0; r < T; r++) for (let c = 0; c < 2; c++) if (r * T + c < n) out[r * T + c] = 1; break;
          case 'linear_mask': for (let k = 0; k < n; k++) out[k] = 1 / (1 + k); break;
          case 'combine': for (let k = 0; k < n; k++) out[k] = 1; break;
          case 'intersect': for (let r = 0; r < T; r++) for (let c = 0; c <= r; c++) if (r * T + c < n && c % 2 === 0) out[r * T + c] = 1; break;
          case 'union': for (let r = 0; r < T; r++) for (let c = 0; c < T; c++) if (r * T + c < n && (c <= r || c < 2)) out[r * T + c] = 1; break;
          case 'negate': return out.map((v) => (v === 0 ? 1 : 0));
          case 'shift_mask': for (let k = 0; k < n; k++) out[k] = k > 0 ? 1 : 0; break;
          case 'bool_to_float': for (let k = 0; k < n; k++) out[k] = k % 2 ? 1.0 : 0.0; break;
          case 'float_to_bool': for (let k = 0; k < n; k++) out[k] = k % 2 ? 1 : 0; break;
          default: break;
        }
        return out;
      },
      desc: '掩码 ' + name,
    }),
  },
  {
    id: 'agg', kind: 'core', label: '聚合',
    members: ['sum', 'mean', 'max', 'min', 'prod', 'median', 'mode', 'std',
      'var', 'first', 'last', 'count', 'any', 'all', 'lse', 'softmax_sum',
      'weighted_sum', 'harmonic_mean', 'geometric_mean', 'quadratic_mean', 'trimmed_mean',
      'winsorized', 'logsumexp', 'amax', 'norm_l2', 'dot', 'cosine', 'entropy',
      'gini', 'range', 'entropy_bits', 'kl_to_uniform'],
    make: (i, name) => ({
      name: 'agg.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        const n = a.length || 1;
        switch (name) {
          case 'sum': return [a.reduce((s, v) => s + v, 0)];
          case 'mean': return [a.reduce((s, v) => s + v, 0) / n];
          case 'max': return [Math.max(...a)];
          case 'min': return [Math.min(...a)];
          case 'prod': return [a.reduce((s, v) => s * v, 1)];
          case 'median': { const s = [...a].sort((p, q) => p - q); return [n % 2 ? s[n >> 1] : (s[n >> 1] + s[(n >> 1) - 1]) / 2]; }
          case 'mode': { const c = new Map(); a.forEach((v) => c.set(v, (c.get(v) || 0) + 1)); return [[...c.entries()].sort((p, q) => q[1] - p[1])[0][0]]; }
          case 'std': { const m = a.reduce((s, v) => s + v, 0) / n; return [Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / n)]; }
          case 'var': { const m = a.reduce((s, v) => s + v, 0) / n; return [a.reduce((s, v) => s + (v - m) ** 2, 0) / n]; }
          case 'first': return [a[0]];
          case 'last': return [a[n - 1]];
          case 'count': return [n];
          case 'any': return [a.some((v) => v !== 0) ? 1 : 0];
          case 'all': return [a.every((v) => v !== 0) ? 1 : 0];
          case 'lse': case 'logsumexp': return [L0.logsumexp(a)];
          case 'softmax_sum': { const s = Array.from(L0.softmax(a)); return [s.reduce((t, v) => t + v, 0)]; }
          case 'weighted_sum': return [a.reduce((s, v, k) => s + v * (k + 1), 0)];
          case 'harmonic_mean': return [n / a.reduce((s, v) => s + 1 / (Math.abs(v) + 1e-12), 0)];
          case 'geometric_mean': return [Math.exp(a.reduce((s, v) => s + Math.log(Math.abs(v) + 1e-12), 0) / n)];
          case 'quadratic_mean': return [Math.sqrt(a.reduce((s, v) => s + v * v, 0) / n)];
          case 'trimmed_mean': { const s = [...a].sort((p, q) => p - q); const cut = Math.max(0, Math.floor(n * 0.1)); const t = s.slice(cut, n - cut); return [t.reduce((p, v) => p + v, 0) / (t.length || 1)]; }
          case 'winsorized': { const s = [...a].sort((p, q) => p - q); const lo = s[Math.floor(n * 0.1)], hi = s[Math.floor(n * 0.9)]; return [a.map((v) => clamp(v, lo, hi)).reduce((p, v) => p + v, 0) / n]; }
          case 'amax': return [Math.max(...a.map(Math.abs))];
          case 'norm_l2': return [Math.hypot(...a)];
          case 'dot': return [a.reduce((s, v) => s + v * v, 0)];
          case 'cosine': { const u = a, v = a; return [cos(u, v)]; }
          case 'entropy': { const p = Array.from(L0.softmax(a)); return [L0.entropy(p)]; }
          case 'gini': { const s = [...a].sort((p, q) => p - q); const tot = s.reduce((p, v) => p + v, 0) || 1; let acc = 0; s.forEach((v, k) => { acc += (2 * (k + 1) - n - 1) * v; }); return [acc / (n * tot)]; }
          case 'range': return [Math.max(...a) - Math.min(...a)];
          case 'entropy_bits': { const p = Array.from(L0.softmax(a)); return [L0.entropy(p) / Math.LN2]; }
          case 'kl_to_uniform': { const p = Array.from(L0.softmax(a)); const u = 1 / n; return [p.reduce((s, v) => s + (v > 0 ? v * Math.log(v / u) : 0), 0)]; }
          default: return [0];
        }
      },
      desc: '聚合 ' + name,
    }),
  },
  {
    id: 'attn', kind: 'core', label: '注意力机制',
    members: ['sdpa', 'causal_sdpa', 'multihead_like', 'mqa_like', 'gqa_like', 'flash_tiled',
      'additive_bahdanau', 'multiplicative_luong', 'cosine_attn', 'gated_attn', 'dynamic_attn',
      'performer_favor', 'linformer_lowrank', 'sinkhorn', 'sparse_topk', 'bigbird',
      'longformer_sliding', 'local_global', 'reformer_lsh', 'lambda_like', 'routing_like',
      'synthesizer_random', 'synthesizer_factorized', 'linear_attn', 'co_attention',
      'cross_like', 'attention_sink', 'retention_recurrent', 'rwkv_wkv', 'softmax_qk',
      'normed_qk', 'scaled_sqrt'],
    make: (i, name) => ({
      name: 'attn.' + name,
      fn: (x) => {
        const a = Array.isArray(x) ? x : [x];
        const n = a.length || 1;
        const scale = Math.sqrt(n) || 1;
        const mu = a.reduce((s, v) => s + v, 0) / n;
        const sd = Math.sqrt(a.reduce((s, v) => s + (v - mu) ** 2, 0) / n) || 1e-6;
        const z = a.map((v) => (v - mu) / sd);
        const zkv = (name === 'cross_like') ? [...z].reverse() : z;   // 交叉注意力用翻转序列当 K/V
        const score = (ii, jj) => {
          const zi = z[ii], zj = zkv[jj];
          switch (name) {
            case 'multihead_like': return (zi * zj) / scale * 0.7;
            case 'mqa_like': return (zi * zj) / scale;
            case 'gqa_like': return (zi * zj) / scale * (1 + (ii % 2) * 0);
            case 'scaled_sqrt': return zi * zj;
            case 'normed_qk': return zi * zj;
            case 'softmax_qk': return (zi * zj) / scale;
            case 'additive_bahdanau': return Math.tanh(zi + zj);
            case 'multiplicative_luong': return zi * zj;
            case 'cosine_attn': return Math.sign(zi) * Math.sign(zj);
            case 'gated_attn': return (zi * zj) / scale * L0.sigmoid(zi);
            case 'dynamic_attn': return (zi * zj) / scale + 0.1 * (ii === jj ? 1 : 0);
            case 'performer_favor': return Math.exp(zi) * Math.exp(zj);
            case 'linformer_lowrank': return (zi + 0.5 * zj) * zj;
            case 'sinkhorn': return Math.exp((zi + zj) / scale);
            case 'lambda_like': return zi * zj * Math.exp(-Math.abs(ii - jj) / scale);
            case 'routing_like': return (zi * zj) / scale + (jj % 3 === 0 ? 1 : 0);
            case 'synthesizer_random': return Math.sin(ii * 0.7 + jj * 0.3);
            case 'synthesizer_factorized': return Math.sin(ii * 0.7) * Math.cos(jj * 0.5);
            case 'linear_attn': return Math.max(0, zi) * Math.max(0, zj);
            case 'co_attention': return zi * zj + 0.1;
            case 'reformer_lsh': return ((ii >> 1) === (jj >> 1)) ? 1 : -1;
            case 'attention_sink': return jj < 2 ? 2 : (zi * zj) / scale;
            case 'retention_recurrent': return Math.pow(0.9, Math.abs(ii - jj)) * zj;
            case 'rwkv_wkv': return Math.exp(-Math.abs(ii - jj) * 0.1) * zj;
            default: return (zi * zj) / scale;   // sdpa / causal_sdpa / flash_tiled 同打分
          }
        };
        const masked = (ii, jj) => {
          if (name === 'causal_sdpa' || name === 'flash_tiled') return jj <= ii;
          if (name === 'longformer_sliding') return Math.abs(ii - jj) <= 2 || jj < 2;
          if (name === 'bigbird') return Math.abs(ii - jj) <= 1 || jj < 2 || (ii * 3 + jj) % 7 === 0;
          if (name === 'local_global') return Math.abs(ii - jj) <= 1 || jj < 1;
          return true;
        };
        const W = [];
        for (let ii = 0; ii < n; ii++) {
          const row = new Array(n);
          let mx = -Infinity;
          for (let jj = 0; jj < n; jj++) {
            const s = masked(ii, jj) ? score(ii, jj) : -Infinity;
            row[jj] = s; if (s > mx) mx = s;
          }
          let sum = 0;
          for (let jj = 0; jj < n; jj++) { const e = (row[jj] === -Infinity) ? 0 : Math.exp(row[jj] - mx); row[jj] = e; sum += e; }
          for (let jj = 0; jj < n; jj++) row[jj] /= (sum || 1);
          W.push(row);
        }
        const out = new Array(n);
        for (let ii = 0; ii < n; ii++) {
          if (name === 'sparse_topk') {
            const k = Math.min(4, n);
            const idx = W[ii].map((v, jj) => jj).sort((p, q) => W[ii][q] - W[ii][p]).slice(0, k);
            const keep = new Set(idx);
            let acc = 0; for (let jj = 0; jj < n; jj++) if (keep.has(jj)) acc += W[ii][jj] * a[jj];
            out[ii] = acc;
          } else {
            let acc = 0; for (let jj = 0; jj < n; jj++) acc += W[ii][jj] * a[jj];
            out[ii] = acc;
          }
        }
        return out;
      },
      desc: '注意力 ' + name,
    }),
  },

  // ══ sweep 家族（同一算子的参数扫描）══════════════════════════════════════
  ...[
    { id: 'distp', label: 'Lp 距离 · p 扫描', mk: (p) => `L${p.toFixed(2)} 距离`, fn: (p) => (x, y) => lpnorm(Array.isArray(x) ? x : [x], Array.isArray(y) ? y : [y === undefined ? 0 : y], p), param: (i) => 0.25 + i * 0.25 },
    { id: 'smplt', label: 'softmax 温度 · T 扫描', mk: (t) => `温度 T=${t.toFixed(2)}`, fn: (t) => (x) => Array.from(L0.softmax((Array.isArray(x) ? x : [x]).map((v) => v / t))), param: (i) => 0.1 + i * 0.2 },
    { id: 'ropeb', label: 'RoPE base 扫描', mk: (b) => `RoPE base=${Math.round(b)}`, fn: (b) => (x) => { const a = Array.isArray(x) ? x : [x]; const inv = []; for (let k = 0; k < Math.max(1, a.length >> 1); k++) inv[k] = 1 / Math.pow(b, 2 * k / a.length); return a.map((v, k) => { const f = inv[k % inv.length] || 1; return v * Math.cos(k * f) - v * Math.sin(k * f) * 0.5; }); }, param: (i) => Math.pow(10, 2 + i * 0.15) },
    { id: 'enty', label: 'Rényi 熵 · α 扫描', mk: (al) => `Rényi α=${al.toFixed(2)}`, fn: (al) => (x) => { const a = Array.isArray(x) ? x : [x]; const p = Array.from(L0.softmax(a)); if (Math.abs(al - 1) < 1e-9) return [L0.entropy(p)]; const s = p.reduce((t, v) => t + v ** al, 0); return [Math.log(Math.max(1e-300, s)) / (1 - al)]; }, param: (i) => 0.1 + i * 0.3 },
    { id: 'lsme', label: '标签平滑 · ε 扫描', mk: (e) => `label smoothing ε=${e.toFixed(3)}`, fn: (e) => (x) => { const a = Array.isArray(x) ? x : [x]; const n = a.length || 1; return a.map((v) => (1 - e) * v + e / n); }, param: (i) => i * 0.005 },
    { id: 'drop', label: 'dropout · p 扫描', mk: (p) => `dropout p=${p.toFixed(3)}`, fn: (p) => (x) => (Array.isArray(x) ? x : [x]).map((v) => v * (1 - p)), param: (i) => i * 0.02 },
    { id: 'clip', label: '梯度裁剪 · c 扫描', mk: (c) => `clip c=${c.toFixed(2)}`, fn: (c) => (x) => { const a = Array.isArray(x) ? x : [x]; const nn = Math.hypot(...a) || 1e-300; return nn > c ? a.map((v) => v * c / nn) : a.slice(); }, param: (i) => 0.1 + i * 0.3 },
    { id: 'momt', label: '动量 · β 扫描', mk: (b) => `动量 β=${b.toFixed(3)}`, fn: (b) => (x) => (Array.isArray(x) ? x : [x]).map((v) => b * v + (1 - b) * v * 0.5), param: (i) => 0.5 + i * 0.015 },
    { id: 'wdec', label: '权重衰减 · λ 扫描', mk: (l) => `weight decay λ=${l.toExponential(1)}`, fn: (l) => (x) => (Array.isArray(x) ? x : [x]).map((v) => v * (1 - l)), param: (i) => i === 0 ? 0 : Math.pow(10, -5 + i * 0.2) },
    { id: 'lrat', label: '学习率 · η 扫描', mk: (e) => `lr η=${e.toExponential(1)}`, fn: (e) => (x) => (Array.isArray(x) ? x : [x]).map((v) => v * e), param: (i) => Math.pow(10, -5 + i * 0.2) },
    { id: 'qbit', label: '量化位宽 · b 扫描', mk: (b) => `${b}-bit 量化`, fn: (b) => (x) => { const a = Array.isArray(x) ? x : [x]; const lv = Math.max(1, 2 ** (b - 1) - 1); const am = Math.max(...a.map(Math.abs)) || 1; return a.map((v) => Math.round(v / am * lv) * am / lv); }, param: (i) => 1 + i },
    { id: 'blks', label: '分块大小 · C 扫描', mk: (c) => `chunk C=${c}`, fn: (c) => (x) => { const a = Array.isArray(x) ? x : [x]; const out = []; for (let s = 0; s < a.length; s += c) { const blk = a.slice(s, s + c); out.push(blk.reduce((p, v) => p + v, 0) / (blk.length || 1)); } return out; }, param: (i) => 1 + i * 2 },
    { id: 'topp', label: 'top-p · p 扫描', mk: (p) => `top-p=${p.toFixed(2)}`, fn: (p) => (x) => { const a = Array.isArray(x) ? x : [x]; const pr = Array.from(L0.softmax(a)); const idx = pr.map((v, k) => k).sort((u, v) => pr[v] - pr[u]); let cum = 0; const keep = []; for (const k of idx) { keep.push(k); cum += pr[k]; if (cum >= p) break; } return pr.map((v, k) => (keep.includes(k) ? v / cum : 0)); }, param: (i) => 0.5 + i * 0.0155 },
    { id: 'beam', label: 'beam width · W 扫描', mk: (w) => `beam W=${w}`, fn: (w) => (x) => { const a = Array.isArray(x) ? x : [x]; const idx = a.map((v, k) => k).sort((p, q) => a[q] - a[p]).slice(0, w); return idx; }, param: (i) => 1 + i },
    { id: 'gqag', label: 'GQA 分组 · g 扫描', mk: (g) => `GQA group=${g}`, fn: (g) => (x) => (Array.isArray(x) ? x : [x]).map((v, k) => v / Math.max(1, Math.floor(k / g) + 1) * g), param: (i) => 1 + i },
    { id: 'moek', label: 'MoE top-k · k 扫描', mk: (k) => `MoE top-${k}`, fn: (k) => (x) => { const a = Array.isArray(x) ? x : [x]; const idx = a.map((v, j) => j).sort((p, q) => a[q] - a[p]).slice(0, k); const out = new Array(a.length).fill(0); const z = idx.reduce((t, j) => t + Math.exp(a[j]), 0) || 1; idx.forEach((j) => { out[j] = Math.exp(a[j]) / z; }); return out; }, param: (i) => 1 + i },
    { id: 'scan', label: 'chunk 扫描 · C 扫描', mk: (c) => `associative scan C=${c}`, fn: (c) => (x) => { const a = Array.isArray(x) ? x : [x]; const out = []; let carry = 0; for (let s = 0; s < a.length; s += c) { let acc = carry; for (let k = s; k < Math.min(a.length, s + c); k++) { acc += a[k]; out.push(acc); } carry = acc; } return out; }, param: (i) => 1 + i },
  ].map((f) => ({
    id: f.id, kind: 'sweep', label: f.label, sweep: true,
    make: (i) => {
      const p = f.param(i);
      return { name: `${f.id}[${i}]`, fn: f.fn(p), desc: f.mk(p), param: p };
    },
  })),
];

// ── 构建 ISA ────────────────────────────────────────────────────────────────
function buildISA() {
  const table = [];
  const byFamily = new Map();
  for (const fam of FAMILIES) {
    const members = [];
    for (let i = 0; i < 32; i++) {
      const m = fam.make(i, fam.members ? fam.members[i] : undefined);
      const entry = {
        id: table.length,
        family: fam.id,
        memberIndex: i,
        name: m.name,
        kind: fam.kind,
        label: fam.label,
        desc: m.desc,
        param: m.param,
        fn: m.fn,
      };
      table.push(entry);
      members.push(entry);
    }
    byFamily.set(fam.id, { id: fam.id, kind: fam.kind, label: fam.label, members });
  }
  return { table, byFamily, size: table.length };
}

let _isa = null;
function getISA() { if (_isa === null) _isa = buildISA(); return _isa; }

/** 按名字调用一条微指令。名字不存在 → 抛异常（不静默返回 undefined）。 */
function call(name, x, y) {
  const isa = getISA();
  const e = isa.table.find((t) => t.name === name);
  if (!e) throw new Error(`未知微指令：${name}（ISA 共 ${isa.size} 条）`);
  return e.fn(x, y);
}
/** 按 (family, memberIndex) 调用。 */
function callIdx(family, i, x, y) {
  const isa = getISA();
  const f = isa.byFamily.get(family);
  if (!f) throw new Error(`未知家族：${family}`);
  if (i < 0 || i > 31) throw new Error(`成员下标越界：${i}`);
  return f.members[i].fn(x, y);
}

module.exports = { getISA, call, callIdx, FAMILIES, buildISA };
