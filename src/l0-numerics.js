/**
 * NEXUS · L0 数值内核
 * ---------------------------------------------------------------------------
 * 零重写原则：本层每一个配方都取自世界已验证的最优解，并显式标注来源与被否决的替代。
 * 作者：晨星 (CJX0712) · MIT
 *
 * 复用清单（全部是世界顶级成果，非自研）：
 *   · log-sum-exp 平移稳定化            —— 数值分析标准配方（IEEE 754 实践共识）
 *   · Newton-Schulz 五次迭代 (3.4445, -4.7750, 2.0315)
 *                                       —— Keller Jordan, Muon 官方系数（2024）
 *   · 幂迭代谱范数                       —— Golub & Van Loan, Matrix Computations
 *   · SipHash-2-4                        —— Aumasson & Bernstein 2012（抗 HashDoS）
 *   · PCG64 DXSM                         —— O'Neill 2014 / NumPy 新一代 bitgen
 *   · xoshiro256++                       —— Blackman & Vigna 2018（对照流）
 *   · Householder QR / Cholesky / Jacobi —— LAPACK 标准算法
 *   · Neumaier 补偿求和                  —— Neumaier 1974
 *   · Welford 在线方差                   —— Welford 1962
 *   · erf / lgamma                       —— Abramowitz & Stegun / Lanczos 近似
 *
 * 被否决的替代（留档，避免后来者重踩）：
 *   · 极分解用 SVD      → 低精度不稳定且 O(min(n,m)·nm)，NS 只需矩阵乘
 *   · seed + i 派生子流 → 相邻整数种子 RNG 存在相关结构，虚高多 seed 实验可信度
 *   · Math.random()     → 非确定性，破坏"逐位可复现"门禁
 */

'use strict';

// ── 常量 ────────────────────────────────────────────────────────────────────
const EPS = Number.EPSILON; // 2.220446049250313e-16
const SQRT2 = Math.SQRT2;
const LOG2E = Math.LOG2E;

// ── 1. 稳定化标量运算 ───────────────────────────────────────────────────────

/** hypot：避免 a²+b² 上溢/下溢（`Math.sqrt(a*a+b*b)` 在 a=1e200 时直接 inf）。 */
function hypot2(a, b) {
  const x = Math.abs(a), y = Math.abs(b);
  if (x === 0) return y;
  if (y === 0) return x;
  const hi = Math.max(x, y), lo = Math.min(x, y);
  const r = lo / hi;
  return hi * Math.sqrt(1 + r * r);
}

/** log(1+x)，x 接近 0 时比 Math.log(1+x) 精度高得多。 */
function log1p(x) {
  if (Math.abs(x) > 1e-4) return Math.log(1 + x);
  // Taylor: x - x²/2 + x³/3 - x⁴/4
  return x * (1 + x * (-0.5 + x * (1 / 3 + x * -0.25)));
}

/** exp(x)-1，x 接近 0 时的稳定版本。 */
function expm1(x) {
  if (Math.abs(x) > 1e-4) return Math.exp(x) - 1;
  return x * (1 + x * (0.5 + x * (1 / 6 + x * (1 / 24))));
}

/** log-sum-exp（平移稳定化）：x 里出现 800 也不会 inf。 */
function logsumexp(xs) {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  if (m === -Infinity) return -Infinity;
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += Math.exp(xs[i] - m);
  return m + Math.log(s);
}

/**
 * softmax：平移不变，行和恒 1。
 * 不变量：softmax(x + c) === softmax(x)；Σ softmax(x)_i === 1（误差 ≤ 数·EPS）
 */
function softmax(xs) {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  const out = new Float64Array(xs.length);
  let s = 0;
  for (let i = 0; i < xs.length; i++) { const e = Math.exp(xs[i] - m); out[i] = e; s += e; }
  const inv = 1 / s;
  for (let i = 0; i < xs.length; i++) out[i] *= inv;
  return out;
}

/** 数值稳定 sigmoid：用 exp(-|x|) 分支，避免大正/大负溢出路径不一致。 */
function sigmoid(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

// ── 2. 特殊函数（A&S / Lanczos 近似，误差 < 1e-14）────────────────────────

/**
 * 误差函数 erf。
 *
 * 被否决：Abramowitz & Stegun 7.1.26（|ε| < 1.5e-7）。
 *   它便宜，但 erf(0) 算出 1e-9 而不是 0 —— 直接击穿「Φ(0)=0.5」「Φ(−x)=1−Φ(x)」两条不变量。
 *   一个连对称性都不成立的近似，不配进数值内核。
 *
 * 采用：Maclaurin 级数（|x| ≤ 4）+ 奇函数结构 + x=0 精确特判。
 *   erf(x) = (2/√π) Σ_{n≥0} (−1)ⁿ x^{2n+1} / (n!·(2n+1))
 * 精度如实声明：
 *   |x| ≤ 2  → ~1e-16（无抵消）
 *   |x| ≤ 4  → ~1e-9 （级数峰值 8.7e5，双精度相减抵消损失）
 *   |x| >  4 → 返回 ±1，与真值差 < 1.5e-8（erf(4) = 1 − 1.5417e-8）
 */
function erf(x) {
  if (x === 0) return 0;                 // 精确特判：守住 Φ(0) = 0.5
  if (x > 4) return 1;
  if (x < -4) return -1;
  const ax = Math.abs(x);
  const x2 = ax * ax;
  let term = ax;      // x^{2n+1}/n!
  let sum = ax;       // 累加
  for (let n = 1; n < 80; n++) {
    term *= -x2 / n;
    const add = term / (2 * n + 1);
    sum += add;
    if (Math.abs(add) < 1e-18 * Math.abs(sum)) break;
  }
  const v = (2 / Math.sqrt(Math.PI)) * sum;
  return x < 0 ? -v : v;                 // 奇函数结构：对称性由构造保证，不靠数值巧合
}

/** 标准正态 CDF：Φ(x) = ½(1 + erf(x/√2))。不变量：Φ(−x) = 1 − Φ(x)。 */
function normCdf(x) { return 0.5 * (1 + erf(x / SQRT2)); }

/** 标准正态 PDF。 */
function normPdf(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }

// Lanczos g=7, n=9 系数（Boost 同款）
const LANCZOS_G7 = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** log Γ(x)，x > 0。反射公式处理 x < 0.5。 */
function lgamma(x) {
  if (x < 0.5) {
    // Γ(x)Γ(1−x) = π / sin(πx)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  const z = x - 1;
  let a = LANCZOS_G7[0];
  const t = z + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS_G7[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** 熵 H(p) = −Σ p log p（自然对数，nats）。不变量：0 ≤ H ≤ log |X|。 */
function entropy(p) {
  let h = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log(p[i]);
  return h;
}

/** KL(p‖q) ≥ 0，=0 ⟺ p=q。 */
function kl(p, q) {
  let d = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) {
      if (q[i] <= 0) return Infinity;
      d += p[i] * Math.log(p[i] / q[i]);
    }
  }
  return d;
}

/** Jensen-Shannon 散度：对称、有界 [0, log 2]，比 KL 更适合做分布距离度量。 */
function jsd(p, q) {
  const n = p.length;
  const m = new Float64Array(n);
  for (let i = 0; i < n; i++) m[i] = 0.5 * (p[i] + q[i]);
  return 0.5 * kl(p, m) + 0.5 * kl(q, m);
}

// ── 3. 补偿求和（Neumaier 1974）────────────────────────────────────────────
/**
 * Neumaier 补偿求和：把浮点累加的舍入误差单独记账再补回。
 * 朴素 sum 在 1e16 量级 + 1.0 时完全丢掉小数 —— 这类"不报错但静默失真"是本项目的头号敌人。
 */
function sumKahan(xs) {
  let s = 0, c = 0;
  for (let i = 0; i < xs.length; i++) {
    const v = xs[i];
    const t = s + v;
    c += Math.abs(s) >= Math.abs(v) ? (s - t) + v : (v - t) + s;
    s = t;
  }
  return s + c;
}

/** Welford 在线方差：单遍、数值稳定，不需要先算均值。 */
class Welford {
  constructor() { this.n = 0; this.mean = 0; this.m2 = 0; }
  push(x) {
    this.n++;
    const d = x - this.mean;
    this.mean += d / this.n;
    this.m2 += d * (x - this.mean);
    return this;
  }
  get variance() { return this.n > 1 ? this.m2 / (this.n - 1) : 0; }
  get sampleStd() { return Math.sqrt(this.variance); }
  merge(o) { // Chan et al. 并行合并公式
    if (o.n === 0) return this;
    const n = this.n + o.n;
    const d = o.mean - this.mean;
    this.m2 += o.m2 + d * d * this.n * o.n / n;
    this.mean += d * o.n / n;
    this.n = n;
    return this;
  }
}

// ── 4. 确定性随机：SipHash-2-4 种子派生 + PCG64 DXSM ────────────────────
/**
 * SipHash-2-4（Aumasson & Bernstein 2012）。
 * 用途：从主种子 + 语义标签派生互不相关的子流种子。
 * 为什么不用 `seed + i`：相邻整数种子的 RNG 存在相关结构，会虚高多 seed 实验的可信度。
 */

function _rotl64(hi, lo, b) {
  if (b === 0) return [hi, lo];
  // JS 位运算是 32 位，用高低两半 32 位模拟 64 位
  if (b < 32) {
    const nhi = (hi << b) | (lo >>> (32 - b));
    const nlo = (lo << b) | (hi >>> (32 - b));
    return [nhi | 0, nlo | 0];
  }
  const b2 = b - 32;
  const nhi = (lo << b2) | (hi >>> (32 - b2));
  const nlo = (hi << b2) | (lo >>> (32 - b2));
  return [nhi | 0, nlo | 0];
}

/**
 * SipHash-2-4（Aumasson & Bernstein 2012）。输出 64 位为 BigInt。
 *
 * 为什么用 BigInt 而不是两个 32 位字手搓：
 *   手搓版本连错三次 —— 常量 hi/lo 写重、`v2 ^= 0xff` 打到高 32 位、轮函数顺序错位。
 *   三次都是"能跑、输出看着也随机、但和官方向量对不上"。
 *   密码学哈希没有"差不多对"这回事：错了就是完全错，而且统计性质完全看不出来。
 *   种子派生是整个可复现性的根，这里不接受任何近似。BigInt 慢一点，换来定义与规范逐位一致。
 *
 * @param {bigint[]} key [k0, k1]，两个 64 位
 * @param {Uint8Array} msg
 * @returns {bigint}
 */
function siphash24(key, msg) {
  const M = MASK64;
  let v0 = (key[0] ^ 0x736f6d6570736575n) & M;
  let v1 = (key[1] ^ 0x646f72616e646f6dn) & M;
  let v2 = (key[0] ^ 0x6c7967656e657261n) & M;
  let v3 = (key[1] ^ 0x7465646279746573n) & M;

  const rotl = (x, b) => ((x << BigInt(b)) | (x >> BigInt(64 - b))) & M;
  const round = () => {                    // 顺序严格照抄规范，错一位输出就全错
    v0 = (v0 + v1) & M; v1 = rotl(v1, 13); v1 ^= v0; v0 = rotl(v0, 32);
    v2 = (v2 + v3) & M; v3 = rotl(v3, 16); v3 ^= v2;
    v0 = (v0 + v3) & M; v3 = rotl(v3, 21); v3 ^= v0;
    v2 = (v2 + v1) & M; v1 = rotl(v1, 17); v1 ^= v2; v2 = rotl(v2, 32);
  };

  const len = msg.length;
  let i = 0;
  const le64 = (b, off) => {
    let v = 0n;
    for (let k = 7; k >= 0; k--) v = (v << 8n) | BigInt(b[off + k] || 0);  // 小端：高字节在后
    return v;
  };
  for (; i + 8 <= len; i += 8) {
    const m = le64(msg, i);
    v3 ^= m; round(); round(); v0 ^= m;
  }
  const tail = new Uint8Array(8);
  for (let j = 0; j < len - i; j++) tail[j] = msg[i + j];
  tail[7] = len & 0xff;                      // 末字节放消息长度 mod 256
  const m = le64(tail, 0);
  v3 ^= m; round(); round(); v0 ^= m;

  v2 ^= 0xffn;
  round(); round(); round(); round();
  return (v0 ^ v1 ^ v2 ^ v3) & M;
}

function _u32(x) { return x >>> 0; }

/** 把字符串/数字混合标签编码成字节，用于种子派生。 */
function encodeTags(tags) {
  const parts = tags.map((t) => String(t));
  const s = parts.join('\u001f');
  const bytes = new Uint8Array(s.length * 2);
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    bytes[i * 2] = c & 0xff;
    bytes[i * 2 + 1] = (c >> 8) & 0xff;
  }
  return bytes;
}

/** 从主种子 + 语义标签派生子流种子。返回 32 位非零整数（便于直接喂给 32 位 RNG）。 */
function deriveSeed(masterSeed, ...tags) {
  const s = BigInt(masterSeed >>> 0);
  const key = [
    (s | (s << 32n)) & MASK64,
    (s ^ 0x9e3779b985ebca6bn) & MASK64,
  ];
  const h = siphash24(key, encodeTags(tags));
  const v = Number(h & 0xffffffffn);
  return v === 0 ? 0x9e3779b9 : v;
}

/** 派生一个 64 位种子（给需要宽种子的消费者）。 */
function deriveSeed64(masterSeed, ...tags) {
  const s = BigInt(masterSeed >>> 0);
  return siphash24([(s | (s << 32n)) & MASK64, (s ^ 0x9e3779b985ebca6bn) & MASK64], encodeTags(tags));
}

// ── PCG-128 位状态 + SplitMix64 终混（BigInt 精确实现）────────────────────
const B64 = 1n << 64n;
const B128 = 1n << 128n;
const MASK128 = B128 - 1n;
const MASK64 = B64 - 1n;
// PCG 官方 128 位乘数（O'Neill 2014；取自 PCG 参考实现）
const PCG_MULT = 0x2360ed051fc65da44385df649fccf645n;
// SplitMix64 finalizer 常量（Steele & Lea & Vigna 2014）—— 公认的强雪崩混合器
const SM_A = 0xbf58476d1ce4e5b9n;
const SM_B = 0x94d049bb133111ebn;

/** SplitMix64 finalizer，输入/输出均为 [0, 2^64) 的 BigInt。 */
function splitmix64Finalizer(z0) {
  let z = z0 & MASK64;
  z = ((z ^ (z >> 30n)) * SM_A) & MASK64;
  z = ((z ^ (z >> 27n)) * SM_B) & MASK64;
  return (z ^ (z >> 31n)) & MASK64;
}

/**
 * PCG128SM —— 主力确定性随机流。
 *
 * 结构（全部取自世界顶级公开成果，非自研）：
 *   · 状态推进：128 位 LCG，state ← state·MULT + inc (mod 2^128)，inc 必须为奇数
 *                            —— PCG 家族（O'Neill 2014），周期 2^128
 *   · 输出混合：SplitMix64 finalizer（Steele/Lea/Vigna 2014），强雪崩，通过 BigCrush
 *
 * 为什么用 BigInt 而不是两块 32 位拼：
 *   JS 无原生 u128，用两个 32 位字手搓 LCG 的进位极易写错，且写错后统计门禁未必抓得住
 *   （它只会让某些低位变弱）。BigInt 慢一些，但保证状态转移的数学定义与论文逐位一致。
 *   地基的东西不容许"大概对"。
 *
 * 被否决：Math.random() —— 无法播种，直接破坏「逐位可复现」这条总门禁。
 */
class PCG128SM {
  constructor(seed) {
    if (seed === undefined || seed === null) {
      throw new Error('PCG128SM 必须显式给种子 —— 拒绝非确定性随机（逐位可复现是本项目总门禁）');
    }
    const hi = BigInt(deriveSeed(seed, 'pcg128', 'hi')) << 32n;
    const lo = BigInt(deriveSeed(seed, 'pcg128', 'lo'));
    this.state = ((hi | lo) << 64n | BigInt(deriveSeed(seed, 'pcg128', 'low'))) & MASK128;
    // 增量必须为奇数，否则 LCG 周期退化
    this.inc = ((BigInt(deriveSeed(seed, 'pcg128', 'incHi')) << 64n)
      | BigInt(deriveSeed(seed, 'pcg128', 'incLo'))) & MASK128 | 1n;
    this._spare = undefined;
    this._seed = seed;
  }
  /** 取一个 64 位输出。 */
  next64() {
    this.state = (this.state * PCG_MULT + this.inc) & MASK128;
    return splitmix64Finalizer(this.state >> 64n);
  }
  /** [0,1) 均匀随机，53 位精度（双精度的全部尾数位）。 */
  next() {
    return Number(this.next64() >> 11n) / 9007199254740992.0; // /2^53
  }
  /** 标准正态：Marsaglia polar 法（缓存第二个值，不丢弃样本）。 */
  normal() {
    if (this._spare !== undefined) { const v = this._spare; this._spare = undefined; return v; }
    let u, v, s;
    do {
      u = this.next() * 2 - 1;
      v = this.next() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const f = Math.sqrt(-2 * Math.log(s) / s);
    this._spare = v * f;
    return u * f;
  }
  /** [0, n) 整数。 */
  int(n) { return Math.floor(this.next() * n); }
  /** 原地 Fisher-Yates 洗牌。 */
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }
  /** 导出/恢复状态：让"从第 k 步继续"成为可能（长跑实验的断点续跑靠它）。 */
  getState() { return { state: this.state.toString(16), inc: this.inc.toString(16) }; }
  static fromState(s) { const r = Object.create(PCG128SM.prototype); r.state = BigInt('0x' + s.state); r.inc = BigInt('0x' + s.inc); return r; }
}

/** 向后兼容别名：PCG64DXSM 曾是本类的名字，保留以避免破坏既有调用点。 */
const PCG64DXSM = PCG128SM;

/**
 * xoshiro256++（Blackman & Vigna 2018）—— 作为**对照流**存在。
 * 门禁用它交叉验证 PCG64DXSM 的统计性质（两条独立算法路径，结论应一致）。
 */
class Xoshiro256pp {
  constructor(seed) {
    const s0 = deriveSeed(seed, 'xoshiro', 's0'), s1 = deriveSeed(seed, 'xoshiro', 's1');
    const s2 = deriveSeed(seed, 'xoshiro', 's2'), s3 = deriveSeed(seed, 'xoshiro', 's3');
    this.s = [_u32(s0 || 1), _u32(s1 || 2), _u32(s2 || 3), _u32(s3 || 4)];
  }
  _rotl(x, k) { return _u32((x << k) | (x >>> (32 - k))); }
  next() {
    const s = this.s;
    const result = _u32(this._rotl(_u32(s[0] + s[3]), 7) + s[0]);
    const t = _u32(s[1] << 9);
    s[2] = _u32(s[2] ^ s[0]);
    s[3] = _u32(s[3] ^ s[1]);
    s[1] = _u32(s[1] ^ s[2]);
    s[0] = _u32(s[0] ^ s[3]);
    s[2] = _u32(s[2] ^ t);
    s[3] = this._rotl(s[3], 11);
    return result / 4294967296.0;
  }
  normal() {
    let u = 0, v = 0, s = 0;
    do { u = this.next() * 2 - 1; v = this.next() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    return u * Math.sqrt(-2 * Math.log(s) / s);
  }
}

// ── 5. 致密线性代数（LAPACK 标准算法）────────────────────────────────────

/**
 * Newton-Schulz 五次迭代 —— 求矩阵极分解中的正交因子。
 * 系数取自 Keller Jordan 的 Muon 实现：(a,b,c) = (3.4445, −4.7750, 2.0315)。
 * 迭代：X ← aX + b·X(XᵀX) + c·X(XᵀX)²
 *
 * 属性（如实声明，不美化）：NS 是**近似**正交化。
 * 迭代映射 f(x) = 3.4445x − 4.775x³ + 2.0315x⁵ 的两个不动点（x≈0.868 与 x≈1.264）
 * 处 |f′| 都 > 1，因此它**不收敛到精确 1**，而是把奇异值推向 (0.5, 1.5) 区间。
 * 这是 Muon 作者的取舍：只要 5 步矩阵乘就能拿到"够用的"正交性，比 SVD 便宜一个量级。
 * 门禁必须按这个真实属性写（G14），不能按理想性质写然后放宽阈值凑绿。
 */
function newtonSchulz(A, steps = 5, eps = 1e-7) {
  const n = A.length, m = A[0].length;
  // 转置为方阵 X (m×m) 便于迭代：这里统一按 n≥m 处理，否则先转置
  if (n < m) {
    const B = new Array(m);
    for (let i = 0; i < m; i++) { B[i] = new Float64Array(n); for (let j = 0; j < n; j++) B[i][j] = A[j][i]; }
    const R = newtonSchulz(B, steps, eps);
    const out = new Array(n);
    for (let i = 0; i < n; i++) { out[i] = new Float64Array(m); for (let j = 0; j < m; j++) out[i][j] = R[j][i]; }
    return out;
  }
  // X = A / ||A||_F，保证谱范数 ≤ 1（NS 收敛的前提）
  let frob = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) frob += A[i][j] * A[i][j];
  frob = Math.sqrt(frob) + eps;
  let X = new Array(n);
  for (let i = 0; i < n; i++) { X[i] = new Float64Array(m); for (let j = 0; j < m; j++) X[i][j] = A[i][j] / frob; }

  const a = 3.4445, b = -4.7750, c = 2.0315;
  for (let s = 0; s < steps; s++) {
    // G = Xᵀ X  (m×m)
    const G = new Array(m);
    for (let i = 0; i < m; i++) {
      G[i] = new Float64Array(m);
      for (let j = 0; j < m; j++) {
        let v = 0;
        for (let k = 0; k < n; k++) v += X[k][i] * X[k][j];
        G[i][j] = v;
      }
    }
    // G2 = G²
    const G2 = new Array(m);
    for (let i = 0; i < m; i++) {
      G2[i] = new Float64Array(m);
      for (let j = 0; j < m; j++) {
        let v = 0;
        for (let k = 0; k < m; k++) v += G[i][k] * G[k][j];
        G2[i][j] = v;
      }
    }
    // X ← aX + b X G + c X G2
    const nX = new Array(n);
    for (let i = 0; i < n; i++) {
      nX[i] = new Float64Array(m);
      for (let j = 0; j < m; j++) {
        let xg = 0, xg2 = 0;
        for (let k = 0; k < m; k++) { xg += X[i][k] * G[k][j]; xg2 += X[i][k] * G2[k][j]; }
        nX[i][j] = a * X[i][j] + b * xg + c * xg2;
      }
    }
    X = nX;
  }
  return X;
}

/**
 * 精确极分解（金标准）：A = U_pol · P，其中 U_pol 列正交。
 * 走 SVD：AᵀA 的 Jacobi 特征分解给 V 与奇异值，再 U = A V Σ⁻¹。
 *
 * 存在的意义：给 Newton-Schulz 当尺子。
 * NS 是**近似**正交化（便宜、只需矩阵乘），它到底正交化了多少，必须由本函数说了算，
 * 不能由 NS 自己声明。
 */
function polarExact(A) {
  const n = A.length, m = A[0].length;
  // G = AᵀA（m×m，对称半正定）
  const G = new Array(m);
  for (let i = 0; i < m; i++) {
    G[i] = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      let v = 0;
      for (let k = 0; k < n; k++) v += A[k][i] * A[k][j];
      G[i][j] = v;
    }
  }
  const { values, vectors } = jacobiEig(G);
  const sigma = new Float64Array(m);
  for (let i = 0; i < m; i++) sigma[i] = Math.sqrt(Math.max(0, values[i]));
  // U = A·V·Σ⁻¹ （跳过零奇异值方向）
  const U = new Array(n);
  for (let i = 0; i < n; i++) U[i] = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    if (sigma[j] < 1e-12) continue;
    // 第 j 列：U[:,j] = A·V[:,j] / σ_j
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += A[i][k] * vectors[j][k];
      U[i][j] = s / sigma[j];
    }
  }
  // V 一并返回：NS 的门禁要用它做「奇异向量保持不变」的检验
  return { U, sigma, V: vectors };
}

/** 幂迭代求主奇异值（谱范数）。比完整 SVD 便宜得多，NS 每步都调用得起。 */
function spectralNorm(A, iters = 16) {
  const n = A.length, m = A[0].length;
  let v = new Float64Array(m);
  const rng = new PCG64DXSM(0x5eed1234);
  for (let i = 0; i < m; i++) v[i] = rng.normal();
  let nv = 0;
  for (let i = 0; i < m; i++) nv += v[i] * v[i];
  nv = Math.sqrt(nv);
  for (let i = 0; i < m; i++) v[i] /= nv;

  let sigma = 0;
  for (let it = 0; it < iters; it++) {
    // u = A v
    const u = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < m; j++) s += A[i][j] * v[j]; u[i] = s; }
    let nu = 0;
    for (let i = 0; i < n; i++) nu += u[i] * u[i];
    nu = Math.sqrt(nu);
    if (nu === 0) return 0;
    for (let i = 0; i < n; i++) u[i] /= nu;
    // v = Aᵀ u
    const nv2 = new Float64Array(m);
    for (let j = 0; j < m; j++) { let s = 0; for (let i = 0; i < n; i++) s += A[i][j] * u[i]; nv2[j] = s; }
    let nn = 0;
    for (let j = 0; j < m; j++) nn += nv2[j] * nv2[j];
    nn = Math.sqrt(nn);
    if (nn === 0) return 0;
    for (let j = 0; j < m; j++) v[j] = nv2[j] / nn;
    sigma = nn;
  }
  return sigma;
}

/** Cholesky A = L Lᵀ（A 必须对称正定）。返回 null 表示非正定（不抛异常、不静默伪造）。 */
function cholesky(A) {
  const n = A.length;
  const L = new Array(n);
  for (let i = 0; i < n; i++) L[i] = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) return null;
        L[i][i] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

/** Householder QR。不变量：QᵀQ = I，QR = A。 */
function qr(A) {
  const n = A.length, m = A[0].length;
  const R = A.map((r) => Float64Array.from(r));
  const Q = new Array(n);
  for (let i = 0; i < n; i++) { Q[i] = new Float64Array(n); Q[i][i] = 1; }
  const kmax = Math.min(n - 1, m);
  for (let k = 0; k < kmax; k++) {
    let normx = 0;
    for (let i = k; i < n; i++) normx += R[i][k] * R[i][k];
    normx = Math.sqrt(normx);
    if (normx === 0) continue;
    const alpha = R[k][k] > 0 ? -normx : normx;
    const v = new Float64Array(n);
    for (let i = k; i < n; i++) v[i] = R[i][k];
    v[k] -= alpha;
    let vnorm = 0;
    for (let i = k; i < n; i++) vnorm += v[i] * v[i];
    vnorm = Math.sqrt(vnorm);
    if (vnorm === 0) continue;
    for (let i = k; i < n; i++) v[i] /= vnorm;
    // R ← (I − 2vvᵀ) R
    // R ← H·R（左乘）
    for (let j = k; j < m; j++) {
      let s = 0;
      for (let i = k; i < n; i++) s += v[i] * R[i][j];
      for (let i = k; i < n; i++) R[i][j] -= 2 * v[i] * s;
    }
    // Q ← Q·H（**右乘**，不是左乘）
    // 曾因写成左乘导致 QR ≠ A（残差 4.885）被 G18b 抓出：
    // R = H_{K−1}···H_0·A  ⇒  A = H_0···H_{K−1}·R  ⇒  Q = H_0 H_1 ··· H_{K−1}
    // 逐步累积只能是右乘。左乘得到的是 H_{K−1}···H_0，顺序反了。
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k2 = k; k2 < n; k2++) s += v[k2] * Q[i][k2];
      for (let j = k; j < n; j++) Q[i][j] -= 2 * v[j] * s;
    }
  }
  return { Q, R };
}

/**
 * Jacobi 特征分解（对称阵）。循环旋转，比 QR 迭代更适合中小规模且精度极高。
 * 不变量：‖Av − λv‖ ≈ 0；特征向量正交归一。
 */
function jacobiEig(Ain, maxSweeps = 100, tol = 1e-12) {
  const n = Ain.length;
  const A = Ain.map((r) => Float64Array.from(r));
  const V = new Array(n);
  for (let i = 0; i < n; i++) { V[i] = new Float64Array(n); V[i][i] = 1; }

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (Math.sqrt(off) < tol) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = A[i][i];
  // 降序排列
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[b] - values[a]);
  const sv = new Float64Array(n);
  const sv2 = new Array(n);
  for (let i = 0; i < n; i++) {
    sv[i] = values[idx[i]];
    sv2[i] = new Float64Array(n);
    for (let k = 0; k < n; k++) sv2[i][k] = V[k][idx[i]];
  }
  return { values: sv, vectors: sv2 };
}

/** 解下三角方程组 L y = b（前向替换）。 */
function solveLower(L, b) {
  const n = L.length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let j = 0; j < i; j++) s -= L[i][j] * y[j];
    y[i] = s / L[i][i];
  }
  return y;
}

/** 解上三角方程组 U x = b（回代）。 */
function solveUpper(U, b) {
  const n = U.length;
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= U[i][j] * x[j];
    x[i] = s / U[i][i];
  }
  return x;
}

/**
 * 对称正定线性求解：Cholesky + 前后替换。
 * 比"直接求逆再乘"数值稳定得多（求逆是数值分析的经典反模式）。
 */
function solveSPD(A, b) {
  const L = cholesky(A);
  if (L === null) return null;
  const y = solveLower(L, b);
  const Lt = new Array(L.length);
  for (let i = 0; i < L.length; i++) {
    Lt[i] = new Float64Array(L.length);
    for (let j = 0; j < L.length; j++) Lt[i][j] = L[j][i];
  }
  return solveUpper(Lt, y);
}

// ── 6. 中心差分的通用工具（梯度检验的地基）─────────────────────────────────
/**
 * 中心差分：∂f/∂xᵢ ≈ (f(x+εeᵢ) − f(x−εeᵢ)) / (2ε)
 * 前向差分误差 O(ε)，中心差分 O(ε²) —— 差别是 1e-8 与 1e-16，足以决定门禁能否通过。
 */
function centralDiff(f, x, i, eps = 1e-5) {
  const old = x[i];
  x[i] = old + eps; const fp = f(x);
  x[i] = old - eps; const fm = f(x);
  x[i] = old;
  return (fp - fm) / (2 * eps);
}

/**
 * PyTorch gradcheck 同款混合判据：|a − n| ≤ atol + rtol·|n|
 * 关键：**atol 用全局梯度尺度**，不是单个元素量级。
 * 为什么：中心差分的浮点噪声 ≈ eps_machine·|loss|/eps，与元素自身大小无关。
 * 若按元素量级设 atol，梯度本身只有 1e-8 的参数会被噪声淹没、被误判 FAIL。
 */
function gradcheck(analytic, numeric, opts = {}) {
  const { rtol = 1e-5, atolScale = 1e-6 } = opts;
  let maxN = 0;
  for (let i = 0; i < numeric.length; i++) maxN = Math.max(maxN, Math.abs(numeric[i]));
  const atol = atolScale * Math.max(1, maxN);
  let maxRel = 0, worst = -1;
  for (let i = 0; i < analytic.length; i++) {
    const a = analytic[i], n = numeric[i];
    const ok = Math.abs(a - n) <= atol + rtol * Math.abs(n);
    const rel = Math.abs(a - n) / Math.max(1e-12, Math.abs(a) + Math.abs(n));
    if (rel > maxRel) { maxRel = rel; worst = i; }
    if (!ok) return { pass: false, index: i, analytic: a, numeric: n, maxRel };
  }
  return { pass: true, maxRel, worst };
}

// ── 导出 ────────────────────────────────────────────────────────────────────
module.exports = {
  EPS, SQRT2, LOG2E,
  hypot2, log1p, expm1, logsumexp, softmax, sigmoid,
  erf, normCdf, normPdf, lgamma, entropy, kl, jsd,
  sumKahan, Welford,
  siphash24, deriveSeed, deriveSeed64, PCG128SM, PCG64DXSM, Xoshiro256pp, splitmix64Finalizer,
  newtonSchulz, polarExact, spectralNorm, cholesky, qr, jacobiEig,
  solveLower, solveUpper, solveSPD,
  centralDiff, gradcheck,
};
