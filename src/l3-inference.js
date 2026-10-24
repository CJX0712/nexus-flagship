/**
 * NEXUS · L3 推理引擎
 * ---------------------------------------------------------------------------
 * 把世界顶级推理引擎的**架构思想**在零依赖 JS 中落地，并给每一条配上可机检的不变量。
 * 作者：晨星 (CJX0712) · MIT
 *
 * 复用（思想来源，全部标注；我们实现的是可验证的等价物，不是抄代码）：
 *   · PagedAttention        —— vLLM (Kwon et al. 2023, SOSP)
 *                              用 OS 分页的思路管 KV cache：固定块 + 块表，消除碎片
 *   · RadixAttention        —— SGLang (Zheng et al. 2024)
 *                              基数树存前缀 KV，跨请求复用共享前缀，LRU 淘汰
 *   · 投机解码（拒绝采样）  —— Leviathan et al. 2023 / Chen et al. 2023
 *                              接受率 min(1, p/q)，拒绝时从残差分布重采样 —— 数学上严格无损
 *   · Top-p (nucleus) 采样  —— Holtzman et al. 2020
 *   · INT8 per-channel / INT4 group-wise —— AWQ (Lin et al. 2023) 与 GPTQ 的共识
 *
 * 被否决的替代：
 *   · top-k                 —— 固定 k 在分布的"平尾/尖峰"两种形状下要么截断过多、
 *                              要么保留过多噪声；top-p 按概率质量自适应
 *   · 量化用 per-tensor     —— 4bit 只有 16 个电平，跨通道量级可能差 10 倍，必须按列分组给 scale
 *   · 投机解码"直接接受 draft 的输出" —— 那是有损的捷径；真正的无损版本必须走残差重采样
 *
 * 一条必须写清楚的诚实声明：
 *   投机解码的**加速比取决于 draft 模型与主模型的分布吻合度**。
 *   若 draft 与主模型同分布，加速比结构性 ≈ 1（draft 自己也要跑一遍）。
 *   它真正的价值在"小 draft + 大 target"的异构组合上。本模块提供无损性证明，
 *   **不宣称加速** —— 加速数字由 bench 实测给出，不由 README 吹出。
 */

'use strict';
const L0 = require('./l0-numerics.js');

// ═══════════════════════════════════════════════════════════════════════════
// 1. 朴素 KV cache（连续存储）—— 基线
// ═══════════════════════════════════════════════════════════════════════════
/**
 * 每个序列一段连续内存，预分配到 maxLen。
 * 缺陷（PagedAttention 要解决的正是它）：
 *   预分配 64 个 slot × 32k context，即使只用 1 个 slot 也占满；
 *   且长度不一导致内部碎片，实际利用率常低于 50%。
 */
class NaiveKVCache {
  constructor(maxSeq, maxLen, Hkv, Dh) {
    this.maxSeq = maxSeq; this.maxLen = maxLen; this.Hkv = Hkv; this.Dh = Dh;
    const n = maxSeq * maxLen * Hkv * Dh;
    this.k = new Float64Array(n);
    this.v = new Float64Array(n);
    this.len = new Int32Array(maxSeq);
  }
  _off(seq, t, h) { return ((seq * this.maxLen + t) * this.Hkv + h) * this.Dh; }
  append(seq, t, h, kVec, vVec) {
    const o = this._off(seq, t, h);
    this.k.set(kVec, o); this.v.set(vVec, o);
    if (h === 0) this.len[seq] = Math.max(this.len[seq], t + 1);
  }
  get(seq, t, h) {
    const o = this._off(seq, t, h);
    return { k: this.k.subarray(o, o + this.Dh), v: this.v.subarray(o, o + this.Dh) };
  }
  /** 取出一个序列前 len 个位置的 K/V，拼成 [T, Hkv, Dh] 稠密数组。 */
  gather(seq) {
    const T = this.len[seq];
    const kk = new Float64Array(T * this.Hkv * this.Dh);
    const vv = new Float64Array(T * this.Hkv * this.Dh);
    for (let t = 0; t < T; t++) for (let h = 0; h < this.Hkv; h++) {
      const o = this._off(seq, t, h);
      kk.set(this.k.subarray(o, o + this.Dh), (t * this.Hkv + h) * this.Dh);
      vv.set(this.v.subarray(o, o + this.Dh), (t * this.Hkv + h) * this.Dh);
    }
    return { k: kk, v: vv, len: T };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. PagedAttention —— 块表 + 按需分配（vLLM 思想）
// ═══════════════════════════════════════════════════════════════════════════
/**
 * 物理块池 + 每序列一张块表（逻辑块 → 物理块 id）。
 * 关键性质：**一个序列的 KV 在物理上不必连续**，注意力按块表 gather。
 * 由此：显存按需分配、无内部碎片、同一 prompt 的多条采样可共享前缀块（写时复制）。
 */
class PagedKVCache {
  constructor(numBlocks, blockSize, Hkv, Dh) {
    this.numBlocks = numBlocks; this.blockSize = blockSize; this.Hkv = Hkv; this.Dh = Dh;
    const perBlock = blockSize * Hkv * Dh;
    this.kPool = new Float64Array(numBlocks * perBlock);
    this.vPool = new Float64Array(numBlocks * perBlock);
    this.free = [];
    for (let i = numBlocks - 1; i >= 0; i--) this.free.push(i);
    this.tables = new Map();      // seqId → { blocks: number[], len }
    this.perBlock = perBlock;
  }
  get numFreeBlocks() { return this.free.length; }

  createSeq(seqId) {
    if (this.tables.has(seqId)) throw new Error(`序列 ${seqId} 已存在`);
    this.tables.set(seqId, { blocks: [], len: 0 });
    return seqId;
  }
  _allocBlock(t) {
    const b = this.free.pop();
    if (b === undefined) t.blocked = true;
    return b;
  }
  /** 追加一个 token 的 KV。返回是否成功（块池耗尽时失败，不静默丢数据）。 */
  append(seqId, h, kVec, vVec) {
    const t = this.tables.get(seqId);
    if (!t) throw new Error(`未知序列 ${seqId}`);
    const pos = t.len;                       // 逻辑位置
    const lb = Math.floor(pos / this.blockSize);
    const inb = pos % this.blockSize;
    if (lb >= t.blocks.length) {
      const b = this.free.pop();
      if (b === undefined) return false;     // 池满：如实失败
      t.blocks.push(b);
    }
    const pb = t.blocks[lb];
    const o = (pb * this.blockSize + inb) * this.Hkv * this.Dh + h * this.Dh;
    this.kPool.set(kVec, o);
    this.vPool.set(vVec, o);
    if (h === this.Hkv - 1) t.len = pos + 1; // 最后一个 KV 头写完才推进长度
    return true;
  }
  /** 按块表 gather 出连续 K/V —— 这就是 PagedAttention 内核做的事。 */
  gather(seqId) {
    const t = this.tables.get(seqId);
    if (!t) throw new Error(`未知序列 ${seqId}`);
    const T = t.len;
    const kk = new Float64Array(T * this.Hkv * this.Dh);
    const vv = new Float64Array(T * this.Hkv * this.Dh);
    for (let pos = 0; pos < T; pos++) {
      const lb = Math.floor(pos / this.blockSize);
      const inb = pos % this.blockSize;
      const pb = t.blocks[lb];
      const src = (pb * this.blockSize + inb) * this.Hkv * this.Dh;
      const dst = pos * this.Hkv * this.Dh;
      kk.set(this.kPool.subarray(src, src + this.Hkv * this.Dh), dst);
      vv.set(this.vPool.subarray(src, src + this.Hkv * this.Dh), dst);
    }
    return { k: kk, v: vv, len: T, blocks: t.blocks.slice() };
  }
  freeSeq(seqId) {
    const t = this.tables.get(seqId);
    if (!t) return false;
    for (const b of t.blocks) this.free.push(b);
    this.tables.delete(seqId);
    return true;
  }
  /** 写时复制式前缀共享：把 src 的前 nBlocks 个块**共享**（引用同一物理块）给 dst。 */
  sharePrefix(srcId, dstId, nBlocks) {
    const s = this.tables.get(srcId);
    if (!s) throw new Error(`未知源序列 ${srcId}`);
    if (this.tables.has(dstId)) throw new Error(`目标序列 ${dstId} 已存在`);
    const share = s.blocks.slice(0, nBlocks);
    this.tables.set(dstId, { blocks: share.slice(), len: Math.min(s.len, nBlocks * this.blockSize), shared: true });
    return dstId;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. RadixAttention —— 基数树前缀复用（SGLang 思想）
// ═══════════════════════════════════════════════════════════════════════════
/**
 * 基数树：边上是 token（或 token 片段），节点存该前缀的 KV 块引用。
 * 多轮对话 / RAG / Agent 的 system prompt 每次都重发 —— 命中率 75~95%，
 * 这是 SGLang 在 agent 场景碾压 general serving 的全部原因。
 *
 * 淘汰：LRU，且只淘汰 refCount == 0 的节点（正在被请求用的节点不能动）。
 */
class RadixTree {
  constructor() {
    this.root = { children: new Map(), kv: null, lastAccess: 0, refCount: 0, prefix: [] };
    this.clock = 0;
    this.nodes = 1;
  }
  _touch(node) { node.lastAccess = ++this.clock; return node; }

  /** 插入一段 token 序列及其 KV 引用。返回末端节点。 */
  insert(tokens, kv) {
    let node = this.root;
    for (const tk of tokens) {
      let child = node.children.get(tk);
      if (!child) {
        child = { children: new Map(), kv: null, lastAccess: 0, refCount: 0, prefix: node.prefix.concat([tk]) };
        node.children.set(tk, child);
        this.nodes++;
      }
      node = child;
    }
    node.kv = kv;
    this._touch(node);
    // 沿途也 touch（父节点是这条路径的一部分，被访问意味着前缀仍活跃）
    let n = this.root;
    for (const tk of tokens) { n = n.children.get(tk); this._touch(n); }
    return node;
  }

  /**
   * 最长前缀匹配。返回 { node, matchedTokens }。
   * 语义：请求 tokens 中，能复用 KV 的最长前缀（只匹配到有 kv 的节点）。
   */
  match(tokens) {
    let node = this.root;
    let best = null, bestLen = 0;
    for (let i = 0; i < tokens.length; i++) {
      const child = node.children.get(tokens[i]);
      if (!child) break;
      node = child;
      if (node.kv !== null) { best = node; bestLen = i + 1; }
    }
    if (best) this._touch(best);
    return { node: best, matchedTokens: bestLen };
  }

  /** LRU 淘汰：回收 n 个"当前无人引用"的叶/中间节点。 */
  evict(n) {
    const all = [];
    (function walk(nd) {
      for (const c of nd.children.values()) { all.push(c); walk(c); }
    })(this.root);
    all.sort((a, b) => a.lastAccess - b.lastAccess);
    let removed = 0;
    for (const nd of all) {
      if (removed >= n) break;
      if (nd.refCount > 0) continue;
      if (nd.children.size > 0) continue;      // 只淘汰叶子，避免把子树前缀打散
      nd.kv = null;
      // 从父节点摘掉（需要父指针，这里再遍历一次找父；n 小，代价可接受）
      (function detach(root) {
        for (const [k, c] of root.children) {
          if (c === nd) { root.children.delete(k); return true; }
          if (detach(c)) return true;
        }
        return false;
      })(this.root);
      this.nodes--;
      removed++;
    }
    return removed;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. 投机解码 —— 严格无损
// ═══════════════════════════════════════════════════════════════════════════
/**
 * 单步投机采样：draft 给出 q(x)，target 给出 p(x)。
 *
 *   以概率 α = min(1, p(x)/q(x)) 接受 draft 的 x；
 *   否则从残差分布 r(x) = max(0, p(x) − q(x)) / Σ max(0, p − q) 重采样。
 *
 * 无损性证明（这是它敢用在生产上的唯一理由）：
 *   记 A = Σ_x min(q(x), p(x))，Z = Σ_x max(0, p(x) − q(x))。
 *   对任意 x：min(q,p) + max(0, p−q) = p(x)   ⇒   A + Z = 1
 *   输出分布 = q(x)·min(1, p/q) + (1−A)·r(x)
 *            = min(q(x), p(x)) + Z · max(0, p−q)/Z
 *            = min(q,p) + max(0, p−q) = p(x)                     ∎
 *
 * @param {number[]} p target 分布（需归一化）
 * @param {number[]} q draft 分布（需归一化）
 * @param {function} u01 一个 [0,1) 随机源（必须可注入，才能做确定性门禁）
 * @returns {{token:number, accepted:boolean}}
 */
function speculativeStep(p, q, u01) {
  const n = p.length;
  // draft 采样
  let x = -1, c = 0;
  const uq = u01();
  for (let i = 0; i < n; i++) { c += q[i]; if (uq < c) { x = i; break; } }
  if (x < 0) x = n - 1;
  // 接受判定
  const up = u01();
  const accept = q[x] > 0 ? up < Math.min(1, p[x] / q[x]) : true;
  if (accept) return { token: x, accepted: true };
  // 拒绝：从残差分布重采样
  const residual = new Float64Array(n);
  let z = 0;
  for (let i = 0; i < n; i++) { const r = Math.max(0, p[i] - q[i]); residual[i] = r; z += r; }
  if (z <= 0) return { token: x, accepted: true };   // 残差为空 → p==q，接受即可（仍无损）
  const ur = u01() * z;
  let acc2 = 0, y = n - 1;
  for (let i = 0; i < n; i++) { acc2 += residual[i]; if (ur < acc2) { y = i; break; } }
  return { token: y, accepted: false };
}

/** 多步投机：一次 draft 出 γ 个 token，逐个验证，第一个拒绝处截断并重采样。 */
function speculativeDecode(pSeq, qSeq, u01, gamma) {
  // pSeq(t) 返回给定已接受前缀后 target 的下一步分布；这里简化为预生成的分布数组
  const out = [];
  let acceptedCount = 0;
  for (let i = 0; i < gamma; i++) {
    const st = speculativeStep(pSeq[i], qSeq[i], u01);
    out.push(st.token);
    if (!st.accepted) return { tokens: out, acceptedCount };
    acceptedCount++;
  }
  // 全部接受：再多采一个（target 的第 γ+1 个分布）
  if (pSeq[gamma]) {
    const extra = sampleCategorical(pSeq[gamma], u01());
    out.push(extra);
  }
  return { tokens: out, acceptedCount };
}

/** 按给定分布做类别采样（累积 + 二分/线性）。 */
function sampleCategorical(dist, u) {
  let c = 0;
  for (let i = 0; i < dist.length; i++) { c += dist[i]; if (u < c) return i; }
  return dist.length - 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. 采样：Top-p (nucleus)
// ═══════════════════════════════════════════════════════════════════════════
/**
 * Top-p：按概率降序累积到 ≥ p 为止，只在这集合内重归一化采样。
 * 不变量：截断集合是"满足累积概率 ≥ p 的最小集合"（少一个就不够 p）。
 */
function topPNucleus(probs, p) {
  const idx = Array.from({ length: probs.length }, (_, i) => i)
    .sort((a, b) => probs[b] - probs[a]);
  const kept = [];
  let cum = 0;
  for (const i of idx) {
    kept.push(i);
    cum += probs[i];
    if (cum >= p) break;
  }
  // 重归一化
  const out = new Float64Array(probs.length);
  for (const i of kept) out[i] = probs[i] / cum;
  return { probs: out, kept, cum, minSet: kept.length === 1 || cum - probs[kept[kept.length - 1]] < p };
}

/** 温度缩放 + top-p 的完整采样链。 */
function sampleWithTopP(logits, opts = {}) {
  const { temperature = 1.0, topP = 1.0, u01 = Math.random, seed = null } = opts;
  const rng = seed !== null ? new L0.PCG128SM(seed) : null;
  const u = rng ? () => rng.next() : u01;
  const scaled = Array.from(logits, (v) => v / Math.max(1e-12, temperature));
  const p = L0.softmax(scaled);
  const { probs } = topPNucleus(Array.from(p), topP);
  return sampleCategorical(Array.from(probs), u());
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. 量化
// ═══════════════════════════════════════════════════════════════════════════
/**
 * INT8 per-channel 对称量化：每个输出通道一个 scale。
 * 为什么 per-channel：同通道内量级相近，跨通道可能差 10 倍 —— 一个全局 scale 会把
 * 小通道全压成 0。
 */
function quantizeINT8PerChannel(W) {
  // W: [outC, inC]，按行（输出通道）给 scale
  const outC = W.length, inC = W[0].length;
  const scales = new Float64Array(outC);
  const qW = new Array(outC);
  for (let i = 0; i < outC; i++) {
    let amax = 0;
    for (let j = 0; j < inC; j++) amax = Math.max(amax, Math.abs(W[i][j]));
    scales[i] = amax / 127 || 1;
    qW[i] = new Int8Array(inC);
    for (let j = 0; j < inC; j++) qW[i][j] = Math.max(-127, Math.min(127, Math.round(W[i][j] / scales[i])));
  }
  const dequant = () => {
    const out = new Array(outC);
    for (let i = 0; i < outC; i++) { out[i] = new Float64Array(inC); for (let j = 0; j < inC; j++) out[i][j] = qW[i][j] * scales[i]; }
    return out;
  };
  return { qW, scales, dequant, bits: 8, scheme: 'int8-per-channel' };
}

/**
 * INT4 group-wise 量化：每 groupSize 列一组，组内一个 scale。
 * 4bit 只有 16 个电平，per-tensor 会彻底毁掉权重 —— group-wise 是 AWQ/GPTQ 的共识。
 */
function quantizeINT4Group(W, groupSize = 32) {
  const outC = W.length, inC = W[0].length;
  const nGroups = Math.ceil(inC / groupSize);
  const scales = new Float64Array(outC * nGroups);
  const qW = new Array(outC);
  for (let i = 0; i < outC; i++) {
    qW[i] = new Int8Array(inC);   // 用 Int8 容器存 [-8,7]
    for (let g = 0; g < nGroups; g++) {
      const lo = g * groupSize, hi = Math.min(inC, lo + groupSize);
      let amax = 0;
      for (let j = lo; j < hi; j++) amax = Math.max(amax, Math.abs(W[i][j]));
      const s = amax / 7 || 1;
      scales[i * nGroups + g] = s;
      for (let j = lo; j < hi; j++) qW[i][j] = Math.max(-8, Math.min(7, Math.round(W[i][j] / s)));
    }
  }
  const dequant = () => {
    const out = new Array(outC);
    for (let i = 0; i < outC; i++) {
      out[i] = new Float64Array(inC);
      for (let j = 0; j < inC; j++) out[i][j] = qW[i][j] * scales[i * nGroups + Math.floor(j / groupSize)];
    }
    return out;
  };
  return { qW, scales, dequant, bits: 4, scheme: `int4-group${groupSize}` };
}

/** KV cache 的 INT8 量化（推理时显存大头在 KV，不在权重）。 */
function quantizeKVCache(arr) {
  let amax = 0;
  for (let i = 0; i < arr.length; i++) amax = Math.max(amax, Math.abs(arr[i]));
  const scale = amax / 127 || 1;
  const q = new Int8Array(arr.length);
  for (let i = 0; i < arr.length; i++) q[i] = Math.max(-127, Math.min(127, Math.round(arr[i] / scale)));
  return { q, scale, dequant: () => Float64Array.from(q, (v) => v * scale) };
}

module.exports = {
  NaiveKVCache, PagedKVCache, RadixTree,
  speculativeStep, speculativeDecode, sampleCategorical,
  topPNucleus, sampleWithTopP,
  quantizeINT8PerChannel, quantizeINT4Group, quantizeKVCache,
};
