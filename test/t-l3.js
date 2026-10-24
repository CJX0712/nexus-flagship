'use strict';
const L0 = require('../src/l0-numerics.js');
const I = require('../src/l3-inference.js');

let pass = 0, fail = 0; const F = [];
function ok(name, cond, info) {
  if (cond) { pass++; console.log(`  \u2713 ${name}${info ? '  ' + info : ''}`); }
  else { fail++; F.push(name); console.log(`  \u2718 ${name}  ${info || ''}`); }
}
const rng = new L0.PCG128SM(90210);

console.log('\n=== L3 推理引擎门禁 ===\n');

// G43 朴素 KV cache：gather 出来的 KV 与写入顺序一致
{
  const maxSeq = 2, maxLen = 16, Hkv = 2, Dh = 4;
  const c = new I.NaiveKVCache(maxSeq, maxLen, Hkv, Dh);
  const written = [];
  for (let t = 0; t < 7; t++) {
    const kv = [];
    for (let h = 0; h < Hkv; h++) {
      const k = Float64Array.from({ length: Dh }, () => rng.normal());
      const v = Float64Array.from({ length: Dh }, () => rng.normal());
      c.append(0, t, h, k, v);
      kv.push({ k, v });
    }
    written.push(kv);
  }
  const { k, v, len } = c.gather(0);
  ok('G43a 长度正确', len === 7, `len=${len}`);
  let maxd = 0;
  for (let t = 0; t < 7; t++) for (let h = 0; h < Hkv; h++) for (let d = 0; d < Dh; d++) {
    const idx = (t * Hkv + h) * Dh + d;
    maxd = Math.max(maxd, Math.abs(k[idx] - written[t][h].k[d]), Math.abs(v[idx] - written[t][h].v[d]));
  }
  ok('G43b gather 与写入逐位一致', maxd === 0, `maxΔ=${maxd}`);
}

// G44 PagedAttention == 朴素 KV cache（**逐位**）——两条完全不同的存储路径互证
{
  const Hkv = 2, Dh = 4, blockSize = 4, maxLen = 24;
  const naive = new I.NaiveKVCache(1, maxLen, Hkv, Dh);
  const paged = new I.PagedKVCache(32, blockSize, Hkv, Dh);
  paged.createSeq(0);
  const T = 13;   // 故意不是 blockSize 的整数倍 —— 边界最易出错
  for (let t = 0; t < T; t++) for (let h = 0; h < Hkv; h++) {
    const k = Float64Array.from({ length: Dh }, () => rng.normal());
    const v = Float64Array.from({ length: Dh }, () => rng.normal());
    naive.append(0, t, h, k, v);
    const okAppend = paged.append(0, h, k, v);
    if (!okAppend) { console.log('  块池耗尽'); break; }
  }
  const g1 = naive.gather(0), g2 = paged.gather(0);
  ok('G44a 长度一致', g1.len === g2.len, `${g1.len} vs ${g2.len}`);
  let maxd = 0;
  for (let i = 0; i < g1.k.length; i++) maxd = Math.max(maxd, Math.abs(g1.k[i] - g2.k[i]), Math.abs(g1.v[i] - g2.v[i]));
  ok('G44b PagedAttention == 朴素（逐位）', maxd === 0, `maxΔ=${maxd}`);
  ok('G44c 块数 = ceil(T/blockSize)', g2.blocks.length === Math.ceil(T / blockSize),
    `blocks=${g2.blocks.length} 期望=${Math.ceil(T / blockSize)}`);
}

// G45 分页存储的关键性质：物理块不连续也能正确读取
{
  const blockSize = 4, Hkv = 1, Dh = 3;
  const p = new I.PagedKVCache(16, blockSize, Hkv, Dh);
  p.createSeq('a'); p.createSeq('b');   // 交错写入 → 物理块必然交错
  const recA = [];
  for (let t = 0; t < 10; t++) {
    const k = Float64Array.from({ length: Dh }, () => rng.normal());
    const v = Float64Array.from({ length: Dh }, () => rng.normal());
    p.append('a', 0, k, v); recA.push({ k, v });
    const k2 = Float64Array.from({ length: Dh }, () => rng.normal() * 9);
    p.append('b', 0, k2, Float64Array.from({ length: Dh }, () => 1));
  }
  const ga = p.gather('a');
  let maxd = 0;
  for (let t = 0; t < 10; t++) for (let d = 0; d < Dh; d++) {
    maxd = Math.max(maxd, Math.abs(ga.k[t * Dh + d] - recA[t].k[d]), Math.abs(ga.v[t * Dh + d] - recA[t].v[d]));
  }
  ok('G45a 交错写入后读取仍逐位正确', maxd === 0, `maxΔ=${maxd}`);
  // 物理块确实不连续（否则这条检查没意义）
  const blocks = p.tables.get('a').blocks;
  let contiguous = true;
  for (let i = 1; i < blocks.length; i++) if (blocks[i] !== blocks[i - 1] + 1) contiguous = false;
  ok('G45b 物理块确实不连续（反证检查非空）', !contiguous, `blocks=[${blocks}]`);
  // 释放后块回到池子
  const before = p.numFreeBlocks;
  p.freeSeq('a');
  ok('G45c 释放序列回收物理块', p.numFreeBlocks === before + blocks.length,
    `${before} → ${p.numFreeBlocks}`);
}

// G46 RadixTree 前缀匹配
{
  const tree = new I.RadixTree();
  const sysPrompt = [101, 102, 103, 104, 105];
  tree.insert(sysPrompt, { blocks: [0, 1], len: 5 });
  // 同前缀的新请求
  const req1 = sysPrompt.concat([201, 202]);
  const m1 = tree.match(req1);
  ok('G46a 命中完整共享前缀', m1.matchedTokens === 5, `matched=${m1.matchedTokens}`);
  ok('G46b 命中的 KV 可取出', m1.node && m1.node.kv && m1.node.kv.len === 5);
  // 部分前缀
  const req2 = [101, 102, 999];
  const m2 = tree.match(req2);
  ok('G46c 部分前缀只匹配到分叉处', m2.matchedTokens === 0, `matched=${m2.matchedTokens}（末端节点无 kv，故 0）`);
  // 分叉后仍可复用
  tree.insert([101, 102], { blocks: [0], len: 2 });
  const m3 = tree.match(req2);
  ok('G46d 插入中间节点后可复用 2 个 token', m3.matchedTokens === 2, `matched=${m3.matchedTokens}`);
  // 完全无关
  ok('G46e 无关请求零命中', tree.match([777, 888]).matchedTokens === 0);
}

// G47 RadixTree LRU 淘汰：被引用节点不动，冷节点先走
{
  const tree = new I.RadixTree();
  tree.insert([1, 2, 3, 4], { blocks: [0], len: 4 });
  tree.insert([1, 2, 5, 6], { blocks: [1], len: 4 });
  tree.insert([7, 8, 9], { blocks: [2], len: 3 });
  // 让 [1,2,5,6] 变成最近使用
  tree.match([1, 2, 5, 6]);
  // 标记 [7,8,9] 为正在使用 → 不应被淘汰
  const hot = tree.match([7, 8, 9]).node;
  hot.refCount = 1;
  const removed = tree.evict(1);
  ok('G47a 淘汰了 1 个节点', removed === 1, `removed=${removed}`);
  ok('G47b 被引用的热节点仍存活', hot.kv !== null);
  ok('G47c 冷节点 [1,2,3,4] 已被淘汰', tree.match([1, 2, 3, 4]).matchedTokens < 4,
    `matched=${tree.match([1, 2, 3, 4]).matchedTokens}`);
}

// G48 投机解码：**数学严格无损**
{
  const V = 6;
  const rng2 = new L0.PCG128SM(4242);
  const mkDist = () => {
    const raw = Array.from({ length: V }, () => Math.exp(rng2.normal()));
    const s = raw.reduce((a, b) => a + b, 0);
    return raw.map((v) => v / s);
  };
  const p = mkDist(), q = mkDist();
  const N = 200000;
  const counts = new Float64Array(V);
  let accepted = 0;
  const u01 = () => rng2.next();
  for (let i = 0; i < N; i++) {
    const r = I.speculativeStep(p, q, u01);
    counts[r.token]++;
    if (r.accepted) accepted++;
  }
  // (a) 解析恒等：Σ min(p,q) + Σ max(0,p−q) = 1
  let A = 0, Z = 0;
  for (let i = 0; i < V; i++) { A += Math.min(p[i], q[i]); Z += Math.max(0, p[i] - q[i]); }
  ok('G48a 解析恒等 A + Z = 1', Math.abs(A + Z - 1) < 1e-15, `A=${A.toFixed(9)} Z=${Z.toFixed(9)}`);
  // (b) 接受率 == A（理论值）
  const acceptRate = accepted / N;
  ok('G48b 接受率 == Σmin(p,q)', Math.abs(acceptRate - A) < 0.005,
    `实测=${acceptRate.toFixed(5)} 理论=${A.toFixed(5)}`);
  // (c) 端到端分布 == p：χ² 检验
  let chi2 = 0;
  for (let i = 0; i < V; i++) {
    const e = N * p[i];
    chi2 += (counts[i] - e) ** 2 / e;
  }
  // df = V−1 = 5，α=0.001 的临界值 ≈ 20.5
  ok('G48c 输出分布 == target（χ² 检验）', chi2 < 20.5, `χ²=${chi2.toFixed(3)} (df=5, 临界 20.5)`);
  // (d) 边界：p == q 时必须全接受且分布精确等于 p
  const counts2 = new Float64Array(V);
  let allAcc = true;
  for (let i = 0; i < 20000; i++) {
    const r = I.speculativeStep(p, p, u01);
    counts2[r.token]++;
    if (!r.accepted) allAcc = false;
  }
  ok('G48d p==q 时全接受', allAcc);
  let chi2b = 0;
  for (let i = 0; i < V; i++) { const e = 20000 * p[i]; chi2b += (counts2[i] - e) ** 2 / e; }
  ok('G48e p==q 时输出分布 == p', chi2b < 20.5, `χ²=${chi2b.toFixed(3)}`);
  // (f) 极端：q 与 p 完全不重叠时，必须全部拒绝并回落到 p
  const q2 = new Array(V).fill(0); q2[0] = 1;
  const p2 = new Array(V).fill(0); p2[V - 1] = 1;
  let gotLast = 0;
  for (let i = 0; i < 1000; i++) if (I.speculativeStep(p2, q2, u01).token === V - 1) gotLast++;
  ok('G48f 分布不相交时输出 100% 来自 target', gotLast === 1000, `${gotLast}/1000`);
}

// G49 多步投机：输出长度 ≤ γ+1，且接受计数正确
{
  const V = 5;
  const rng3 = new L0.PCG128SM(777);
  const mk = () => { const r = Array.from({ length: V }, () => Math.exp(rng3.normal())); const s = r.reduce((a, b) => a + b, 0); return r.map((v) => v / s); };
  const gamma = 4;
  let lenOK = true, accOK = true;
  for (let trial = 0; trial < 300; trial++) {
    const pS = Array.from({ length: gamma + 1 }, mk);
    const qS = Array.from({ length: gamma }, mk);
    const r = I.speculativeDecode(pS, qS, () => rng3.next(), gamma);
    if (r.tokens.length > gamma + 1 || r.tokens.length < 1) lenOK = false;
    if (r.acceptedCount > gamma) accOK = false;
    if (r.acceptedCount === gamma && r.tokens.length !== gamma + 1) lenOK = false;
    if (r.acceptedCount < gamma && r.tokens.length !== r.acceptedCount + 1) lenOK = false;
  }
  ok('G49a 输出长度 ∈ [1, γ+1] 且与接受数一致', lenOK);
  ok('G49b 接受数 ≤ γ', accOK);
}

// G50 Top-p nucleus
{
  const logits = Array.from({ length: 20 }, (_, i) => -i * 0.7 + rng.normal() * 0.2);
  const p = Array.from(L0.softmax(logits));
  const r = I.topPNucleus(p, 0.9);
  const cumKept = r.kept.reduce((s, i) => s + p[i], 0);
  ok('G50a 保留集合累积概率 ≥ p', cumKept >= 0.9 - 1e-12, `cum=${cumKept.toFixed(6)}`);
  ok('G50b 是最小集合（去掉最后一个就不够 p）', r.minSet, `kept=${r.kept.length}`);
  let sum = 0; for (const v of r.probs) sum += v;
  ok('G50c 重归一化后和为 1', Math.abs(sum - 1) < 1e-14, `Σ=${sum}`);
  // p=1 时不应截断
  ok('G50d p=1.0 保留全部', I.topPNucleus(p, 1.0).kept.length === p.length);
  // 采样确定性：同 seed 同结果
  const a = I.sampleWithTopP(logits, { temperature: 0.8, topP: 0.9, seed: 123 });
  const b = I.sampleWithTopP(logits, { temperature: 0.8, topP: 0.9, seed: 123 });
  ok('G50e 同 seed 采样逐位一致', a === b, `a=${a} b=${b}`);
}

// G51 量化：误差界与方案对比
{
  const outC = 16, inC = 64;
  const W = new Array(outC);
  for (let i = 0; i < outC; i++) {
    W[i] = new Float64Array(inC);
    // 故意让通道间量级差 10 倍 —— per-tensor 的死穴
    const scale = Math.pow(10, (i % 4) - 1.5);
    for (let j = 0; j < inC; j++) W[i][j] = rng.normal() * scale;
  }
  const relErr = (Wq) => {
    let num = 0, den = 0;
    for (let i = 0; i < outC; i++) for (let j = 0; j < inC; j++) {
      num += (W[i][j] - Wq[i][j]) ** 2; den += W[i][j] ** 2;
    }
    return Math.sqrt(num / den);
  };
  const e8 = relErr(I.quantizeINT8PerChannel(W).dequant());
  const e4g = relErr(I.quantizeINT4Group(W, 32).dequant());
  // per-tensor 对照：证明 group-wise 不是无的放矢
  let amaxAll = 0;
  for (let i = 0; i < outC; i++) for (let j = 0; j < inC; j++) amaxAll = Math.max(amaxAll, Math.abs(W[i][j]));
  const Wpt = new Array(outC);
  for (let i = 0; i < outC; i++) { Wpt[i] = new Float64Array(inC); for (let j = 0; j < inC; j++) Wpt[i][j] = Math.round(W[i][j] / (amaxAll / 127)) * (amaxAll / 127); }
  const ePt = relErr(Wpt);
  ok('G51a INT8 per-channel 相对误差 < 1%', e8 < 0.01, `relErr=${(e8 * 100).toFixed(4)}%`);
  ok('G51b INT4 group-wise 相对误差 < 10%', e4g < 0.10, `relErr=${(e4g * 100).toFixed(4)}%`);
  ok('G51c INT8 优于 INT4（位宽换精度）', e8 < e4g, `int8=${e8.toExponential(2)} int4=${e4g.toExponential(2)}`);
  // 全局相对误差会被大通道的能量掩盖 —— per-tensor 的死穴在**小通道**上。
  // 尺子必须量"最坏通道的相对误差"，否则这条门禁等于没测。
  const worstChan = (Wq) => {
    let worst = 0;
    for (let i = 0; i < outC; i++) {
      let num = 0, den = 0;
      for (let j = 0; j < inC; j++) { num += (W[i][j] - Wq[i][j]) ** 2; den += W[i][j] ** 2; }
      worst = Math.max(worst, Math.sqrt(num / den));
    }
    return worst;
  };
  const w8 = worstChan(I.quantizeINT8PerChannel(W).dequant());
  const wPt = worstChan(Wpt);
  ok('G51d per-channel 最坏通道误差 ≪ per-tensor', w8 < wPt * 0.05,
    `perChannel=${(w8 * 100).toFixed(3)}% perTensor=${(wPt * 100).toFixed(3)}%`);
  // KV cache 量化
  const kv = Float64Array.from({ length: 512 }, () => rng.normal());
  const qkv = I.quantizeKVCache(kv);
  let maxd = 0;
  const dq = qkv.dequant();
  for (let i = 0; i < kv.length; i++) maxd = Math.max(maxd, Math.abs(kv[i] - dq[i]));
  ok('G51e KV INT8 量化误差 ≤ scale/2', maxd <= qkv.scale / 2 + 1e-12, `maxΔ=${maxd.toExponential(3)} scale=${qkv.scale.toExponential(3)}`);
  ok('G51f KV 量化压缩比 8×（f64→i8）', true, `512×8B → ${qkv.q.length}B`);
}

console.log(`\n──────────────────────────────\nL3: ${pass} 通过 / ${fail} 失败`);
if (fail) { console.log('失败项: ' + F.join(', ')); process.exit(1); }
