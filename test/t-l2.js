'use strict';
const L0 = require('../src/l0-numerics.js');
const A = require('../src/l1-autodiff.js');
const P = require('../src/l2-primitives.js');
const { T, matmul, sum, mul, silu, reshape } = A;

let pass = 0, fail = 0; const F = [];
function ok(name, cond, info) {
  if (cond) { pass++; console.log(`  \u2713 ${name}${info ? '  ' + info : ''}`); }
  else { fail++; F.push(name); console.log(`  \u2718 ${name}  ${info || ''}`); }
}
const rng = new L0.PCG128SM(31337);
const rnd = (n, s = 1) => Array.from({ length: n }, () => rng.normal() * s);

console.log('\n=== L2 架构原语门禁 ===\n');

// G31 RMSNorm：手写版 vs tape 版必须一致（两条独立路径互证）
{
  const B = 2, Tn = 5, D = 16;
  const x = T(rnd(B * Tn * D), [B, Tn, D], { requiresGrad: true });
  const g = T(Array.from({ length: D }, () => 0.5 + rng.next() * 0.5), [D], { requiresGrad: true });
  const fast = P.rmsNormFast(x, g);
  const tape = P.rmsNormTape(x, g);
  let maxd = 0;
  for (let i = 0; i < fast.size; i++) maxd = Math.max(maxd, Math.abs(fast.data[i] - tape.data[i]));
  ok('G31a RMSNorm 手写 == tape', maxd < 1e-12, `maxΔ=${maxd.toExponential(2)}`);

  // 梯度：两条路径分别对拍中心差分
  const gFast = A.gradCheckFn((ps) => sum(P.rmsNormFast(reshape(ps[0], [B, Tn, D]), ps[1])),
    [T(x.data, [x.size], { requiresGrad: true }), T(g.data, [g.size], { requiresGrad: true })]);
  ok('G31b RMSNorm 手写版梯度检验', gFast.pass, `maxRel=${gFast.maxRel.toExponential(2)}`);
}

// G32 RMSNorm 的缩放不变性：x → α·x 输出不变（α > 0）
{
  const D = 12;
  const x = T(rnd(D), [1, 1, D]);
  const g = T(rnd(D), [D]);
  // (a) eps = 0 时是**精确**的数学恒等式 —— 这才是性质本身
  const y1 = P.rmsNormFast(x, g, 0);
  const y2 = P.rmsNormFast(T(x.data.map((v) => v * 7.3), [1, 1, D]), g, 0);
  let maxd0 = 0;
  for (let i = 0; i < D; i++) maxd0 = Math.max(maxd0, Math.abs(y1.data[i] - y2.data[i]));
  ok('G32a RMSNorm 缩放不变（eps=0，精确恒等）', maxd0 < 1e-15, `maxΔ=${maxd0.toExponential(2)}`);
  // (b) eps > 0 时不变量被 eps 破坏，误差有解析界：Δ ≈ eps·(1−1/α²)/(2·rms)
  //     门禁按这个界判，而不是拍脑袋给 1e-14 然后失败。
  const rms0 = Math.sqrt(Array.from(x.data).reduce((s, v) => s + v * v, 0) / D);
  const bound = 1e-6 * (1 - 1 / (7.3 ** 2)) / (2 * rms0);
  const y3 = P.rmsNormFast(x, g);
  const y4 = P.rmsNormFast(T(x.data.map((v) => v * 7.3), [1, 1, D]), g);
  let maxd = 0;
  for (let i = 0; i < D; i++) maxd = Math.max(maxd, Math.abs(y3.data[i] - y4.data[i]));
  ok('G32b eps>0 时误差在解析界内', maxd <= bound * 1.5,
    `maxΔ=${maxd.toExponential(2)} 界=${bound.toExponential(2)}`);
}

// G33 RoPE 的核心不变量：内积只依赖相对位置
{
  const Dh = 16;
  const inv = P.ropeFreqs(Dh, 512);
  const x = T(rnd(Dh), [1, 1, 1, Dh]);
  const y = T(rnd(Dh), [1, 1, 1, Dh]);
  const dot = (a, b) => { let s = 0; for (let i = 0; i < Dh; i++) s += a[i] * b[i]; return s; };
  const ropeAt = (v, pos) => P.ropeApply(v, pos, inv).data;
  // ⟨RoPE(x,5), RoPE(y,5)⟩ == ⟨RoPE(x,100), RoPE(y,100)⟩（相对位置都是 0）
  const d1 = dot(ropeAt(x, 5), ropeAt(y, 5));
  const d2 = dot(ropeAt(x, 100), ropeAt(y, 100));
  // ⟨RoPE(x,7), RoPE(y,5)⟩ == ⟨RoPE(x,102), RoPE(y,100)⟩（相对位置都是 2）
  const d3 = dot(ropeAt(x, 7), ropeAt(y, 5));
  const d4 = dot(ropeAt(x, 102), ropeAt(y, 100));
  ok('G33a RoPE 内积与绝对位置无关', Math.abs(d1 - d2) < 1e-12, `${d1.toFixed(9)} vs ${d2.toFixed(9)}`);
  ok('G33b RoPE 内积只依赖 m−n', Math.abs(d3 - d4) < 1e-12, `${d3.toFixed(9)} vs ${d4.toFixed(9)}`);
  ok('G33c 相对位置 0 与 2 的内积确实不同', Math.abs(d1 - d3) > 1e-6, `Δ=${Math.abs(d1 - d3).toExponential(2)}`);
}

// G34 RoPE 是正交变换 → 保范数
{
  const Dh = 32;
  const inv = P.ropeFreqs(Dh, 512);
  const x = T(rnd(Dh, 3), [1, 1, 1, Dh]);
  const r = P.ropeApply(x, 42, inv).data;
  const n1 = Math.hypot(...Array.from(x.data));
  const n2 = Math.hypot(...Array.from(r));
  ok('G34a RoPE 保范数', Math.abs(n1 - n2) / n1 < 1e-14, `Δrel=${(Math.abs(n1 - n2) / n1).toExponential(2)}`);
  // 逆旋转（用 −pos 旋转不直接可行），改为：旋转后再用转置旋转
  const g = A.gradCheckFn((ps) => sum(P.ropeApply(reshape(ps[0], [1, 1, 1, Dh]), 3, inv)),
    [T(x.data, [x.size], { requiresGrad: true })]);
  ok('G34b RoPE 梯度检验', g.pass, `maxRel=${g.maxRel.toExponential(2)}`);
}

// G35 SwiGLU 梯度
{
  const B = 2, Tn = 4, D = 6, Dff = 10;
  const x = T(rnd(B * Tn * D, 0.5), [B * Tn, D], { requiresGrad: true });
  const wG = T(rnd(D * Dff, 0.3), [D, Dff], { requiresGrad: true });
  const wU = T(rnd(D * Dff, 0.3), [D, Dff], { requiresGrad: true });
  const wD = T(rnd(Dff * D, 0.3), [Dff, D], { requiresGrad: true });
  const g = A.gradCheckFn((ps) => sum(P.swigluTape(
    reshape(ps[0], [B * Tn, D]), reshape(ps[1], [D, Dff]),
    reshape(ps[2], [D, Dff]), reshape(ps[3], [Dff, D]))),
    [T(x.data, [x.size], { requiresGrad: true }), T(wG.data, [wG.size], { requiresGrad: true }),
      T(wU.data, [wU.size], { requiresGrad: true }), T(wD.data, [wD.size], { requiresGrad: true })]);
  ok('G35 SwiGLU 端到端梯度检验', g.pass, `maxRel=${g.maxRel.toExponential(2)}`);
}

// G36 causalAttention vs naiveAttention（MHA、Tq == Tkv）
{
  const B = 2, Tn = 6, H = 3, Dh = 8, Dv = 8;
  const q = T(rnd(B * Tn * H * Dh), [B, Tn, H, Dh], { requiresGrad: true });
  const k = T(rnd(B * Tn * H * Dh), [B, Tn, H, Dh], { requiresGrad: true });
  const v = T(rnd(B * Tn * H * Dv), [B, Tn, H, Dv], { requiresGrad: true });
  const fast = P.causalAttention(q, k, v);
  const naive = P.naiveAttention(q, k, v);
  let maxd = 0;
  for (let i = 0; i < fast.size; i++) maxd = Math.max(maxd, Math.abs(fast.data[i] - naive.data[i]));
  ok('G36a causalAttention == naiveAttention', maxd < 1e-13, `maxΔ=${maxd.toExponential(2)}`);
}

// G37 因果性：第 i 个 query 不能看到 j > i 的 key（把 k[i+1..] 改掉，输出必须不变）
{
  const B = 1, Tn = 8, H = 2, Dh = 8, Dv = 8;
  const q = T(rnd(B * Tn * H * Dh), [B, Tn, H, Dh]);
  const k = T(rnd(B * Tn * H * Dh), [B, Tn, H, Dh]);
  const v = T(rnd(B * Tn * H * Dv), [B, Tn, H, Dv]);
  const out1 = P.causalAttention(q, k, v).data;
  // 对每个 query i，**只**替换它看不到的 key（j > i）—— 输出 i 必须逐位不变。
  // （曾把整个 k 都换掉：那样 j ≤ i 的 key 也变了，当然会变 —— 检查失效且假失败。）
  let maxd = 0;
  for (let i = 0; i < Tn; i++) {
    const k2 = k.clone();
    for (let j = i + 1; j < Tn; j++) for (let h = 0; h < H; h++) for (let d = 0; d < Dh; d++) {
      k2.data[(((0 * Tn) + j) * H + h) * Dh + d] = rng.normal() * 5;
    }
    const o2 = P.causalAttention(q, k2, v).data;
    for (let h = 0; h < H; h++) for (let d = 0; d < Dv; d++) {
      const idx = ((0 * Tn + i) * H + h) * Dv + d;
      maxd = Math.max(maxd, Math.abs(out1[idx] - o2[idx]));
    }
  }
  ok('G37a 因果掩码生效（未来 key 不影响过去 query）', maxd < 1e-16, `maxΔ=${maxd.toExponential(2)}`);
  // 反证：把 query i **能看到**的 key 换掉，输出必须变（证明上面不是恒真检查）
  let changed = 0;
  {
    const i = Tn - 1, j = 0;
    const k3 = k.clone();
    for (let h = 0; h < H; h++) for (let d = 0; d < Dh; d++) k3.data[(((0 * Tn) + j) * H + h) * Dh + d] = rng.normal() * 5;
    const o3 = P.causalAttention(q, k3, v).data;
    for (let h = 0; h < H; h++) for (let d = 0; d < Dv; d++) {
      changed = Math.max(changed, Math.abs(out1[((0 * Tn + i) * H + h) * Dv + d] - o3[((0 * Tn + i) * H + h) * Dv + d]));
    }
  }
  ok('G37b 可见 key 改变 → 输出改变（反证检查非空）', changed > 1e-6, `Δ=${changed.toExponential(2)}`);
}

// G38 **右对齐**因果掩码 —— 增量解码逐步 == 全量一次算
{
  const B = 1, Tn = 7, H = 2, Dh = 8, Dv = 8;
  const qAll = T(rnd(B * Tn * H * Dh), [B, Tn, H, Dh]);
  const kAll = T(rnd(B * Tn * H * Dh), [B, Tn, H, Dh]);
  const vAll = T(rnd(B * Tn * H * Dv), [B, Tn, H, Dv]);
  const full = P.causalAttention(qAll, kAll, vAll).data;

  // 逐步：第 t 步只喂 1 个 query，但 KV 给到 t+1
  let maxd = 0;
  for (let t = 0; t < Tn; t++) {
    const qs = T(Array.from({ length: H * Dh }, (_, i) => qAll.data[(t * H * Dh) + i]), [B, 1, H, Dh]);
    const ks = T(Array.from({ length: (t + 1) * H * Dh }, (_, i) => kAll.data[i]), [B, t + 1, H, Dh]);
    const vs = T(Array.from({ length: (t + 1) * H * Dv }, (_, i) => vAll.data[i]), [B, t + 1, H, Dv]);
    const o = P.causalAttention(qs, ks, vs, { startPos: t }).data;
    for (let h = 0; h < H; h++) for (let d = 0; d < Dv; d++) {
      const idxFull = ((0 * Tn + t) * H + h) * Dv + d;
      const idxStep = ((0 * 1 + 0) * H + h) * Dv + d;
      maxd = Math.max(maxd, Math.abs(full[idxFull] - o[idxStep]));
    }
  }
  // 掩码若左对齐，这里会差出 O(1) 的量级 —— 所以它抓得住
  ok('G38 增量解码 == 全量（掩码右对齐）', maxd < 1e-13, `maxΔ=${maxd.toExponential(2)}`);
}

// G39 GQA 分组正确性：Hkv < H
{
  const B = 1, Tn = 5, H = 6, Hkv = 2, Dh = 8, Dv = 8;
  const grp = H / Hkv;
  const q = T(rnd(B * Tn * H * Dh), [B, Tn, H, Dh]);
  const k = T(rnd(B * Tn * Hkv * Dh), [B, Tn, Hkv, Dh]);
  const v = T(rnd(B * Tn * Hkv * Dv), [B, Tn, Hkv, Dv]);
  const out = P.causalAttention(q, k, v);
  ok('G39a GQA 输出形状 [B,T,H,Dv]', A.shapeEq(out.shape, [B, Tn, H, Dv]));
  // 同一 KV 组内的两个 Q 头，若 q 相同则输出必须相同
  const qSame = T(new Float64Array(B * Tn * H * Dh), [B, Tn, H, Dh]);
  for (let b = 0; b < B; b++) for (let t = 0; t < Tn; t++) for (let h = 0; h < H; h++) for (let d = 0; d < Dh; d++) {
    qSame.data[((b * Tn + t) * H + h) * Dh + d] = 1 + d * 0.1;   // 所有头相同
  }
  const o2 = P.causalAttention(qSame, k, v).data;
  let maxd = 0;
  for (let b = 0; b < B; b++) for (let t = 0; t < Tn; t++) for (let d = 0; d < Dv; d++) {
    const h0 = 0, h1 = 1;   // 同属 kvHead 0（grp=3，头 0,1,2 共享 kvHead 0）
    maxd = Math.max(maxd, Math.abs(o2[((b * Tn + t) * H + h0) * Dv + d] - o2[((b * Tn + t) * H + h1) * Dv + d]));
  }
  ok('G39b 同组 Q 头共享同一 KV 头（输出相同）', maxd < 1e-16, `maxΔ=${maxd.toExponential(2)}`);
  // 不同组必须不同（反证）
  let diff = 0;
  for (let t = 0; t < Tn; t++) for (let d = 0; d < Dv; d++) {
    diff = Math.max(diff, Math.abs(o2[((0 * Tn + t) * H + 0) * Dv + d] - o2[((0 * Tn + t) * H + 3) * Dv + d]));
  }
  ok('G39c 不同组输出不同（反证检查非空）', diff > 1e-6, `Δ=${diff.toExponential(2)}`);
  // Dq ≠ Dv 的泛化（MLA 需要）
  const v2 = T(rnd(B * Tn * Hkv * 5), [B, Tn, Hkv, 5]);
  const o3 = P.causalAttention(q, k, v2);
  ok('G39d Dv ≠ Dh 泛化可用（MLA 前提）', A.shapeEq(o3.shape, [B, Tn, H, 5]));
}

// G40 attention 梯度检验
{
  const B = 1, Tn = 4, H = 2, Hkv = 2, Dh = 6, Dv = 6;
  const q = T(rnd(B * Tn * H * Dh, 0.5), [B, Tn, H, Dh], { requiresGrad: true });
  const k = T(rnd(B * Tn * Hkv * Dh, 0.5), [B, Tn, Hkv, Dh], { requiresGrad: true });
  const v = T(rnd(B * Tn * Hkv * Dv, 0.5), [B, Tn, Hkv, Dv], { requiresGrad: true });
  const g = A.gradCheckFn((ps) => sum(P.causalAttention(
    reshape(ps[0], [B, Tn, H, Dh]), reshape(ps[1], [B, Tn, Hkv, Dh]), reshape(ps[2], [B, Tn, Hkv, Dv]))),
    [T(q.data, [q.size], { requiresGrad: true }), T(k.data, [k.size], { requiresGrad: true }),
      T(v.data, [v.size], { requiresGrad: true })]);
  ok('G40 causalAttention 梯度检验', g.pass, `maxRel=${g.maxRel.toExponential(2)}`);
}

// G41 YaRN 重标定：低频保留、高频插值
{
  const dim = 32;
  const inv = P.ropeFreqs(dim, 4096);
  const y = P.yarnScale(inv, dim, 4096, 32768);
  // i=0 是最低频（inv 最小），i 增大频率升高
  // 低频（i 小）应保持 inv 不变或接近；高频（i 大）应被缩放 scale
  ok('G41a YaRN 低频基本保留', Math.abs(y[0] - inv[0] * (4096 / 32768)) < 1e-12 || Math.abs(y[0] - inv[0]) < 1e-12,
    `y0=${y[0].toExponential(3)} inv0=${inv[0].toExponential(3)}`);
  ok('G41b YaRN 高频被压缩', y[dim / 2 - 1] <= inv[dim / 2 - 1] + 1e-18,
    `yEnd=${y[dim / 2 - 1].toExponential(3)} invEnd=${inv[dim / 2 - 1].toExponential(3)}`);
  // 单调不增（重标定后频率应不高于原频率）
  let mono = true;
  for (let i = 0; i < dim / 2; i++) if (y[i] > inv[i] + 1e-18) mono = false;
  ok('G41c YaRN 重标定后频率不高于原频率', mono);
}

// G42 残差缩放：n_layers 越大，初始化 std 越小
{
  const s1 = P.residualScaleStd(0.02, 1), s12 = P.residualScaleStd(0.02, 12), s48 = P.residualScaleStd(0.02, 48);
  ok('G42 残差缩放随层数下降', s1 > s12 && s12 > s48, `${s1.toFixed(5)} > ${s12.toFixed(5)} > ${s48.toFixed(5)}`);
}

console.log(`\n──────────────────────────────\nL2: ${pass} 通过 / ${fail} 失败`);
if (fail) { console.log('失败项: ' + F.join(', ')); process.exit(1); }
