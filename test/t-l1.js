'use strict';
const L0 = require('../src/l0-numerics.js');
const A = require('../src/l1-autodiff.js');
const { T, add, sub, mul, div, matmul, transpose, reshape, sum, mean, tanh, relu, sigmoid, gelu, silu, exp, log, sqrt, square, softmaxRows, crossEntropy, mse, gradCheckFn } = A;

let pass = 0, fail = 0; const F = [];
function ok(name, cond, info) {
  if (cond) { pass++; console.log(`  \u2713 ${name}${info ? '  ' + info : ''}`); }
  else { fail++; F.push(name); console.log(`  \u2718 ${name}  ${info || ''}`); }
}
const rng = new L0.PCG128SM(20241024);
const R = (n, rg = true) => T(Array.from({ length: n }, () => rng.normal()), [n], { requiresGrad: rg });

console.log('\n=== L1 反向契约门禁 ===\n');

// G21 每个算子的 tape 梯度 vs 中心差分（这是 L1 的总门禁）
{
  const ops = [
    ['add', (ps) => sum(add(ps[0], ps[1]))],
    ['sub', (ps) => sum(sub(ps[0], ps[1]))],
    ['mul', (ps) => sum(mul(ps[0], ps[1]))],
    // 分母远离 0：用 square+常数做安全变换，**必须保持计算图连通** ——
    // 若另建 Tensor 包一层，ps[1] 就被踢出图外，梯度为 null。
    ['div', (ps) => sum(div(ps[0], add(square(ps[1]), T(new Float64Array(ps[1].size).fill(0.5), ps[1].shape))))],
    ['tanh', (ps) => sum(tanh(ps[0]))],
    ['relu', (ps) => sum(relu(ps[0]))],
    ['sigmoid', (ps) => sum(sigmoid(ps[0]))],
    ['gelu', (ps) => sum(gelu(ps[0]))],
    ['silu', (ps) => sum(silu(ps[0]))],
    ['square', (ps) => sum(square(ps[0]))],
    ['exp', (ps) => sum(exp(mul(ps[0], T(new Float64Array(ps[0].size).fill(0.1), ps[0].shape))))],
    ['sqrt', (ps) => sum(sqrt(add(square(ps[0]), T(new Float64Array(ps[0].size).fill(0.3), ps[0].shape))))],
    ['log', (ps) => sum(log(add(square(ps[0]), T(new Float64Array(ps[0].size).fill(0.3), ps[0].shape))))],
    ['sum', (ps) => sum(ps[0])],
    ['mean', (ps) => mean(ps[0])],
  ];
  let worst = 0, worstOp = '';
  // 一元算子的第二个参数不进图 → 若硬塞进 gradCheckFn 会因 grad===null 报错。
  // 这是 gradCheckFn 主动抛出的诊断，不是缺陷：它逼你把"参数是否真在图上"讲清楚。
  const arity = { add: 2, sub: 2, mul: 2, div: 2, sum: 1, mean: 1 };
  for (const [name, fn] of ops) {
    const k = arity[name] || 1;
    const params = Array.from({ length: k }, () => R(8));
    const g = gradCheckFn(fn, params);
    if (g.maxRel > worst) { worst = g.maxRel; worstOp = name; }
    ok(`G21.${name} 梯度检验`, g.pass, `maxRel=${g.maxRel.toExponential(2)}`);
  }
  ok('G21z 最差相对误差 < 1e-6', worst < 1e-6, `worst=${worstOp} ${worst.toExponential(2)}`);
}

// G22 matmul 梯度（含批量）
{
  const g2 = gradCheckFn((ps) => sum(matmul(
    reshape(ps[0], [3, 4]),
    reshape(ps[1], [4, 5]))), [R(12), R(20)]);
  ok('G22a matmul 2D 梯度', g2.pass, `maxRel=${g2.maxRel.toExponential(2)}`);
  const g3 = gradCheckFn((ps) => sum(matmul(
    reshape(ps[0], [2, 3, 4]),
    reshape(ps[1], [2, 4, 5]))), [R(24), R(40)]);
  ok('G22b matmul 3D 批量梯度', g3.pass, `maxRel=${g3.maxRel.toExponential(2)}`);
}

// G23 transpose 梯度（且 T(T(x)) == x）
{
  const g = gradCheckFn((ps) => sum(transpose(reshape(ps[0], [3, 5]))), [R(15)]);
  ok('G23a transpose 梯度', g.pass, `maxRel=${g.maxRel.toExponential(2)}`);
  const x = T([1, 2, 3, 4, 5, 6], [2, 3]);
  const tt = transpose(transpose(x));
  ok('G23b 转置两次 = 原张量', Array.from(tt.data).every((v, i) => v === x.data[i]) && A.shapeEq(tt.shape, [2, 3]));
}

// G24 softmax + 交叉熵：解析梯度有闭式解，可直接对拍
{
  const C = 6, B = 4;
  const logits = T(Array.from({ length: B * C }, () => rng.normal() * 2), [B, C], { requiresGrad: true });
  const targets = Array.from({ length: B }, () => Math.floor(rng.next() * C));
  const loss = crossEntropy(logits, targets);
  loss.backward();
  // 解析：∂L/∂x_ij = (p_ij − 1[j==y_i]) / B
  let maxErr = 0;
  for (let i = 0; i < B; i++) {
    const row = [];
    for (let j = 0; j < C; j++) row.push(logits.data[i * C + j]);
    const p = L0.softmax(row);
    for (let j = 0; j < C; j++) {
      const want = (p[j] - (j === targets[i] ? 1 : 0)) / B;
      maxErr = Math.max(maxErr, Math.abs(logits.grad[i * C + j] - want));
    }
  }
  ok('G24a 交叉熵梯度 == 闭式解', maxErr < 1e-15, `maxErr=${maxErr.toExponential(2)}`);
  const gc = gradCheckFn((ps) => crossEntropy(reshape(ps[0], [B, C]), targets), [R(B * C)]);
  ok('G24b 交叉熵 vs 中心差分', gc.pass, `maxRel=${gc.maxRel.toExponential(2)}`);
  // softmax 行和
  const s = softmaxRows(logits);
  let rowSumErr = 0;
  for (let i = 0; i < B; i++) {
    let s2 = 0; for (let j = 0; j < C; j++) s2 += s.data[i * C + j];
    rowSumErr = Math.max(rowSumErr, Math.abs(s2 - 1));
  }
  ok('G24c softmax 行和 = 1', rowSumErr < 1e-15, `maxErr=${rowSumErr.toExponential(2)}`);
}

// G25 梯度累加：一个张量被消费两次，梯度必须是和
{
  const x = T([3, 4], [2], { requiresGrad: true });
  const y = mul(x, x);              // y = x*x，x 被消费两次
  sum(y).backward();
  // ∂Σx²/∂x = 2x = [6, 8]
  ok('G25 重复消费的梯度累加正确', Math.abs(x.grad[0] - 6) < 1e-12 && Math.abs(x.grad[1] - 8) < 1e-12,
    `[${x.grad[0]}, ${x.grad[1]}]`);
}

// G26 形状契约破裂必须抛异常（不静默产生错形状）
{
  let threw = false;
  try {
    const a = T([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = T([1, 2, 3, 4], [2, 2]);
    matmul(a, b);
  } catch (e) { threw = /维度不匹配/.test(e.message); }
  ok('G26a matmul 维度不匹配 → 抛异常', threw);
  let threw2 = false;
  try { T([1, 2, 3], [2, 2]); } catch (e) { threw2 = true; }
  ok('G26b 形状/数据长度不符 → 抛异常', threw2);
  let threw3 = false;
  try { sum(T([1, 2, 3, 4], [2, 2], { requiresGrad: true })).backward([1, 2]); } catch (e) { threw3 = true; }
  ok('G26c backward 起始梯度形状不符 → 抛异常', threw3);
}

// G27 广播：右对齐规则
{
  const a = T([1, 2, 3, 4, 5, 6], [2, 3]);
  const b = T([10, 20, 30], [3]);
  const c = add(a, b);
  ok('G27a 广播形状 [2,3]+[3] → [2,3]', A.shapeEq(c.shape, [2, 3]));
  ok('G27b 广播值正确', c.data[0] === 11 && c.data[5] === 36, `[${c.data[0]}, ${c.data[5]}]`);
  const d = T([1, 2], [2, 1]);
  const e = add(a, d);
  // [[1,2,3],[4,5,6]] + [[1],[2]] = [[2,3,4],[6,7,8]] —— 注意 e.data[3] 是第二行首元素 = 4+2 = 6
  ok('G27c 广播 [2,3]+[2,1] → [2,3]', A.shapeEq(e.shape, [2, 3]) && e.data[0] === 2 && e.data[3] === 6,
    `[${Array.from(e.data).join(',')}]`);
  const g = gradCheckFn((ps) => sum(add(reshape(ps[0], [2, 3]), ps[1])), [R(6), R(3)]);
  ok('G27d 广播梯度检验', g.pass, `maxRel=${g.maxRel.toExponential(2)}`);
}

// G28 端到端：两层 MLP 训练，loss 必须单调下降
{
  const rng2 = new L0.PCG128SM(7);
  const D = 4, H = 8, OUT = 3, N = 40;
  const X = Array.from({ length: N * D }, () => rng2.normal());
  const y = Array.from({ length: N }, () => Math.floor(rng2.next() * OUT));
  const W1 = T(Array.from({ length: D * H }, () => rng2.normal() * 0.5), [D, H], { requiresGrad: true });
  const b1 = T(new Float64Array(H), [1, H], { requiresGrad: true });
  const W2 = T(Array.from({ length: H * OUT }, () => rng2.normal() * 0.5), [H, OUT], { requiresGrad: true });
  const b2 = T(new Float64Array(OUT), [1, OUT], { requiresGrad: true });
  const lr = 0.1;
  const losses = [];
  for (let step = 0; step < 60; step++) {
    const xt = T(X, [N, D]);
    const h = tanh(add(matmul(xt, W1), b1));
    const lg = add(matmul(h, W2), b2);
    const loss = crossEntropy(lg, y);
    losses.push(loss.item());
    [W1, b1, W2, b2].forEach((p) => p.zeroGrad());
    loss.backward();
    for (const p of [W1, b1, W2, b2]) {
      for (let i = 0; i < p.size; i++) p.data[i] -= lr * p.grad[i];
    }
  }
  const first5 = losses.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
  const last5 = losses.slice(-5).reduce((a, b) => a + b, 0) / 5;
  ok('G28a MLP 训练 loss 下降', last5 < first5 * 0.7, `${first5.toFixed(4)} → ${last5.toFixed(4)}`);
  // 梯度检验：MLP 首步
  const gc = gradCheckFn((ps) => {
    const xt = T(X, [N, D]);
    const w1 = reshape(ps[0], [D, H]);
    const bb1 = reshape(ps[1], [1, H]);
    const w2 = reshape(ps[2], [H, OUT]);
    const bb2 = reshape(ps[3], [1, OUT]);
    const h = tanh(add(matmul(xt, w1), bb1));
    return crossEntropy(add(matmul(h, w2), bb2), y);
  }, [T(W1.data, [W1.size], { requiresGrad: true }), T(b1.data, [b1.size], { requiresGrad: true }),
    T(W2.data, [W2.size], { requiresGrad: true }), T(b2.data, [b2.size], { requiresGrad: true })]);
  ok('G28b MLP 端到端梯度检验', gc.pass, `maxRel=${gc.maxRel.toExponential(2)}`);
}

// G29 reshape 梯度 + 元素数守恒断言
{
  const g = gradCheckFn((ps) => sum(mul(reshape(ps[0], [3, 4]), reshape(ps[0], [3, 4]))), [R(12)]);
  ok('G29a reshape 后的计算图梯度', g.pass, `maxRel=${g.maxRel.toExponential(2)}`);
  let threw = false;
  try { A.reshape(R(12), [4, 4]); } catch (e) { threw = /元素数不符/.test(e.message); }
  ok('G29b reshape 元素数不符 → 抛异常', threw);
}

// G30 拓扑序：DAG 中出现菱形依赖时梯度必须完整（不能漏算一条路径）
{
  const x = T([2], [1], { requiresGrad: true });
  const a1 = mul(x, T([3], [1], { requiresGrad: true }));   // 3x
  const a2 = mul(x, T([5], [1], { requiresGrad: true }));   // 5x
  const y = add(mul(a1, a1), mul(a2, a2));                  // (3x)² + (5x)²
  y.backward();
  // dy/dx = 2·3x·3 + 2·5x·5 = 18x + 50x = 68x = 136
  ok('G30 菱形 DAG 梯度完整', Math.abs(x.grad[0] - 136) < 1e-12, `got=${x.grad[0]} want=136`);
}

console.log(`\n──────────────────────────────\nL1: ${pass} 通过 / ${fail} 失败`);
if (fail) { console.log('失败项: ' + F.join(', ')); process.exit(1); }
