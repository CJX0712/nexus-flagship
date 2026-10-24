'use strict';
const L4 = require('../src/l4-isa.js');
const { getISA, call, callIdx } = L4;

let pass = 0, fail = 0; const F = [];
function ok(name, cond, info) {
  if (cond) { pass++; console.log(`  ✓ ${name}${info ? '  ' + info : ''}`); }
  else { fail++; F.push(name); console.log(`  ✗ ${name}  ${info || ''}`); }
}
const deq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const isa = getISA();
const SAMPLE = [0.1, -0.3, 0.5, 0.2, -0.1, 0.4, 0.05, -0.25];   // 8 维确定性样本
const SAMPLE2 = [0.3, 0.1, -0.2, 0.6, -0.4, 0.2, 0.0, 0.15];

console.log('\n=== L4 微指令 ISA 门禁 (目标 1024) ===\n');

// G52a/b/c/e：规模、家族齐整、名字唯一、全可调、全确定
ok('G52a ISA 总数 = 1024', isa.size === 1024, `size=${isa.size}`);
{
  let all32 = true, bad = [];
  for (const [fid, f] of isa.byFamily) if (f.members.length !== 32) { all32 = false; bad.push(fid); }
  ok('G52b 每个家族恰好 32 成员', all32, bad.length ? bad.join(',') : `${isa.byFamily.size} 家族`);
}
{
  const names = isa.table.map((t) => t.name);
  ok('G52c 1024 条名字全部唯一', new Set(names).size === names.length, `unique=${new Set(names).size}`);
}
{
  const core = isa.table.filter((t) => t.kind === 'core').length;
  const sweep = isa.table.filter((t) => t.kind === 'sweep').length;
  ok('G52d 诚实记账：15 core + 17 sweep = 32 家族', core === 480 && sweep === 544, `core=${core} sweep=${sweep}`);
}

// G52e：1024 条逐条可调用且不抛异常
{
  let threw = 0, firstBad = '';
  for (const e of isa.table) {
    try { e.fn(SAMPLE, SAMPLE2); }
    catch (err) { threw++; if (!firstBad) firstBad = e.name + ': ' + err.message; }
  }
  ok('G52e 1024 条全部可调用（无异常）', threw === 0, threw ? firstBad : '0 thrown');
}

// G52f：1024 条逐条确定性（同输入 → 同输出，逐位一致）
{
  let nondet = 0, firstBad = '';
  for (const e of isa.table) {
    let o1, o2;
    try { o1 = e.fn(SAMPLE, SAMPLE2); o2 = e.fn(SAMPLE, SAMPLE2); }
    catch (err) { nondet++; if (!firstBad) firstBad = e.name + ': ' + err.message; continue; }
    if (!deq(o1, o2)) { nondet++; if (!firstBad) firstBad = e.name; }
  }
  ok('G52f 1024 条全部确定性（双调用逐位一致）', nondet === 0, nondet ? firstBad : '0 nondeterministic');
}

// G53 语义抽查：几条关键指令必须算对
{
  // 注意力是自注意力：输出是输入的凸组合 → 落在 [min,max] 内，且长度一致
  const y = call('attn.sdpa', SAMPLE);
  const lo = Math.min(...SAMPLE), hi = Math.max(...SAMPLE);
  const inRange = y.length === SAMPLE.length && y.every((v) => v >= lo - 1e-9 && v <= hi + 1e-9);
  ok('G53a attn.sdpa 输出长度=输入且为凸组合(落于值域)', inRange, `len=${y.length}`);

  // 因果注意力：第 0 个 query 只看自己，输出 == 输入第 0 位
  const yc = call('attn.causal_sdpa', [0.5, 0.2, -0.3]);
  ok('G53b attn.causal 首位只看自身', Math.abs(yc[0] - 0.5) < 1e-12, `yc0=${yc[0]}`);

  // MSE 损失
  ok('G53c loss.mse 同向量 → 0', Math.abs(call('loss.mse', [1, 2, 3], [1, 2, 3])) < 1e-15);
  ok('G53d loss.mse ([0],[2]) → 4', Math.abs(call('loss.mse', [0], [2]) - 4) < 1e-12, `got=${call('loss.mse', [0], [2])}`);

  // 激活函数基准值（act 家族返回标量，非数组）
  ok('G53e act.gelu(0) = 0', call('act.gelu', 0) === 0);
  ok('G53f act.sigmoid(0) = 0.5', Math.abs(call('act.sigmoid', 0) - 0.5) < 1e-12, `got=${call('act.sigmoid', 0)}`);

  // 扫描族：distp[0] 的 p = 0.25，lpnorm 是 Lp 距离：||[2]-[0]||_0.25 = 2^(1/0.25) = 2
  const d0 = call('distp[0]', [2], [0]);
  ok('G53g distp[0] L0.25 距离校验 (||[2]-[0]||=2)', Math.abs(d0 - 2) < 1e-9, `got=${d0}`);

  // 调度器接口（callIdx 标量输入 → 标量输出；act 现已逐元素）
  ok('G53h callIdx(act,0) = identity', callIdx('act', 0, 2) === 2);
  ok('G53h2 act.gelu 逐元素（数组→数组）', deq(call('act.gelu', [0, 2]), [0, call('act.gelu', 2)]));
  let thrown = false;
  try { call('nope.nope'); } catch (e) { thrown = true; }
  ok('G53i 未知微指令必须抛异常（不静默 undefined）', thrown);
}

console.log(`\n── L4 结果：${pass} 通过 / ${fail} 失败 ──`);
if (fail) { console.log('失败项：\n - ' + F.join('\n - ')); process.exit(1); }
else console.log('L4 全绿 ✓');
