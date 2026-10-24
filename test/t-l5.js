'use strict';
const L5 = require('../src/l5-orchestration.js');
const { Runtime, planIntent, planPipeline, checkInvariant, allFinite } = L5;

let pass = 0, fail = 0; const F = [];
function ok(name, cond, info) {
  if (cond) { pass++; console.log(`  ✓ ${name}${info ? '  ' + info : ''}`); }
  else { fail++; F.push(name); console.log(`  ✗ ${name}  ${info || ''}`); }
}
const deq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const SEQ = [0.1, -0.3, 0.5, 0.2, -0.1, 0.4, 0.05, -0.25];

console.log('\n=== L5 编排运行时门禁 ===\n');

// G54a 注意力块：可运行、输出同长、全有限、过门禁
{
  const rt = new Runtime();
  const { output, gates } = rt.execute('attn_block', { seq: SEQ.slice() });
  ok('G54a attn_block 输出长度=输入', Array.isArray(output) && output.length === SEQ.length, `len=${output.length}`);
  ok('G54a2 attn_block 输出全有限', allFinite(output));
  ok('G54a3 门禁计数=3步×3类', gates.determinism === 3 && gates.finite === 3 && gates.invariant === 3, JSON.stringify(gates));
}

// G54b 分类意图：softmax 不变量（行和=1）在运行时被校验通过
{
  const rt = new Runtime();
  const { output } = rt.execute('classify', { seq: SEQ.slice() });
  const s = output.reduce((a, b) => a + b, 0);
  ok('G54b classify → 分布和=1（门禁G3通过）', Math.abs(s - 1) < 1e-9, `Σ=${s}`);
}

// G54c G1 确定性：同程序两次执行逐位一致
{
  const r1 = new Runtime().execute('attn_block', { seq: SEQ.slice() });
  const r2 = new Runtime().execute('attn_block', { seq: SEQ.slice() });
  ok('G54c 同程序双执行逐位一致（门禁G1）', deq(r1.output, r2.output));
}

// G54d 流水线编排：attn → ffn → classify 串成多阶段程序
{
  const prog = planPipeline(['attn_block', 'ffn_block', 'classify'], { seq: SEQ });
  const rt = new Runtime();
  rt.store.set('seq', SEQ.slice());
  const out = rt.run(prog);
  ok('G54d 流水线可执行且输出有限', allFinite(out) && out.length >= 1, `steps=${prog.steps.length} out=${JSON.stringify(out).slice(0,40)}`);
}

// G54e planner 结构正确
{
  const p = planIntent('attn_block', { seq: [] });
  ok('G54e planner 生成 3 步且末步=gelu', p.steps.length === 3 && p.steps[2].op === 'act.gelu' && p.output === 'a');
}

// G54f 未知意图必须抛异常
{
  let threw = false;
  try { planIntent('no_such_intent', {}); } catch (e) { threw = true; }
  ok('G54f 未知意图抛异常', threw);
}

// G54g 负面测试：不变量检查器必须抓出违例（证明门禁非空）
{
  ok('G54g1 softmax 行和≠1 被抓出', checkInvariant('norm.softmax', [0.4, 0.3, 0.2]) !== null);
  ok('G54g2 softmax 行和=1 通过', checkInvariant('norm.softmax', [1/3, 1/3, 1/3]) === null);
  ok('G54g3 topk one-hot 含 0.5 被抓出', checkInvariant('topk.top1', [0.5, 0, 0.5]) !== null);
}

// G54h 负面测试：finite 守卫逻辑
{
  ok('G54h1 allFinite 正常数组=true', allFinite([1, 2, 3]));
  ok('G54h2 allFinite 含 NaN=false', !allFinite([1, NaN, 3]));
  ok('G54h3 allFinite 含 Inf=false', !allFinite([1, Infinity, 3]));
}

console.log(`\n── L5 结果：${pass} 通过 / ${fail} 失败 ──`);
if (fail) { console.log('失败项：\n - ' + F.join('\n - ')); process.exit(1); }
else console.log('L5 全绿 ✓');
