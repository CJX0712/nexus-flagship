'use strict';
const { NEXUS } = require('../src/nexus.js');
const { EngineAdapter } = require('../src/l6-engines.js');

let pass = 0, fail = 0; const F = [];
function ok(name, cond, info) {
  if (cond) { pass++; console.log(`  ✓ ${name}${info ? '  ' + info : ''}`); }
  else { fail++; F.push(name); console.log(`  ✗ ${name}  ${info || ''}`); }
}

console.log('\n=== L7/L8/L9 顶层门面门禁 ===\n');

// G56a ISA 统计正确
{
  const n = new NEXUS();
  const s = n.isa();
  ok('G56a isa() 总条数=1024', s.total === 1024, `total=${s.total}`);
  ok('G56a2 isa() 家族=32 且 core/sweep 记账正确', s.families === 32 && s.byKind.core === 480 && s.byKind.sweep === 544);
}

// G56b runIntent 可执行且过门禁、写审计
{
  const n = new NEXUS();
  const r = n.runIntent('attn_block', { seq: [0.1, -0.2, 0.3, 0.05] });
  ok('G56b runIntent 输出有限', Array.isArray(r.output) && r.output.every(Number.isFinite));
  ok('G56b2 门禁计数完整', r.gates.determinism === 3 && r.gates.finite === 3 && r.gates.invariant === 3);
  ok('G56b3 审计日志已写入', n.auditLog().length >= 1 && n.auditLog()[0].kind === 'intent');
}

// G56c benchmark 跑完整流水线（attn 3 + ffn 3 + classify 1 = 7 步）
{
  const n = new NEXUS();
  const r = n.benchmark([0.1, -0.2, 0.3, 0.05, -0.1, 0.4, 0.2, -0.3]);
  ok('G56c benchmark 7 步流水线 + 门禁', r.steps === 7 && r.gates.determinism === 7, `steps=${r.steps}`);
}

// G56d 默认注册内置引擎，classify 可用
{
  const n = new NEXUS();
  const r = n.classify('晨星旗舰');
  ok('G56d classify 经内置引擎→分布和=1', Math.abs(r.result.distribution.reduce((a, b) => a + b, 0) - 1) < 1e-9, `engine=${r.engine}`);
}

// G56e 注册世界顶级外部引擎后被优先调度
{
  const n = new NEXUS();
  n.registerEngine({ name: 'sglang', kind: 'external-world-class', capabilities: { modalities: ['text'], maxTokens: 64000 }, call: () => ({ distribution: [0.2, 0.3, 0.5] }) });
  const r = n.dispatch({ tokens: 'x' });
  ok('G56e 调度优先选 sglang（外部顶级）', r.engine === 'sglang');
}

// G56f 治理：审计日志可追因（含耗时与门禁）
{
  const n = new NEXUS();
  n.classify('a'); n.benchmark(); n.runIntent('classify', { seq: [1, 2, 3] });
  const log = n.auditLog();
  ok('G56f 审计含 dispatch/benchmark/intent 三类', log.some((l) => l.kind === 'dispatch') && log.some((l) => l.kind === 'benchmark') && log.some((l) => l.kind === 'intent'));
  ok('G56f2 每条审计含耗时字段', log.every((l) => typeof l.ms === 'number'));
}

// G56g 负面：分布非法的引擎被门禁拦截并记入审计失败
{
  const n = new NEXUS();
  n.registry.register(new EngineAdapter({ name: 'bad', kind: 'builtin-reference', capabilities: { modalities: ['text'] }, call: () => ({ distribution: [0.5, 0.5, 0.5] }) }));
  let threw = false;
  try { n.dispatch({ engine: 'bad', tokens: 'x' }); } catch (e) { threw = true; }
  ok('G56g 非法分布被门禁拦截', threw);
}

console.log(`\n── L7/L8/L9 结果：${pass} 通过 / ${fail} 失败 ──`);
if (fail) { console.log('失败项：\n - ' + F.join('\n - ')); process.exit(1); }
else console.log('L7/L8/L9 全绿 ✓');
