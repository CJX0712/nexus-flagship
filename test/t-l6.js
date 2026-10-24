'use strict';
const L6 = require('../src/l6-engines.js');
const { EngineAdapter, BuiltInRefEngine, ModelRegistry, Dispatch } = L6;
const L0 = require('../src/l0-numerics.js');

let pass = 0, fail = 0; const F = [];
function ok(name, cond, info) {
  if (cond) { pass++; console.log(`  ✓ ${name}${info ? '  ' + info : ''}`); }
  else { fail++; F.push(name); console.log(`  ✗ ${name}  ${info || ''}`); }
}

console.log('\n=== L6 引擎注册与调度门禁 ===\n');

// G55a 内置参考引擎：可运行，输出为合法分布（Σ=1），且过门禁
{
  const eng = new BuiltInRefEngine();
  const r1 = eng.run({ tokens: [0.1, -0.2, 0.3, 0.05] });
  const r2 = eng.run({ tokens: [0.1, -0.2, 0.3, 0.05] });
  const s1 = r1.distribution.reduce((a, b) => a + b, 0);
  const s2 = r2.distribution.reduce((a, b) => a + b, 0);
  ok('G55a 参考引擎输出分布和=1', Math.abs(s1 - 1) < 1e-9 && Math.abs(s2 - 1) < 1e-9, `Σ=${s1.toFixed(6)}`);
  // 确定性
  const deq = JSON.stringify(r1.distribution) === JSON.stringify(r2.distribution);
  ok('G55b 参考引擎确定性（同输入逐位一致）', deq);
  ok('G55c 返回熵且满足 0≤H≤log|C|', r1.entropy >= 0 && r1.entropy <= Math.log(r1.distribution.length) + 1e-9, `H=${r1.entropy.toFixed(4)}`);
}

// G55d 注册表：注册外部世界顶级引擎 + 内置引擎
{
  const reg = new ModelRegistry();
  const llama = new EngineAdapter({
    name: 'llama.cpp', kind: 'external-world-class',
    capabilities: { modalities: ['text'], maxTokens: 128000, deterministic: false },
    call: (req) => ({ distribution: [0.25, 0.25, 0.25, 0.25] }),   // 适配层：真实接入时改这里
  });
  const builtin = new EngineAdapter({
    name: 'nexus-ref', kind: 'builtin-reference',
    capabilities: { modalities: ['text'], maxTokens: 2048, deterministic: true },
    call: (req) => new BuiltInRefEngine().run(req),
  });
  reg.register(llama).register(builtin);
  ok('G55d 注册表含 2 引擎', reg.list().length === 2);
  ok('G55d2 worldClass 正确识别外部顶级引擎', reg.worldClass().length === 1 && reg.worldClass()[0].name === 'llama.cpp');
}

// G55e 调度：未指定引擎时优先选「世界顶级外部引擎」
{
  const reg = new ModelRegistry();
  reg.register(new EngineAdapter({ name: 'llama.cpp', kind: 'external-world-class', capabilities: { modalities: ['text'], maxTokens: 128000 }, call: (req) => ({ distribution: [0.5, 0.5] }) }));
  reg.register(new EngineAdapter({ name: 'nexus-ref', kind: 'builtin-reference', capabilities: { modalities: ['text'], maxTokens: 2048 }, call: (req) => new BuiltInRefEngine().run(req) }));
  const d = new Dispatch(reg);
  const out = d.run({ tokens: [0.1, 0.2] });
  ok('G55e 调度优先选 llama.cpp（世界顶级）', out.engine === 'llama.cpp', `engine=${out.engine}`);
}

// G55f 调度：无外部引擎时退回内置参考，且门禁通过
{
  const reg = new ModelRegistry();
  reg.register(new EngineAdapter({ name: 'nexus-ref', kind: 'builtin-reference', capabilities: { modalities: ['text'], maxTokens: 2048 }, call: (req) => new BuiltInRefEngine().run(req) }));
  const d = new Dispatch(reg);
  const out = d.run({ tokens: [0.3, -0.1, 0.2] });
  ok('G55f 无外部引擎→退回内置参考', out.engine === 'nexus-ref');
  ok('G55f2 门禁：分布和=1', Math.abs(out.result.distribution.reduce((a, b) => a + b, 0) - 1) < 1e-9);
}

// G55g 显式指定引擎
{
  const reg = new ModelRegistry();
  reg.register(new EngineAdapter({ name: 'vllm', kind: 'external-world-class', capabilities: { modalities: ['text'], maxTokens: 32000 }, call: () => ({ distribution: [1, 0] }) }));
  const d = new Dispatch(reg);
  ok('G55g 显式指定 vllm 生效', d.run({ engine: 'vllm', tokens: [1] }).engine === 'vllm');
}

// G55h 负面：分布和≠1 的引擎输出必须被门禁拦下
{
  const reg = new ModelRegistry();
  reg.register(new EngineAdapter({ name: 'bad', kind: 'builtin-reference', capabilities: { modalities: ['text'] }, call: () => ({ distribution: [0.5, 0.5, 0.5] }) }));
  const d = new Dispatch(reg);
  let threw = false;
  try { d.run({ tokens: [1] }); } catch (e) { threw = true; }
  ok('G55h 分布和≠1 被门禁拦截', threw);
}

console.log(`\n── L6 结果：${pass} 通过 / ${fail} 失败 ──`);
if (fail) { console.log('失败项：\n - ' + F.join('\n - ')); process.exit(1); }
else console.log('L6 全绿 ✓');
