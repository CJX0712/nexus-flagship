/**
 * NEXUS · 旗舰 AI 操作系统 · 顶层门面 (L7 API / L8 治理 / L9 元编排)
 * ---------------------------------------------------------------------------
 * 把 L0–L6 全部零件 + 1024 条微指令 ISA 收束为一个可调用、可审计、可编排的系统。
 * 作者：晨星 (CJX0712) · MIT
 *
 *   L7  API      —— registerEngine / dispatch / run / isa / benchmark
 *   L8  治理      —— 每次调用写审计日志（引擎、门禁结果、耗时、不变量）
 *   L9  元编排    —— 把「意图」翻译成 ISA 程序并驱动运行时，产出可复现结果
 *
 * 零重写宣言：底层数学/算法复用已验证配方；世界顶级引擎经 Adapter 接入；
 * 本层只做「没人做的编排 + 自证」，不重造轮子。
 */
'use strict';
const { getISA, call } = require('./l4-isa.js');
const { Runtime, planIntent, planPipeline } = require('./l5-orchestration.js');
const { EngineAdapter, BuiltInRefEngine, ModelRegistry, Dispatch } = require('./l6-engines.js');

class NEXUS {
  constructor() {
    this.registry = new ModelRegistry();
    this._dispatch = new Dispatch(this.registry);
    this.audit = [];                                   // L8 治理审计日志
    // 默认注册内置确定性参考引擎（保证系统开箱可跑）
    this.registry.register(new EngineAdapter({
      name: 'nexus-builtin-ref', kind: 'builtin-reference',
      capabilities: { modalities: ['text'], maxTokens: 2048, deterministic: true },
      call: (req) => new BuiltInRefEngine().run(req),
    }));
  }

  /** L7：注册一个世界顶级外部引擎（llama.cpp / vLLM / SGLang …）。 */
  registerEngine(spec) { const e = spec instanceof EngineAdapter ? spec : new EngineAdapter(spec); this.registry.register(e); return this; }

  /** L4：返回 ISA 统计。 */
  isa() {
    const isa = getISA();
    const byKind = { core: 0, sweep: 0 };
    const byFamily = {};
    for (const t of isa.table) { byKind[t.kind]++; byFamily[t.family] = (byFamily[t.family] || 0) + 1; }
    return { total: isa.size, families: isa.byFamily.size, byKind, byFamily };
  }

  /** L9：把高层意图编译为 ISA 程序并直接执行（不经过外部引擎）。 */
  runIntent(intent, data) {
    const t0 = Date.now();
    const rt = new Runtime();
    const { output, gates } = rt.execute(intent, data);
    const rec = { kind: 'intent', intent, gates, ms: Date.now() - t0, ok: true };
    this.audit.push(rec);
    return { output, gates: rec.gates, ms: rec.ms };
  }

  /** L7/L9：执行一次推理请求（经调度 + 门禁 + 审计）。 */
  dispatch(req) {
    const t0 = Date.now();
    const res = this._dispatch.run(req);
    const rec = { kind: 'dispatch', engine: res.engine, kind2: res.kind, ms: Date.now() - t0, ok: true };
    this.audit.push(rec);
    return res;
  }

  /** L7：把一句话/数字序列分类为分布。 */
  classify(tokens) { return this.dispatch({ tokens }); }

  /** L7：端到端基准——跑一条 attn→ffn→classify 流水线并返回指标。 */
  benchmark(seq) {
    const t0 = Date.now();
    const prog = planPipeline(['attn_block', 'ffn_block', 'classify'], { seq: seq || [0.1, -0.2, 0.3, 0.05, -0.1, 0.4] });
    const rt = new Runtime();
    rt.store.set('seq', seq || [0.1, -0.2, 0.3, 0.05, -0.1, 0.4]);
    const out = rt.run(prog);
    const ms = Date.now() - t0;
    const rec = { kind: 'benchmark', steps: prog.steps.length, ms, ok: true };
    this.audit.push(rec);
    return { output: out, steps: prog.steps.length, ms, gates: rt.gates };
  }

  /** L8：导出审计日志（治理可追因）。 */
  auditLog() { return this.audit.slice(); }
  resetAudit() { this.audit = []; }
}

// ── CLI ─────────────────────────────────────────────────────────────────────
function cli(argv) {
  const args = argv.slice(2);
  const cmd = args[0];
  const nexus = new NEXUS();
  const flag = (f) => args.includes(f);
  const val = (f) => { const i = args.indexOf(f); return i >= 0 ? args[i + 1] : null; };

  if (cmd === 'isa' || flag('--isa')) {
    console.log(JSON.stringify(nexus.isa(), null, 2));
  } else if (cmd === 'benchmark' || flag('--benchmark')) {
    const r = nexus.benchmark();
    console.log(`流水线 ${r.steps} 步 · ${r.ms}ms · 门禁 ${JSON.stringify(r.gates)}`);
    console.log('输出:', JSON.stringify(r.output));
  } else if (cmd === 'classify' || flag('--classify')) {
    const txt = val('--classify') || (cmd === 'classify' ? args[1] : 'hello nexus');
    const r = nexus.classify(txt);
    console.log(`引擎=${r.engine} 分布=${JSON.stringify(r.result.distribution.map((x) => +x.toFixed(4)))}`);
  } else if (cmd === 'run' || flag('--run')) {
    const intent = val('--run') || (cmd === 'run' ? args[1] : 'attn_block');
    const r = nexus.runIntent(intent, { seq: [0.1, -0.2, 0.3, 0.05, -0.1, 0.4] });
    console.log(`意图=${intent} 输出有限=${Array.isArray(r.output) ? r.output.every(Number.isFinite) : Number.isFinite(r.output)} 门禁=${JSON.stringify(r.gates)}`);
  } else if (cmd === 'audit' || flag('--audit')) {
    console.log(JSON.stringify(nexus.auditLog(), null, 2));
  } else {
    console.log([
      'NEXUS · 晨星旗舰 AI 操作系统',
      '用法：',
      '  node nexus.js isa            查看 1024 微指令 ISA 统计',
      '  node nexus.js benchmark      跑 attn→ffn→classify 流水线',
      '  node nexus.js classify "文本"  把文本分类为分布（经调度+门禁）',
      '  node nexus.js run attn_block 直接执行某意图',
      '  node nexus.js audit          打印治理审计日志',
    ].join('\n'));
  }
}

// 一并再导出引擎层类型，便于外部注册自定义引擎（浏览器构建同样可见）
module.exports = { NEXUS, cli, EngineAdapter, BuiltInRefEngine, ModelRegistry, Dispatch };
// 仅在 Node CLI 直接执行时启动；打包进浏览器时 process/require.main 均不存在，必须守卫。
if (typeof process !== 'undefined' && typeof require !== 'undefined' && require.main === module) cli(process.argv);
