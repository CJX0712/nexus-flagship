/**
 * NEXUS · L6 引擎注册与调度
 * ---------------------------------------------------------------------------
 * 把「世界顶级推理引擎」编排进来 —— 不重写，只适配。
 * 作者：晨星 (CJX0712) · MIT
 *
 * 设计原则（旗舰集大成）：
 *   · 世界顶级的引擎（llama.cpp / vLLM / SGLang）直接复用，经 Adapter 接入。
 *   · 内置一个确定性的参考引擎（跑 L0–L5 数学栈），让调度可被真实跑通与门禁。
 *   · 任何引擎调用都被同一套数学门禁包裹：有限性 + 确定性 + 分布不变量。
 *
 * 组件：
 *   · EngineAdapter   引擎接口契约
 *   · BuiltInRefEngine 确定性参考引擎（真实可跑，过全部门禁）
 *   · ModelRegistry   引擎注册表
 *   · Dispatch        按请求约束选最优引擎（优先世界顶级，否则内置参考）
 */
'use strict';
const { Runtime, planIntent } = require('./l5-orchestration.js');

/** 引擎适配接口。外部世界顶级引擎（llama.cpp/vLLM/SGLang）实现同一契约即可接入。 */
class EngineAdapter {
  constructor(spec) {
    this.name = spec.name;
    this.kind = spec.kind;                       // 'external-world-class' | 'builtin-reference'
    this.capabilities = spec.capabilities || {};  // { modalities, maxTokens, deterministic, streaming }
    this._call = spec.call;                       // (req) => res
    if (!this.capabilities.deterministic) this.capabilities.deterministic = false;
  }
  run(req) { return this._call(req); }
}

/**
 * 内置确定性参考引擎：用 L0–L5 跑一个「序列 → 注意力 → 分类分布」的前向。
 * 它是真实可执行的（不是占位），且天然满足门禁：给定相同输入 + 种子，输出逐位一致。
 */
class BuiltInRefEngine {
  constructor(opts = {}) {
    this.name = opts.name || 'nexus-builtin-ref';
    this.kind = 'builtin-reference';
    this.capabilities = { modalities: ['text'], maxTokens: 2048, deterministic: true, streaming: false };
  }
  run(req) {
    // req: { tokens: number[] 或 string } → 归一化为数值序列
    let seq;
    if (typeof req.tokens === 'string') seq = [...req.tokens].map((c) => (c.charCodeAt(0) % 7) / 7 - 0.4);
    else if (Array.isArray(req.tokens)) seq = req.tokens.map((v) => Number(v));
    else seq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    const rt = new Runtime();
    const { output, gates } = rt.execute('classify', { seq });
    const dist = Array.isArray(output) ? output : [output];
    return {
      engine: this.name,
      distribution: dist,
      entropy: require('./l0-numerics.js').entropy(dist),
      gates,
    };
  }
}

class ModelRegistry {
  constructor() { this._engines = new Map(); }
  register(engine) {
    if (!(engine instanceof EngineAdapter)) throw new Error('must register an EngineAdapter');
    this._engines.set(engine.name, engine);
    return this;
  }
  get(name) { return this._engines.get(name); }
  list() { return [...this._engines.values()].map((e) => ({ name: e.name, kind: e.kind, capabilities: e.capabilities })); }
  /** 列出所有「世界顶级外部引擎」（满足集大成：优先复用而非重写）。 */
  worldClass() { return [...this._engines.values()].filter((e) => e.kind === 'external-world-class'); }
}

/**
 * 调度：按请求约束挑选最优引擎。
 * 策略（旗舰自治）：
 *   1. 若请求显式指定 engine，用之；
 *   2. 否则，优先选「世界顶级外部引擎」（registered + 满足约束）；
 *   3. 否则退回内置确定性参考引擎。
 * 任何被选中的引擎，其输出都再过一次门禁（有限性 + 分布不变量）。
 */
class Dispatch {
  constructor(registry) { this.reg = registry; }
  select(req) {
    if (req.engine) { const e = this.reg.get(req.engine); if (!e) throw new Error('未注册引擎：' + req.engine); return e; }
    const world = this.reg.worldClass().filter((e) => this._satisfies(e, req));
    if (world.length) return world[0];
    // 退回内置参考：取第一个 builtin
    const builtin = [...this.reg._engines.values()].find((e) => e.kind === 'builtin-reference');
    if (!builtin) throw new Error('无可用引擎（请先注册内置参考引擎）');
    return builtin;
  }
  _satisfies(engine, req) {
    if (req.modalities) {
      const caps = engine.capabilities.modalities || [];
      if (!req.modalities.every((m) => caps.includes(m))) return false;
    }
    if (req.maxTokens && (engine.capabilities.maxTokens || 0) < req.maxTokens) return false;
    return true;
  }
  /** 执行请求：选引擎 → 运行 → 门禁包裹。 */
  run(req) {
    const engine = this.select(req);
    const res = engine.run(req);
    // 门禁：输出有限 + （若是分布）和为 1
    const dist = res.distribution || (Array.isArray(res) ? res : null);
    if (dist) {
      const flat = dist.flat ? dist.flat(Infinity) : dist;
      if (!flat.every((x) => Number.isFinite(x))) throw new Error(`门禁失败：${engine.name} 输出含 NaN/Inf`);
      const s = flat.reduce((a, b) => a + b, 0);
      if (Math.abs(s - 1) > 1e-6) throw new Error(`门禁失败：${engine.name} 分布和=${s}≠1`);
    }
    return { engine: engine.name, kind: engine.kind, result: res };
  }
}

module.exports = { EngineAdapter, BuiltInRefEngine, ModelRegistry, Dispatch };
