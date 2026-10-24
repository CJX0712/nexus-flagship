/**
 * NEXUS · L5 编排运行时
 * ---------------------------------------------------------------------------
 * 把 L0–L4 的零件组装成「可执行的程序」并强制数学门禁。
 * 作者：晨星 (CJX0712) · MIT
 *
 * 核心组件：
 *   · Store        —— 命名张量/值的存储（程序的数据上下文）
 *   · Invariant    —— 每条微指令执行后的数学不变量检查（门禁）
 *   · Runtime      —— 程序虚拟机：逐步解析输入 → 调用 ISA → 存输出 → 过门禁
 *   · Planner      —— 高层意图 → ISA 程序（编排，而非重写）
 *
 * 三条硬门禁（在每一步强制执行）：
 *   G1  确定性：同输入双调用，逐位一致（防随机数泄漏 / 浮点不确定）
 *   G2  有限性：输出无 NaN / ±Infinity
 *   G3  不变量：按算子登记的数学性质（如 softmax 各行和为 1）
 *
 * 设计原则：本层只「编排」世界顶级零件，绝不重新实现注意力/优化器/引擎。
 */
'use strict';
const { getISA, call } = require('./l4-isa.js');

const isNum = (x) => typeof x === 'number';
const flattenNum = (v, out = []) => {
  if (isNum(v)) out.push(v);
  else if (Array.isArray(v)) v.forEach((c) => flattenNum(c, out));
  return out;
};
const allFinite = (v) => flattenNum(v).every((x) => Number.isFinite(x));
const deq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** 按算子名登记的不变量检查器；返回 null 表示通过，否则返回错误信息。 */
const INVARIANTS = [
  {
    // softmax 类：归一化输出各行和为 1
    match: (op) => /^norm\.(softmax|row_softmax|col_softmax|log_softmax|sharp|temperature_down|gumbel_softmax_like)/.test(op),
    check: (op, out) => {
      const rows = Array.isArray(out[0]) ? out : [out];
      for (const r of rows) {
        const s = r.reduce((a, b) => a + b, 0);
        if (op.startsWith('norm.log_softmax')) { /* log 域，和不为 1，跳过 */ }
        else if (Math.abs(s - 1) > 1e-9) return `softmax 行和应为1，实得 ${s}`;
      }
      return null;
    },
  },
  {
    match: (op) => /^gate\.(softmax_gate|moe_softmax|top2_softmax|sigmoid|temperature_gate|shared_plus_routed)/.test(op)
      || op === 'agg.softmax_sum',
    check: (op, out) => {
      if (op === 'agg.softmax_sum') { const s = out[0]; return Math.abs(s - 1) > 1e-9 ? `softmax_sum 应为1，实得 ${s}` : null; }
      const rows = Array.isArray(out[0]) ? out : [out];
      for (const r of rows) { const s = r.reduce((a, b) => a + b, 0); if (Math.abs(s - 1) > 5e-9) return `gate 分布和应为1，实得 ${s}`; }
      return null;
    },
  },
  {
    // 选择类算子：one-hot / 子集选择输出 ∈ {0,1} 或 [0,1]
    match: (op) => /^topk\.(top|bottom|threshold|argmax|argmin|nucleus|min_p|typical|median_select|mode_select|contrastive_search|beam_like)/.test(op),
    check: (op, out) => {
      if (op === 'topk.argmax' || op === 'topk.argmin' || op === 'topk.greedy') {
        return out.length === 1 && Number.isInteger(out[0]) ? null : 'argmax/greedy 应返回单整数索引';
      }
      const flat = flattenNum(out);
      for (const x of flat) if (x !== 0 && x !== 1) return `选择算子输出应为0/1，实得 ${x}`;
      return null;
    },
  },
  {
    // 归一化标量类：输出长度不变
    match: (op) => /^norm\.(zscore|minmax|unit_l2|mean_center|robust)/.test(op),
    check: (op, out) => (Array.isArray(out) ? null : '归一化应返回数组'),
  },
];

function checkInvariant(op, out) {
  for (const inv of INVARIANTS) if (inv.match(op)) return inv.check(op, out);
  return null;
}

/** 解析输入引用：支持字面量（数组/数字）与 store 中的命名值。 */
function resolve(ref, store) {
  if (ref === null || ref === undefined) return ref;
  if (typeof ref === 'string') return store.get(ref);
  return ref; // 字面量
}

class Runtime {
  constructor() {
    this.store = new Map();
    this.log = [];
    this.gates = { determinism: 0, finite: 0, invariant: 0 };
  }
  /** 运行一个程序。program = { steps:[{id,op,inputs:[ref...],out}], output } */
  run(program) {
    for (const step of program.steps) {
      const inputs = step.inputs.map((r) => resolve(r, this.store));
      let o1, o2, err = null;
      try { o1 = call(step.op, ...inputs); } catch (e) { err = e; }
      if (err) throw new Error(`步骤 ${step.id} 调用 ${step.op} 失败：${err.message}`);
      // G1 确定性：再算一次比对
      o2 = call(step.op, ...inputs);
      this.gates.determinism++;
      if (!deq(o1, o2)) throw new Error(`门禁G1失败：${step.op} 非确定性`);
      // G2 有限性
      this.gates.finite++;
      if (!allFinite(o1)) throw new Error(`门禁G2失败：${step.op} 输出含 NaN/Inf`);
      // G3 不变量
      const invErr = checkInvariant(step.op, o1);
      this.gates.invariant++;
      if (invErr) throw new Error(`门禁G3失败：${step.op} 不变量违例：${invErr}`);
      this.store.set(step.out, o1);
      this.log.push({ step: step.id, op: step.op, out: step.out, ok: true });
    }
    return this.store.get(program.output);
  }
  /** 直接执行一个高层意图（经 planner），返回最终输出与门禁统计。 */
  execute(intent, seedData) {
    const prog = planIntent(intent, seedData);
    if (seedData) for (const [k, v] of Object.entries(seedData)) this.store.set(k, v);
    const out = this.run(prog);
    return { output: out, gates: this.gates, log: this.log };
  }
}

/**
 * Planner：高层意图 → ISA 程序。仅做编排，所有计算都落到 L4 的 1024 条微指令。
 * 支持意图：
 *   'attn_block'     单头自注意力块（attn.sdpa → norm → act）
 *   'ffn_block'      前馈块（线性近似 → gelu → 线性近似）
 *   'classify'       序列 → softmax 分类分布
 *   'quantize'       量化链（quant.int8_per_channel → dequant 近似）
 */
function planIntent(intent, seed) {
  const has = (k) => seed && k in seed;
  const seq = has('seq') ? 'seq' : (has('x') ? 'x' : 'input');
  switch (intent) {
    case 'attn_block':
      return {
        steps: [
          { id: 1, op: 'attn.sdpa', inputs: [seq], out: 'h' },
          { id: 2, op: 'norm.layer_scale', inputs: ['h'], out: 'hn' },
          { id: 3, op: 'act.gelu', inputs: ['hn'], out: 'a' },
        ],
        output: 'a',
      };
    case 'ffn_block':
      return {
        steps: [
          { id: 1, op: 'seq.ema', inputs: [seq], out: 'p1' },
          { id: 2, op: 'act.gelu', inputs: ['p1'], out: 'g' },
          { id: 3, op: 'seq.decay_ema', inputs: ['g'], out: 'p2' },
        ],
        output: 'p2',
      };
    case 'classify':
      // 把序列当作类别 logits，直接 softmax → 一个合法的多类分布
      return {
        steps: [
          { id: 1, op: 'norm.softmax', inputs: [seq], out: 'dist' },
        ],
        output: 'dist',
      };
    case 'quantize':
      return {
        steps: [
          { id: 1, op: 'quant.int8_per_channel', inputs: [seq], out: 'q' },
          { id: 2, op: 'quant.dequantize_kv', inputs: ['q'], out: 'dq' },
        ],
        output: 'dq',
      };
    default:
      throw new Error(`未知意图：${intent}`);
  }
}

/** 批量编排：把多个意图串成一条流水线（如 attn → ffn → classify）。
 *  每个子程序的内部变量名被唯一重命名，并将上一阶段的输出喂入下一阶段。 */
function planPipeline(intents, seed) {
  const seedVar = (seed && ('seq' in seed ? 'seq' : 'x' in seed ? 'x' : 'input'));
  let prev = seedVar;
  const steps = [];
  let n = 0;
  for (const it of intents) {
    const sub = planIntent(it, seed);
    const rename = {};
    let lastNew = null;
    for (const s of sub.steps) {
      n++;
      const newOut = 's' + n + '_' + s.op.replace(/[^a-zA-Z0-9]/g, '_');
      const newInputs = s.inputs.map((r) => {
        if (r === seedVar) return prev;     // 接入上一阶段输出（首阶段为原始输入）
        if (rename[r]) return rename[r];    // 子程序内部引用
        return r;
      });
      rename[s.out] = newOut;
      lastNew = newOut;
      steps.push({ id: n, op: s.op, inputs: newInputs, out: newOut });
    }
    prev = lastNew;
  }
  return { steps, output: prev };
}

module.exports = { Runtime, planIntent, planPipeline, checkInvariant, INVARIANTS, allFinite };
