# NEXUS · 晨星旗舰 AI 操作系统

> **旗舰零重写** · 旗舰集大成 · 旗舰唯一目标：锻造世界顶级 AI —— 架构最先进、性能最强、代码最优、文档最全。
> 本仓库是 NEXUS：一个把「世界顶级数学 / 算法 / 引擎」编排起来的**旗舰 AI 操作系统**，
> 零依赖纯 JavaScript，Node 与浏览器双端可运行、可复现、可验证。

作者：**晨星 (CJX0712)** · 协议：**MIT** · 完整开源、可运行、可复现、可验证。

---

## 零重写宣言（本系统的铁律）

1. **底层数学/算法**：复用已验证配方（PACF 偏自相关、Jacobi 特征分解、Newton–Schulz 极分解、Box–Muller、SipHash-2-4、PCG64-DXSM…），不重造轮子。
2. **世界顶级推理引擎**（llama.cpp / vLLM / SGLang）：**不重写**，只经 `EngineAdapter` 接入（见 `src/l6-engines.js`）。
3. **本系统只做「没人做的编排 + 自证」**：1024 条微指令 ISA、运行时门禁、引擎调度、治理审计。
4. **诚实记账**：参数扫描（sweep）家族与语义独立（core）家族分开计数，绝不虚报能力。

---

## 十层架构 + 1024 微指令 ISA

| 层 | 名称 | 文件 | 门禁 | 说明 |
|----|------|------|------|------|
| L0 | 数值内核 | `src/l0-numerics.js` | 57 | erf/logsumexp/cholesky/QR/Jacobi 特征值/SVD 极分解/PCG64/SipHash… |
| L1 | 自动微分 | `src/l1-autodiff.js` | 36 | 反向模式 tape + 中心差分梯度契约（解析梯度 vs 数值梯度） |
| L2 | 注意力原语 | `src/l2-primitives.js` | 23 | RMSNorm / 因果注意力 / RoPE / SwiGLU（双实现互证） |
| L3 | 推理引擎 | `src/l3-inference.js` | 35 | PagedAttention / RadixAttention / 严格无损投机解码 / 量化 |
| **L4** | **微指令 ISA** | `src/l4-isa.js` | 16 | **32 家族 × 32 成员 = 1024 条真实可调用微指令** |
| L5 | 编排运行时 | `src/l5-orchestration.js` | 14 | 程序虚拟机 + 数学门禁（确定性 / 有限性 / 不变量） |
| L6 | 引擎注册与调度 | `src/l6-engines.js` | 10 | 世界顶级引擎经 Adapter 接入，按约束选最优引擎 |
| L7 | API 门面 | `src/nexus.js` | 11 | `registerEngine / dispatch / run / isa / benchmark` |
| L8 | 治理审计 | `src/nexus.js` | — | 每次调用写审计日志（引擎 / 门禁 / 耗时 / 不变量） |
| L9 | 元编排 | `src/nexus.js` | — | 把「意图」编译为 ISA 程序并驱动运行时 |
| **L10** | **1024 微指令集** | `src/l4-isa.js` | — | 见下；旗舰的「指令级」底座 |
| — | 浏览器构建 | `tools/build-browser.js` → `dist/` | 12 | UMD 单文件 + 双端逐位一致性 |

> 全量门禁：**217 通过 / 0 失败 · 覆盖 9 个测试层**。运行 `npm test` 即可复现。

---

## 1024 微指令 ISA（L4 / L10）

32 个家族，每族 32 条，共 **1024** 条，**每条都是真实可调用的函数**，不是占位符。

- **15 个 core 家族（成员语义彼此独立，共 480 条）**：`act`（激活）`loss`（损失）`kern`（核）`norm`（归一化）`gate`（门控）`moe`（MoE 路由）`seq`（序列混合）`grad`（梯度算子）`cache`（缓存策略）`quant`（量化）`topk`（选择）`init`（初始化）`mask`（掩码）`agg`（聚合）`attn`（注意力机制）。
- **17 个 sweep 家族（同一算子的参数扫描，共 544 条）**：`distp`（Lp 距离）`smplt`（softmax 温度）`ropeb`（RoPE base）`enty`（Rényi 熵）`lsme`（标签平滑）`drop`（dropout）`clip`（梯度裁剪）`momt`（动量）`wdec`（权重衰减）`lrat`（学习率）`qbit`（量化位宽）`blks`（分块）`topp`（top-p）`beam`（beam 宽）`gqag`（GQA 分组）`moek`（MoE top-k）`scan`（关联扫描）。

三条硬门禁（已用测试锁死）：
1. **恰好 1024 条**，不多不少；
2. **每条真实可调用**，无异常；
3. **每条确定性**：同输入 → 同输出，逐位一致。

---

## 数学门禁体系（旗舰自证）

每一层都有可独立复现的「不变量」门禁，尺子不撒谎：

- **L0**：SipHash 三条官方向量逐位匹配；Jacobi 残差 ‖Av−λv‖≈ε；SVD 极因子余弦相似度 > 0.99。
- **L1**：`maxRelErr≈1e-7`（中心差分 vs 解析梯度）；广播形状正确。
- **L2**：RMSNorm 缩放不变；因果掩码使「未来 key 不影响过去 query」。
- **L3**：投机解码 **χ² 无损性检验**（df=5，临界 20.5，实测 0.75）；per-channel 量化误差 0.7% vs per-tensor 100%。
- **L5/L8**：运行时每一步强制 **G1 确定性 + G2 有限性（无 NaN/Inf）+ G3 不变量**（如 softmax 行和=1），违反即拦截并记入审计。

---

## 快速开始

```bash
# 依赖：零。仅 Node ≥ 14。
npm test                        # 全量门禁（217 项，含浏览器双端一致性）
npm run build:browser           # 生成 dist/nexus.browser.js（UMD，171 KB）
node src/nexus.js isa           # 查看 1024 微指令 ISA 统计
node src/nexus.js benchmark     # 跑 attn→ffn→classify 流水线
node src/nexus.js classify "晨星旗舰AI"   # 经调度+门禁做分类
node src/nexus.js run attn_block          # 直接执行某意图
```

浏览器（同一份代码，双端逐位一致）：

```html
<script src="dist/nexus.browser.js"></script>
<script>
  const nexus = new NEXUS.NEXUS();
  console.log(nexus.isa());          // { total: 1024, families: 32, ... }
  nexus.classify('晨星旗舰AI');       // 经调度 + 门禁 + 审计
</script>
```

打开 `dist/index.html` 即可看到浏览器内自检面板（ISA 家族表 / 在线分类 / 门禁计数）。

在代码中使用：

```js
const { NEXUS } = require('nexus-flagship/src/nexus.js');
const nexus = new NEXUS();

// 1) 接入世界顶级引擎（零重写，只写适配层）
nexus.registerEngine({
  name: 'llama.cpp', kind: 'external-world-class',
  capabilities: { modalities: ['text'], maxTokens: 128000 },
  call: (req) => ({ distribution: [/* …接 llama.cpp 输出… */] }),
});

// 2) 直接跑一条 ISA 编排的意图
const r = nexus.runIntent('attn_block', { seq: [0.1, -0.2, 0.3] });

// 3) 经调度 + 门禁 + 审计做推理
const out = nexus.dispatch({ tokens: 'hello nexus' });
```

---

## 与世界顶级引擎集成（集大成）

`EngineAdapter` 是统一契约。任何引擎（本地 llama.cpp、远程 vLLM/SGLang、甚至你自己的内核）只要实现 `call(req)` 即可接入；
`Dispatch` 按请求约束（模态、最大 token、确定性）**优先选择世界顶级外部引擎**，无外部引擎时退回内置确定性参考引擎。
所有引擎输出都再过同一套数学门禁，**非法分布（和≠1）/ NaN 一律拦截**。

---

## 验证与署名

- 全量门禁： **217 通过 / 0 失败**（L0–L7 + ISA 双层 + 浏览器构建）。
- 双端：Node 22 实测全绿；`dist/nexus.browser.js`（UMD，171 KB）在浏览器直接可用，且**与 Node 路径逐位一致**（G57e/f/g 已锁死）。
- 署名：**晨星 (CJX0712)**。本成果完整开源、可运行、可复现、可验证。

> 旗舰闭环交付：本仓库即终点 —— 拉取即可跑、改即可验、读即可懂。
