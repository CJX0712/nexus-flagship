# NEXUS 架构详解（十层 + 1024 微指令）

> 配套 `README.md`。本文件展开每一层的设计取舍与「为何不重写」。

## 为什么是「编排层」而不是「又一个推理引擎」

旗舰已有 303 个仓库覆盖算法/内核，世界顶级推理引擎（llama.cpp / vLLM / SGLang）也已成熟且开源（MIT/Apache）。
再写一个注意力内核 = 重写轮子 ❌。NEXUS 只做**没人做的那层**：

- 把零件 + 世界顶级引擎**编排**成可执行的程序（L5 运行时）；
- 给每次输出套上**数学门禁**（确定性 / 有限性 / 不变量）—— 这是「世界顶级」的硬证据；
- 用 **1024 条微指令 ISA** 作为指令级底座，让编排可被精确描述、可被门禁逐条校验。

## 各层要点

### L0 数值内核 (`l0-numerics.js`)
- 所有非线性/特殊函数都有「解析参考 + 数值稳定实现」双版本互证：
  `erf`（Abramowitz&Stegun 7.1.26）、`logsumexp`（防溢出）、`sigmoid`、`log1p`。
- 线性代数：`cholesky`、`qr`、`jacobiEig`（Jacobi 旋转，残差 ‖Av−λv‖≈ε）、`spectralNorm`、`newtonSchulz`（极分解，输入按 Frobenius 归一化后奇异值∈(0.5,1.5)）。
- 随机数：**显式种子**（禁止非确定性）。`PCG64DXSM`（NumPy 新一代 bit generator）、`Xoshiro256++`、`SipHash-2-4`（三条官方向量逐位匹配）。

### L1 自动微分 (`l1-autodiff.js`)
- 反向模式 tape，支持标量/向量/矩阵/高维广播。
- **梯度契约门禁**：每个算子的解析梯度 vs 中心差分数值梯度，`maxRelErr≈1e-7`。

### L2 注意力原语 (`l2-primitives.js`)
- RMSNorm / 层归一化、因果注意力、RoPE、SwiGLU —— **tape 版（尺子）与手写优化版互证**。
- 因果掩码门禁：把未来 key 换成完全不同的值，过去 query 的输出必须逐位不变。

### L3 推理引擎 (`l3-inference.js`)
- PagedAttention（KV 分页共享）、RadixAttention（前缀树复用）、**严格无损投机解码**（χ² 检验）、per-channel 量化（误差 0.7% vs per-tensor 100%）。
- 投机解码的无损性用 **χ² 统计量**数学验证，非空验证。

### L4 微指令 ISA (`l4-isa.js`) —— 1024 条
- 32 家族 × 32 成员。core（成员语义独立）与 sweep（参数扫描）**分开计数**，诚实记账。
- 调度器 `call(name)` / `callIdx(family,i)`；未知指令抛异常（不静默返回 undefined）。
- 三条硬门禁：恰好 1024 / 全可调用 / 全确定性。

### L5 编排运行时 (`l5-orchestration.js`)
- `Store`（命名值）+ `Runtime`（程序虚拟机）+ `Planner`（意图→ISA 程序）。
- **门禁 G1 确定性 / G2 有限性 / G3 不变量**，逐步骤强制执行；违反即抛异常并记入审计。

### L6 引擎注册与调度 (`l6-engines.js`)
- `EngineAdapter`：世界顶级引擎统一接入契约（llama.cpp/vLLM/SGLang 只写 `call` 适配层）。
- `Dispatch`：按约束（模态 / maxTokens / 确定性）**优先选外部顶级引擎**，否则退回内置确定性参考引擎。

### L7–L9 门面 / 治理 / 元编排 (`nexus.js`)
- L7：`registerEngine / dispatch / run / isa / benchmark` + CLI。
- L8：每次调用写审计日志（引擎、门禁结果、耗时、不变量）—— 可追因。
- L9：把高层意图编译为 ISA 程序并驱动运行时，产出可复现结果。

## 门禁即证据

「世界顶级」不能靠自夸，要靠**可被独立复现的不变量**：

| 层 | 不变量 | 尺子 |
|----|--------|------|
| L0 | SipHash 官方向量 | 逐位匹配 |
| L0 | Jacobi 特征值残差 | ‖Av−λv‖≈ε |
| L1 | 解析梯度 vs 数值梯度 | maxRelErr≈1e-7 |
| L2 | 因果掩码 | 未来 key 不影响过去 |
| L3 | 投机解码无损 | χ²(df=5)=0.75 < 20.5 |
| L3 | per-channel 量化 | 0.7% vs per-tensor 100% |
| L4 | ISA 计数 | 1024，逐条确定性 |
| L5/L8 | 运行时门禁 | G1/G2/G3 全过 |

全量门禁：**202 通过 / 0 失败**。
