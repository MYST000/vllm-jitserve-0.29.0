# 验证记录：2026-10-08

## 最新增量：QRF 初步接入，验证未完成

本次在完成固定估计阶段后开始 QRF 移植，随后按用户要求转为跨机器迁移准备。
新资产及源码见 [HANDOFF.md](HANDOFF.md) 顶部；复现步骤见 [MIGRATION.md](MIGRATION.md)。

| 项目 | 当前证据 |
| --- | --- |
| 原版模型与向量器下载 | 固定 HF revision 已下载，SHA-256 见 `migration/environment.json` |
| Ruff | `prediction.py`、`async_llm.py`、`test_prediction.py` 通过 |
| Pyright | 原项目配置覆盖 `vllm/v1/jitserve`，0 errors；未覆盖完整 AsyncLLM |
| CPU 全套 pytest | 中断，无最终汇总；日志为 `.agent_tmp/qrf-pytest-first.log`，不能计为通过 |
| 宿主定向 pytest | 执行请求被用户中止，无执行结果 |
| 原版 QRF 数值差分 / 新生命周期用例 | 已写测试，尚无完成证据 |
| QRF HTTP / OpenHands / GPU | 未运行 |
| graph deadline / 生产效果 | 未实现 / 无证据 |

因此 QRF 只能表述为 **implementation in progress / local verification incomplete**。
以下既有结果继续作为固定估计源码快照的证据，不能转记为当前 QRF 版本的回归通过。

状态：**执行层修复、HTTP 补验及固定估计 OpenHands 工作流本地验证通过**。
本轮 22 个完成请求、984 个生成 token 及停止原因与原生 V1 基线一致，
背景序列另有 545 个共同前缀 token 核对一致。
这是 `jitserve-v1-port` 的本地正确性证据。新增受控语料的 benchmark adapter 闭环，
仍无公开 benchmark 成绩或性能收益结论。

## 固定估计工作流接入：20261008-integration

用户选择固定输出长度估计、并发验证与归档；在线 QRF 与 graph stage deadline 不纳入本轮。
OpenHands 的改动仅在 `benchmarks/flowpilot/`：默认关闭的 `JITServeConfig`、
`RecordedLLM` 传输注入、runner 身份与任务预算、验证脚本和回归。
FlowPilot、OpenHands Agent/Conversation/LLM 核心及 vLLM 策略/复制实现均未修改。
vLLM 增加默认关闭的 metadata-only ingress 日志和成对验证入口。

| 验证 | 结果 |
| --- | --- |
| HTTP/SSE/Tool | Chat、Responses、正文/Tool 分片、非法 SLO 400、真实本地 Tool、抢占和断连清理通过 |
| JITServe CUDA 回归 | 67 passed，包含真实复制及原版函数差分 |
| JITServe 隐藏 GPU 回归 | 66 passed、1 skipped；跳过项为 CUDA 字节复制 |
| SDK transport/identity/config 回归 | 33 passed；新增覆盖关闭 adapter、两 backend、重试及固定 deadline |
| 原 FlowPilot SDK 回归 | 56 passed，涵盖既有 adapter 路径 |
| 原 benchmark identity/reuse 配置 | 14 passed |
| 静态检查 | 适用 pre-commit、Ruff、Pyright、compile/import、diff 检查分别留存日志 |

成对配置为 Qwen3.5-27B、TP4、BF16、MRV1、同步 eager、APC/Mamba align、
16384 上下文、worker 容量 4、token budget 512、每 rank KV 预算 6442450944 bytes。
JITServe 逻辑初始批大小 2，sliding_window、interval 4、top-k 2、max swaps 1、
CPU 总预算 8 GiB。调用方输出估计 128，输出上限 512，workflow SLO 120 秒，
TTFT/TBT 参数均为 10 秒；deadline 起点为环境准备前的任务启动时刻。
每个 workflow 独立 Job/collection，重试与 Tool 轮次不重置起点。

`paired-v2` 在每个并发等级 1/2/4 各运行两轮：

| 观察 | stock-v1 | jitserve-v1-port |
| --- | --- | --- |
| workflow / LLM / 实际 Tool | 14 / 56 / 42 | 14 / 56 / 42 |
| 官方 scorer joint EM | 14 个均为 1 | 14 个均为 1 |
| 120 秒内完成 | 14 / 14 | 14 / 14 |
| 平均 workflow JCT | 36.496 秒 | 36.663 秒 |
| 最大 workflow JCT | 57.428 秒 | 57.743 秒 |

上述任务是两文档合成语料，经现有 Hotpot adapter 执行真实 search/read_document 并用
固定官方 scorer 评分；不是 HotpotQA 公开数据集准确率。固定任务、到达批次与资源，
但样本很小，不能据此声称性能或 SLO 改善。本组未触发抢占，CPU 恢复仍由独立真实引擎
矩阵及 HTTP 抢占测试证明；HTTP 指标不能冒充 worker CPU 快照/完成复制计数。
保存每阶段原始 `/metrics`、各任务耗时和 SLO、完整 benchmark 轨迹及评分。

`engine-identity.json` 按返回的外部 ID 关联 EngineCore 实际内部 ID，逐字段核验收到的
SLO 与客户端注入值；stock 确认不发送调度字段。trace 仅含 identity 和白名单 SLO，
不包含 prompt、Tool payload 或鉴权头，默认关闭。没有新增 RESTORE 或 FlowPilot KV API。

接入时发现默认 SDK pin `c4f3ea6` 落后于当前 checkout `6152623`，差异是既有 DCS 提交。
保留旧 pin，并允许显式选择当前已测试提交；原核心源码与 editable 安装校验保持执行。
`paired-v2` 保留发现问题前的旧默认配置，实际运行当前 checkout；`paired-final` 使用
显式当前 pin，并在执行前生成 `sdk-provenance.json`，作为最终复现配置。
该最终轮次在 1/2/4 并发各运行一轮，两组各 7 个 workflow、28 次 LLM、21 次真实 Tool，
全部评分正确并在 120 秒 SLO 内完成。两次完整配对合计 42 个 workflow、168 次 LLM、
126 次真实 Tool；第一次依赖失败的试跑不计入这个合计。

最终 HTTP 补验结果为 `http-final.json`，启动参数见 `http-final-launch.json`。
`serve-http-final.log` 在 CUDA 同步完成后记录四个 TP worker 各完成一次 D2H 和 H2D，
每 rank 每方向 51380224 bytes。断连后 running/waiting 均为 0，生成停止且无正常完成事件。
测试服务退出后，六张 GPU 均回到 1 MiB，记录于 `gpus-final.txt`。

失败记录保留：沙箱拒绝 CUDA/回环网络、第一次官方评分缺少 `ujson`、测试的 trace
字段冲突与 token-limit 字段断言、静态检查失败。SDK 环境补齐已有声明的 `ujson==5.13.0`；
没有更换 scorer、伪造推理/Tool 成功、放宽评分或 token 一致性断言。

当前边界：仅同步非流式 benchmark actor 接入；服务的 SSE/Responses 验收是另一组证据。
多 line/delegation、在线预测、graph 分段预算、正式公开数据集、长时间资源压力与公平性
收益仍待单独验证。CPU SWAP 支持不代表已实现 FlowPilot KV control/target-prefix 能力。

可恢复源码与证据归档路径为
`/home/liyachen/workspace/experiments/jitserve/20261008-fixed-estimate.tar.gz`，
旁附 SHA-256；展开目录的 `manifest.json` 包含两个仓库基点和文件摘要，`RESTORE.txt`
说明如何覆盖恢复未提交源码。归档包含全部新增文件、tracked patch、历次验证日志与配置。
第三方 wheel 与模型权重未打包；记录依赖版本、模型配置及实际扩展二进制 SHA-256。
已安装 vLLM distribution 没有原始 wheel URL 元数据，不能宣称完全离线重建环境。

## 执行层修复后：20261008-execution

本轮补齐原版 SLO 执行预算、两 token prefill 预留、`stop_allocate`、按序资源抢占，
以及 swapped → running → waiting 提交顺序。实现与残留边界见
[策略及执行对齐记录](POLICY_ALIGNMENT.md)。原生 V1 输出构造提取为共同方法，
继续统一处理 block copy/zero、连接器元数据、运行进度与完成事件。

| 验证 | 结果 |
| --- | --- |
| 原函数执行预算对照 | 240 组，每组 12 候选，检查预留、换出、提交顺序和剩余预算 |
| JITServe 回归 | 66 passed，包含原源码对照、真实 V1 分配器和 CUDA 字节复制，无跳过 |
| 原生针对性回归 | 69 passed，包含全部 Mamba 对齐、抢占/恢复、优先级、输出和取消测试 |
| 自然周期真实模型 | 4 场景通过，8 个完成请求、486 token 与本轮新 stock 基线一致 |
| 两 token 预留 | 扩批场景实测 2 个 SLO 轮次的运行 prefill 以 2 token 推进 |
| 停止分配 | 扩批场景实测 3 个轮次触发；小请求不补位、按序换出另有物理分配器测试 |
| 双序列恢复矩阵 | 6 场景通过，14 个完成请求、498 token，以及 545 个背景共同 token 一致 |
| 清理 | 每场景结束 CPU 快照、预留字节、live requests 均为 0 |

自然周期配置与上一轮相同：TP4、物理容量 3、token budget 130、上下文 4096、
同步 MRV1、eager、BF16、APC、Mamba align。逻辑扩批 2→3→4；实际保存水位变为
784、1296（prefill）和 529（decode），这是执行分配修复后的变化。
预算与真实 token 分配逐轮保存在 `policy-jitserve.json` 的 `batches` 中。

边界矩阵使用容量 2、budget 129；decode 重复水位 135/143/151，块边界 783/784/785，
prefill 水位 128/256/784，另有换出后取消。每次恢复均验证已释放页被使用或恢复到不同页。
两组实验均重新生成 stock 基线，不复用上一轮模型输出。

每个 TP rank 合计实际完成 **15 次 D2H、13 次 H2D**，两次差额对应取消；
实际 D2H **834928640 字节**、H2D **693633024 字节**。这是 worker 完成计数，
不是调度器提交计数；H2D 可以跳过恢复已不需要的历史检查点。

证据目录 `.agent_tmp/jitserve-validation/20261008-execution/` 包含两对基线/移植结果、
`pytest-jitserve-final.log`、`pytest-native-final.log`、检查日志及源码摘要。
适用 pre-commit 检查全部通过（含 Ruff、mypy、Markdown 与导入检查），Pyright 0 errors；
验证进程已退出，6 张 GPU 均回到 1 MiB 显存占用。检查日志为 `pre-commit-complete.log`
和 `pyright-final.log`，汇总与摘要为 `summary.json`、`source-sha256.json`。
本轮没有重跑 HTTP/SSE/Tool 验证，也没有 OpenHands/benchmark 工作流或性能验收。

失败尝试同样保留：CPU 隐藏设备时原生配置不能推断设备；广泛原生回归遇到
Transformers 的 `PixtralRotaryEmbedding` 缺失及仅暴露单卡时的 PP 配置失败。
之后按明确测试节点运行相关原生回归，不将这些未通过的多模态测试计入 69 项结果。
新增测试的入场准备也曾因 native prefill 排序和块对齐失败，修正为先完成 decode 入场，
再以可容纳一个混合块的预算建立运行 prefill；未修改策略断言或 token 相等标准。

## 策略对齐后：20261008-policy

以下为执行层修复前的历史记录，其运行水位和复制字节不能替代本轮证据。

实现及与旧执行器的边界见 [策略对齐记录](POLICY_ALIGNMENT.md)。参照原 JITServe
`84584a119b892cf90c65e2b3f5bac67250d39d27` 的实际函数，不引入未执行的分组轮转
或集合需求扩展。原 `/home/liyachen/vllm`、JITServe、FlowPilot、OpenHands 工作区未改动。

| 实验 | 本轮结果 |
| --- | --- |
| 自然周期，间隔 4 | 滑窗开放上界、length_bin 换出恢复、扩批 2→3→4、换出后取消均通过 |
| 自然周期输出 | 8 个完成请求、486 token 与新启动的 stock 基线一致 |
| 固定容量 | worker 容量 3；扩批场景实际触及容量 3 次，记录 `capacity_limited_rounds=3` |
| 扩批时实际保存水位 | 912、1296（prefill）和 1027（decode），四个 worker 均实际 D2H/H2D |
| 双序列 decode 重复恢复 | 135、143、151 token；另一序列持续生成 |
| 混合块边界恢复 | 783、784、785 token，恢复后计算下一 token |
| prefill 连续恢复 | 128、256、784 token，恢复后继续下一段 128 token |
| 双序列输出 | 14 个完成请求、498 token，与归档的同配置 stock 基线一致 |
| 取消与资源回收 | 两组取消均通过；每个场景结束 CPU 快照、预留、live requests 均为 0 |

自然周期测试的执行配置为物理容量 3、token budget 130、上下文 4096、TP4、eager、
同步 V1、temperature=0、seed=42。滑窗扩批初始逻辑批大小为 2，其余场景为 1，
top-k 2、max swaps 1、显式 TBT 10 秒。测试不修改调度计数来制造自然周期决策。

边界矩阵使用物理容量 2、token budget 129、length_bin、间隔 2、top-k 2、max swaps 1。
它会对齐注入时的 SLO 周期，用于隔离指定水位的恢复行为，不作为自然周期证据。
背景请求必须实际生成首 token 后才开始前景水位计量，不能假设一次 `engine.step()`
一定完成入场。此矩阵复用 `20261007-qwen35/batch2-stock.json`：原生 V1 执行代码、
模型、TP、采样和 token budget 未变，脚本校验模型、TP、块大小及并发配置。

两组成功实验合计，每个 TP rank **15 次实际换出、13 次实际恢复**；两次差额对应取消。
每 rank 实际 D2H **899153920 字节**、H2D **719323136 字节**。
H2D 可跳过恢复已不需要的旧循环检查点，所以字节数不同。统计来自 worker 完成计数，
不是 scheduler 提交计数。双序列矩阵还确认释放页被其他请求使用或恢复到不同页。

最终检查：原函数差分与真实 V1 分配器/CUDA 复制回归 `58 passed`；全套适用 pre-commit
检查（含 Ruff、mypy、文档格式与导入检查）通过；Pyright 0 errors。
源函数差分包含 36 请求 × 80 次奖励/预测更新、两种分组策略各 60 轮状态变化及 CPU 迟滞。
未设置 `JITSERVE_REFERENCE` 时该组会跳过；本轮明确设置了原仓库路径，没有跳过。

成功证据位于 `.agent_tmp/jitserve-validation/20261008-policy/`：

- `policy-stock-v3.json`、`policy-jitserve-v3.json`：自然周期基线及移植版。
- `batch2-jitserve-v2.json`：精确水位矩阵，基线为上文的归档文件。
- `pytest-cuda-final.log`、`pre-commit-final.log`：最终回归和检查。
- `summary.json`、`source-sha256.json`：汇总与最终工作区/参照源码摘要。

失败尝试的日志保留：首次基线脚本错误处理 tokenizer 的 BatchEncoding；初始批大小 1
的扩批场景碰到原版冷启动/TBT 缩批行为，改为初始 2、容量 3 的明确扩批实验；
首次边界矩阵假设背景请求一步进入 decode，改为验证真实进度。没有通过修改策略收益、
跳过复制或放宽 token 相等断言消除失败。本轮未重新运行 HTTP 全套验证，以下 HTTP
记录属于对齐前版本。

## 策略对齐前的归档说明

以下章节来自 `20261007-qwen35/` 的早期移植验证：30 个完成请求、1094 token，以及
背景序列 545 个共同 token 一致，包含当时的 HTTP/SSE/Tool 验证。
它们保留为历史证据，不能替代上面的策略对齐后验证。

## 环境与对照口径

- 独立源码：`/home/liyachen/workspace/vllm-jitserve-0.29.0`，上游 v0.29.0 提交
  `98dff2a81d747d1dba01a47f939f48c3526d4206`；本目录独立 `.venv`。
- Python 3.12.9、PyTorch 2.13.0+cu129、Transformers 5.19.0、Triton 3.7.1；
  使用该提交官方 CUDA 12.9 二进制。运行版本带 SCM dev 后缀，源码基点仍为上述提交。
- GPU 0–3：4 张 RTX 4090，TP4、BF16、同步 V1、eager、prefix caching、Mamba align。
- 本地模型 `/docker/data/HF_MODELS/Qwen3.5-27B`，16 层全注意力、48 层线性注意力。
  11 个权重分片实际加载；索引总字节数 `55562872800`，每 rank 权重约 12.89 GiB。
- 模型配置 SHA-256：
  `f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c`。
- GPU memory utilization 0.85，CPU 换出预算为 TP 合计 8 GiB。
  不同启动的自动显存 profile 结果不作为固定 KV 字节预算的性能对照。

原生基线使用同一新源码树的原生 V1 调度器，不启用 JITServe scheduler/connector。
两端分别启动进程，使用相同模型、输入和执行参数，temperature=0、seed=42。
恢复完全在引擎内部执行，没有外部 RESTORE RPC；测试未修改 EOS 或提供伪造模型输出。
原 `/home/liyachen/vllm` 未参与此安装和推理。

## 真实引擎对照

基础测试使用 1024 上下文、每批最多 1 条序列、128 token 预算。
4 个 rank 均实际完成 1 次 D2H 和 1 次 H2D，每 rank 每方向 `51380224` 字节；
2 个完成请求的 98 个生成 token 和停止原因与基线相同。

扩展测试使用 4096 上下文，实际混合块大小 784 tokens。
单序列矩阵使用 128 token 批预算；双序列矩阵使用 129，其中持续生成序列每步占 1 token。
各场景开始前清空 prefix cache；每个矩阵均有 14 个完成请求、498 个生成 token 与基线一致。

| 场景 | 单序列保存水位 | 双序列保存水位 | 每 rank 换出 / 恢复（每个矩阵） |
| --- | --- | --- | --- |
| 连续 decode 抢占 | 47、55、63 | 135、143、151 | 3 / 3 |
| 块边界之前 | 783 | 783 | 1 / 1 |
| 块边界处 | 784 | 784 | 1 / 1 |
| 块边界之后 | 785 | 785 | 1 / 1 |
| prefill 中途连续抢占 | 128、256、784 | 128、256、784 | 3 / 3 |
| 换出后取消全部请求 | 47 | 135 | 1 / 0 |

Decode 恢复步骤各只计算下一 token；prefill 恢复后继续下一段 128 tokens。
两种矩阵都检查了每个 TP worker 的实际完成计数和字节数。
每个场景结束后，各 worker CPU 快照、调度器 CPU 预留和 live requests 均为 0。
取消最后一批请求后，清理不依赖新推理请求到来。
双序列测试确认实际调度批中有 2 条序列，并比较持续生成序列与基线的共同 token 前缀。

基础测试加两个矩阵，每 rank 合计实际完成 **21 次换出、19 次恢复**；
差额对应两个主动取消的已换出请求。每 rank 实际 D2H `1181745152` 字节、
H2D `1001914368` 字节，四个 rank 的记录相同。
H2D 只加载恢复时仍需要的循环检查点，实际字节数可小于调度器按完整快照记录的提交字节数。
这些是 worker 完成记录，不用提交计数冒充实际恢复。

物理有效性检查观察到已释放 GPU 块被其他真实请求复用，以及部分恢复重新分配目标块。
HMA 不同组覆盖同一物理分配，未缓存块按 LIFO 复用，所以恢复页号相同不表示旧内容仍有效。
`blocks_reused_by_other_requests` 按调度步骤累计，同一块多次使用会重复计数。
CUDA 单测还直接覆盖原 GPU 字节后恢复到其他页，并逐字节核对。

## HTTP、流式响应与 Tool

独立 `vllm serve` 使用真实多进程服务、TP4、8192 上下文、512 token 批预算、
最大序列数 1、schedule interval 1，reasoning parser 为 `qwen3`，Tool parser 为 `qwen3_coder`。
该序列上限与间隔用于可重复触发抢占，不是吞吐推荐配置。

| 检查 | 实测结果 |
| --- | --- |
| 缺少 SLO / 非法预测长度 | 均为 HTTP 400，随后合法请求正常推理 |
| Chat Completions 普通 / SSE | 文本、停止原因、usage 一致，收到 `[DONE]` |
| Tool Call 普通 / SSE | 函数名、JSON 参数一致，调用 ID 与片段拼接有效 |
| Tool 往返 | 实际执行本地整数加法，按 `tool_call_id` 返回结果，模型正常生成后续回复 |
| Responses 普通 / SSE | 普通响应 completed；流式返回正确文本和 `response.completed` |
| 并发 HTTP 抢占 | preemption 增加 1，各 TP rank 均实际 D2H/H2D，文本和停止原因与无竞争请求一致 |
| SSE 断连 | 运行请求由 1 变为 0，仅生成 1 token 后停止，等待请求为 0，无正常完成事件 |

上游前端 `OutputProcessor.abort_requests` 直接删除请求状态，
这条路径不增加 `request_success{finished_reason="abort"}`，本轮实测增量为 0。
断连验收使用运行请求归零、生成停止和无正常完成事件；不把该计数当作取消证明。
引擎矩阵另外核对了换出请求取消后的每 rank CPU 副本和预算回收。

## 回归和调试记录

| 检查 | 结果 |
| --- | --- |
| `tests/v1/jitserve`，GPU 可见 | **45 passed**，包含真实 CUDA 页复制 |
| 同一测试集，隐藏 GPU | 44 passed、1 skipped |
| 上游 input processor 回归 | 5 passed（2026-10-07） |
| 验证脚本 Ruff check / format、pre-commit Ruff、compileall | 通过 |
| 新模块 Pyright / 依赖检查 | 先前 0 errors、0 warnings / 236 个包兼容；本轮未更改引擎实现和依赖 |

本轮修正了验证脚本的 Transformers 模板返回格式、HMA 跨组物理块统计，以及取消指标口径。
失败日志保留；没有修改推理结果、屏蔽模型错误或添加静默重算。
先前小模型目录缺权重、GPU 被占用的启动失败不计入通过结果。

## 可复现入口与原始证据

命令见 [README](README.md#真实推理验证)，脚本为 `validate_swap.py`、`validate_matrix.py`、
`validate_http.py`。双序列矩阵在两端均加 `--max-num-seqs 2`，使用独立结果文件。

本工作区忽略目录 `.agent_tmp/jitserve-validation/20261007-qwen35/` 保留：

- `stock.json`、`jitserve.json`：基础结果，原生基线于 10 月 7 日暂停前完成。
- `matrix-stock.json`、`matrix-jitserve.json`：单序列矩阵。
- `batch2-stock.json`、`batch2-jitserve.json`：双序列矩阵和持续生成序列对照。
- `http.json`、`http-client-r2.log`、`http-server.log`：通过的 HTTP 结果和 TP worker 日志。
- `summary.json`、`environment.json`：结果汇总、源码与环境指纹。
- `pytest-cuda.log` 和各次启动、失败、通过日志。

## 尚未覆盖

尚未验证默认 16 序列并发、模型完整长上下文、图片/视频、异步调度、Model Runner V2、
PP/DP/CP、speculative/MTP、worker 故障、长期压力和显存不足触发的被动抢占端到端场景。
OpenHands / benchmark adapter、新迁移后端的 FlowPilot 三组件工作流和旧 QRF 预测服务尚未接入。
原版策略差异见 [README](README.md#与旧仓库的区别)。正式实验应使用 `jitserve-v1-port` 标签，
固定资源预算后再测吞吐、TTFT/TBT 和 SLO goodput；本轮结果不能外推为生产或性能保证。
