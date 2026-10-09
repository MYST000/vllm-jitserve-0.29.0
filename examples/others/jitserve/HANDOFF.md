# JITServe 移植交接：固定估计已验证，QRF 接入待验收

## 最新状态：2026-10-08 QRF 中途转为机器迁移

用户已选择复用原版预训练 QRF、明确标记跨模型使用、暂不重训；随后要求先同步文档并迁移到另一台机器。
迁移步骤见 [MIGRATION.md](MIGRATION.md)，支持 GitHub 源码迁移；以下历史章节中的“QRF 未开始”
或“本轮不纳入”只描述先前固定估计阶段，当前状态以本节为准。

- **已写入、尚未验收：** 新增 `vllm/v1/jitserve/prediction.py` 和 `tests/v1/jitserve/test_prediction.py`；
  修改 `vllm/v1/engine/async_llm.py`，在普通提交后启动 QRF，并接入完成、取消和关闭清理。
  `test_scheduler.py` 增加重复预测版本拒绝断言。未重写已验收的策略、分配器或 CPU 复制。
- **资产已下载：** 原作者 `En-2863/jitserve-qrf-length-predictor`，revision
  `1e587124a9671f54f4c659f97bccbbe890b3d8af`，默认 Llama3-8B 模型与配对 TF-IDF 向量器，
  路径 `.agent_tmp/qrf-assets/`。模型约 1.34 GB；0.95 分位及 `int` 转换沿用旧函数。
  这是跨模型应用到 Qwen3.5-27B，不是为它训练/校准的预测器，不使用当前测试任务的真实输出长度。
- **依赖已安装：** quantile-forest 1.4.2、scikit-learn 1.5.1、joblib 1.6.0、scipy 1.18.1、
  threadpoolctl 3.7.0。1.5.1 来自原资产的序列化记录；QRF 扩展兼容性仍待实际预测验证。
- **静态证据：** 新增预测模块/测试及 AsyncLLM 的 Ruff 通过；原 `.agent_tmp/pyright-jitserve.json`
  检查 `vllm/v1/jitserve` 得到 0 errors，不代表 AsyncLLM 全文件类型检查通过。
- **测试未完成：** CPU 全套命令日志 `.agent_tmp/qrf-pytest-first.log` 没有 pytest 汇总，已中断；
  随后的宿主定向测试请求在执行前被用户中止，没有通过记录。未完成原 QRF 函数差分、生命周期回归、
  HTTP/OpenHands QRF 实机验证。未为 QRF 启动新的 GPU 推理服务。
- **默认行为：** 仅配置 `additional_config.jitserve.prediction` 才启用 QRF；原固定估计启动配置未改变。
  初始估计仍由调用方提供，模型或向量器缺失会使启用时加载失败；异步预测错误显式记录并保留初始估计。
- **尚待处理：** 验证实际加载/预测兼容性、取消/晚到/版本与 request ID、模型输出一致性；
  补充 benchmark 的 QRF 实验标签（当前仍写 `caller_fixed`），完成资产/特征 provenance 和真实推理验收。
  graph stage deadline 仍未移植。FlowPilot、OpenHands 核心及 benchmark 在 QRF 这一轮没有新改动。

迁移包另存 `workspace/experiments/jitserve/20261008-qrf-wip-migration.tar.gz`，包含当前源码 overlay、
OpenHands HEAD bundle、旧固定估计验收包和本轮未完成日志；模型权重、QRF 大文件和 Python 环境单独迁移。
旧 `20261008-fixed-estimate.tar.gz` 原样保留，可以恢复最后已验收的固定估计源码。
原 CUDA 12.9 wheel URL 已从 uv 缓存找回，见迁移文档；下方旧归档说明的“URL 未找回”已过时。

更新日期：2026-10-08。供新的上下文窗口直接接续；本文件记录当前事实和后续建议，
后续工作尚未因写入本文件而完成。实际源码、用户最新指示和当前验证结果优先。

## 1. 目标与当前落点

用户最终希望将 JITServe 作为对照组，与现有 OpenHands 和 benchmark adapter 结合进行实验。
已明确要求：使用新建的 vLLM 0.29.0 工作区迁移，模型为 Qwen3.5-27B，
保留主动抢占及真实 CPU KV 换出恢复。不能退化成仅奖励排序或抢占后重算。

**后端、策略对齐、执行层修复及固定估计的 OpenHands 接入已完成本地验证。**
2026-10-08 后续工作补齐当前 HTTP/SSE/Tool，并完成 stock/JITServe 的 1/2/4 并发闭环。
用户明确选择先完成固定估计、并发验证与归档；QRF 预测服务与工作流图 deadline 处理仍未迁移。
它们应分别列为完整 JITServe 系统复现的工作，不能把后端验收等同全部系统验收。

| 状态 | 已确认内容 |
| --- | --- |
| 实现完成 | 新 V1 后端、主动抢占、CPU 换出/恢复、混合 KV 状态、策略/预算、默认关闭的固定估计 benchmark adapter |
| 本地验证完成 | Qwen3.5-27B / TP4、逐 token 恢复对照、HTTP/SSE/Tool、OpenHands 受控语料闭环及 1/2/4 并发 |
| 尚缺证据 | 公开 benchmark 正式成绩、长时间压力稳定性、性能/SLO 收益、QRF/graph 全系统复现 |
| 实验标签 | `jitserve-v1-port`；不能称为旧 V0 引擎逐步等价 |

## 2. 开始前读取与工作约束

1. 读取 `/home/liyachen/.codex/skills/flowpilot-maintainer/SKILL.md` 及其验证说明。
2. 读取所改仓库的 `AGENTS.md`，检查 `git status`、diff；不要清理已有未提交文件。
3. 读取本目录的 [README](README.md)、[对齐记录](POLICY_ALIGNMENT.md)、
   [验证记录](VALIDATION.md)，优先看最前面的 `20261008-integration` 及其后的执行层记录。
4. 涉及集成时读取 `/home/liyachen/workspace/flowpilot/design.md`，尤其 §§0、2.6、5.4、
   7.11、9.2、12、13，以及技能的 `references/architecture-contract.md`。

用户已多次授权真实推理和继续实验；无需重复询问是否开始既定验证。
但必须重新查看 GPU 占用，不能假定上次空闲仍然有效，也不能终止其他任务。
无空闲 GPU 时先完成代码、测试与配置，明确标记真实推理尚待执行。
工具沙箱权限仍按当前会话规则处理；此前 GPU 访问通常需要获准的宿主环境。

- 只在新工作区安装或运行本次 JITServe。旧 `/home/liyachen/vllm` 不参与安装、推理或修改。
- 使用各仓库自己的虚拟环境；在新 vLLM 内使用 `.venv/bin/python` 或 `uv`，禁用系统 Python/bare pip。
- Shell 工具使用 `login=false`，避免登录 shell 注入旧环境。
- 不启动子 agent，除非用户明确要求；不自动提交、合并或发布。
- 失败必须保留证据，不放宽 token 相等断言、不伪造复制成功、不添加自动放行掩盖策略问题。
- 修改前先说明所修行为和最小回归；不要因上下文切换重复重写已验收模块。

## 3. 路径、版本与工作区

| 对象 | 路径或版本 |
| --- | --- |
| 新引擎 | `/home/liyachen/workspace/vllm-jitserve-0.29.0` |
| 工作分支 | `jitserve-v0.29.0` |
| 上游源码基点 | `98dff2a81d747d1dba01a47f939f48c3526d4206`，v0.29.0 |
| 原 JITServe 参照 | `/home/liyachen/workspace/JITServe`，`84584a119b892cf90c65e2b3f5bac67250d39d27` |
| 模型 | `/docker/data/HF_MODELS/Qwen3.5-27B` |
| OpenHands SDK | `/home/liyachen/openhands/software-agent-sdk` |
| benchmark adapter | 上述 SDK 下的 `benchmarks/flowpilot/` |
| FlowPilot | `/home/liyachen/workspace/flowpilot` |
| 实验 trace 根目录 | `/home/liyachen/workspace/experiments/traces` |

新引擎 `.venv`：Python 3.12.9、PyTorch 2.13.0+cu129、Transformers 5.19.0、Triton 3.7.1，
使用对应上游提交的官方 CUDA 12.9 二进制。SCM 显示 dev/dirty 后缀，不意味着改用了旧引擎。
本机驱动使用该 CUDA 12.9 配置；不要顺手升级依赖或重新装默认 CUDA 13 wheel。

当前迁移修改尚未提交。`HEAD` 仍是上游基点，**单独 checkout HEAD 不能恢复移植**。
除已跟踪文件的修改，以下目录含未跟踪实现：

```text
vllm/v1/jitserve/
tests/v1/jitserve/
examples/others/jitserve/
```

`git diff` 不显示未跟踪文件；审查和保存快照必须包含它们。
`.agent_tmp/` 被本地忽略，验证 JSON 和日志不是 Git 中的持久归档，不要清理或覆盖。

## 4. 已实现代码导航

| 文件，相对新引擎根目录 | 职责与接续注意点 |
| --- | --- |
| `vllm/v1/jitserve/policy.py` | SLO 字段校验、三类奖励、观测/缓存、sliding_window 与 length_bin、逻辑批大小 |
| `vllm/v1/jitserve/execution.py` | 原 `_schedule_process_list` 的独立预算 pass；完整页需求、两 token 预留、stop_allocate |
| `vllm/v1/jitserve/scheduler.py` | 普通轮次与 SLO 执行分支、按序抢占/提交、真实分配器、预测更新和状态回收 |
| `vllm/v1/jitserve/connector.py` | 按实际页布局保存/恢复，CPU 预留、私有恢复页、worker 完成计数与取消清理 |
| `vllm/v1/core/sched/scheduler.py` | 共用 `_make_scheduler_output`；Mamba 小块推进可选参数，原生默认行为不变 |
| `vllm/v1/worker/gpu_model_runner.py` | 在状态更新/清零后、Mamba 预处理前加载 CPU 状态 |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | 新加载时序钩子，其他 connector 默认为空操作 |
| `vllm/v1/engine/core.py` | stats 与版本化预测更新 utility |
| `vllm/v1/engine/input_processor.py` | 仅 JITServe 启用时校验请求 SLO |
| `tests/v1/jitserve/test_policy_reference.py` | 从旧仓库 AST 加载真实函数体进行差分；不是手写参考算法 |
| `tests/v1/jitserve/test_scheduler.py` | 真实 V1 allocator/HMA 回归；给定 token 输出，不充当模型推理证据 |

`tests/v1/jitserve/test_jitserve.py` 覆盖策略、connector 和复制等行为；完整清单以
`rg --files tests/v1/jitserve` 为准。
`tools/pre_commit/check_forbidden_imports.py` 的本地修改用于测试 IPC 序列化导入许可，不要误删。

SLO 执行顺序已经修复：按 process_list 预留预算；prefill 每项最多先占 2 token；
首项失败后停止后续分配；资源不够时按列表顺序换出运行请求；按 swapped → running → waiting
提交；新请求按序吃剩余 token；零预算未执行请求仍保留，不复刻旧版丢失队列项的缺陷。
页复制、清零、COW、连接器元数据和计算水位统一走 V1 输出链路。

## 5. 最新验证证据

最新接入证据为 `.agent_tmp/jitserve-validation/20261008-integration/`，详见
[验证记录](VALIDATION.md) 的接入章节。`paired-v2` 两组各 14 个 workflow、56 次 LLM、
42 次真实 Tool；两文档合成语料的官方 Hotpot scorer 全通过，不属于公开 HotpotQA 成绩。
首次评分失败因 SDK 环境缺少已声明的 `ujson`，补装 5.13.0 后通过，失败记录仍保留。

OpenHands 仅修改 `benchmarks/flowpilot/`，FlowPilot 与 SDK 核心未改。
新增 `jitserve.py`、`config.JITServeConfig`、`RecordedLLM` 注入及 runner 接线，默认关闭；
`scripts/validate_jitserve.py` 与本目录 `validate_benchmark.py` 是复现入口。
既有 FlowPilot SDK 56 项、benchmark identity/reuse 14 项均通过，目前无需专用 SDK 副本。
用户允许在需要影响原 FlowPilot 时改用专用副本；后续如扩大修改，应重新评估隔离。

SDK 当前 HEAD 为 `6152623b7b171eb0fe2917d9989425c54c27d789`；默认旧 pin `c4f3ea6`
不匹配其中已提交的 DCS 修改。新增显式支持当前 pin，保留旧 pin；继续执行原 provenance
校验，未放宽核心文件比较。`paired-final` 使用显式当前 pin 并记录 SDK 校验结果；
`paired-v2` 的默认 pin 是发现此问题前的历史配置，实际运行的也是当前 SDK checkout。

以下执行层证据继续有效；本轮没有重写策略、分配器或 CPU 换出恢复。

证据目录：新引擎下 `.agent_tmp/jitserve-validation/20261008-execution/`。

| 证据 | 结果 |
| --- | --- |
| `policy-stock.json` / `policy-jitserve.json` | 自然周期 4 场景，8 个完成请求、486 token 一致 |
| `batch2-stock.json` / `batch2-jitserve.json` | 双序列 6 场景，14 个完成请求、498 token，另有 545 个背景共同 token 一致 |
| `pytest-jitserve-final.log` | 66 passed，含真实 CUDA 字节复制、原函数执行预算 240 组差分 |
| `pytest-native-final.log` | 69 passed：明确选定的原生调度节点及全部 Mamba 对齐测试；不是原生全套通过 |
| `pre-commit-complete.log` / `pyright-final.log` | 适用 pre-commit 全通过；Pyright 0 errors |
| `summary.json` / `source-sha256.json` | 汇总与当时工作区/参照源码摘要；后续修改另建新摘要 |

合计 22 个完成请求、984 个生成 token 与停止原因完全一致。每个 TP rank 实际完成
15 次 D2H、13 次 H2D，两次差额是取消；D2H 834928640 字节，H2D 693633024 字节。
这些是 worker 完成量，不是 scheduler 提交量。每场景结束 CPU 快照、预留字节和 live requests 为 0。

自然周期实验触发两 token 运行 prefill 和 stop_allocate；物理容量 3，逻辑扩批 2→3→4。
水位矩阵覆盖 decode 135/143/151、混合块边界 783/784/785、prefill 128/256/784，
并核验旧页被其他请求使用或恢复到不同页。矩阵会对齐 SLO 计数，不能当自然周期证据。

`20261008-policy/` 是执行修复前记录；`20261007-qwen35/` 的 HTTP/SSE/Tool 证据更早。
当前 HTTP 补验依据 `20261008-integration/http-final.json` 与 `serve-http-final.log`，
不能用上述历史 HTTP 结果替代。最终补验后所有 GPU 都为 1 MiB，进程已释放；
这只是交接时的状态，不是下一轮 GPU 资源承诺。

## 6. 剩余工作与建议顺序

### P0：当前版本 HTTP/SSE/Tool（已补验）

复用 `serve_qwen35.sh` 与 `validate_http.py`，先跑当前代码，不先重写测试。
覆盖 Chat Completions、Responses、流式正文/Tool 参数、非法 SLO 400、并发抢占和断连清理。
若失败，定位执行层变更还是既有前端问题，保留失败日志后修复。

验收：真实模型服务通过现有断言；流关闭后无继续生成和残留请求；保存服务日志、客户端 JSON、
配置及源码摘要。HTTP 日志不能冒充 worker CPU 快照统计，内存回收证据沿用相应真实引擎测量。
上游断连路径不增加 `request_success{finished_reason="abort"}`，不能以该计数作为唯一取消证明。

### P1：OpenHands 最小真实闭环（受控语料已完成，正式数据集待测）

固定估计 adapter 与复现脚本已实现。后续优先使用现有入口，不再从头添加另一套
Agent loop；真实公开数据集/正式采集仍需按对应任务与数据 provenance 运行。

先读实际实现，不假定现有 adapter 已支持 JITServe。重点位置位于 benchmark 的
`src/benchmark_adapters/`：`runner.py`、`sdk_bridge.py`、`config.py`、`workflow_slo.py`、
`predictor_integration.py`、`prediction_export.py`、`tracing.py`，以及 SDK 的 LLM/FlowPilot hooks。
其中已有 predictor 和 workflow SLO 功能，不代表其协议等同 JITServe。

推荐先建立独立、显式启用的 `jitserve-v1-port` 实验配置，复用相同 Agent、Tools、任务和评测器。
基础推理优先使用 `LLM.base_url`；动态 SLO 元数据通过最小 adapter 注入，避免复制 agent loop。
不要默认叠加 FlowPilot 的 admission、retention、reuse 或 DCS，从而混淆对照组变量。
是否通过纯代理、各组启用哪些能力，应在实验配置中明确记录。

必需字段通过 OpenAI `extra_body={"vllm_xargs": {...}}` 传递，不能塞进 messages/Tool 参数：

| 字段 | 契约 |
| --- | --- |
| `jitserve_request_type` | 整数 0 latency、1 throughput、2 collective |
| `jitserve_deadline` | 绝对 Unix epoch 秒；旧 RequestInfo 的相对 deadline 必须在明确的起点转换 |
| `jitserve_output_len` | 正整数预测长度；与 max_tokens、最终真实输出长度分开 |
| `jitserve_collection_id` | 非空字符串；建议映射稳定 workflow/Job，不能把整个 benchmark run 当单一 Job |
| `jitserve_ttft` / `jitserve_tbt` | 秒；latency 两项都要求正数，其他实验也应明确 TBT 配置 |

保留 job、conversation/line、logical request、transport attempt、engine request ID 的映射。
OpenHands 继续拥有真实 Tools、权威历史和 provider tool_call_id；工具完成后提交完整的下一请求。
如使用 FlowPilot，它只有外部 admission/retention/query 权限，不能发送 RESTORE、控制内部抢占或等待 GPU-ready。
当前 CPU SWAP 能力不等于实现了 FlowPilot KV control/target-prefix 协议；不得虚报这些 capability。

验收：同一 benchmark 任务在 stock 和 JITServe 配置下各走至少两轮 LLM + 真实 Tool，
验证输入参数到达引擎、身份关联、取消/失败清理、未启用 adapter 时行为不变，再扩到并发任务。
FlowPilot 路径若有修改，运行技能规定的对应 gateway/identity/Tool 集成回归。

### P2：在线预测与 graph deadline（用户本轮明确暂不纳入）

目前只有初始预测参数和 `jitserve_update_prediction(request_id, output_len, version)` utility，
它位于 EngineCore，不是现成 HTTP 管理端点。应使用引擎实际 request ID；版本严格递增，
陈旧或已结束请求的更新返回 False；排序效果遵循原优先级缓存周期。

参照旧 `jitserve/request_analyzer/prediction.py`、`graph_context.py` 和 `jitserve/server.py`：
旧预测路径加载模型/向量器并使用 0.95 分位输出；graph/collection 处理还调整 stage 与总 deadline。
完整复现需确认模型资产、特征、tokenizer、训练数据隔离和 deadline 起点，然后适配当前异步请求生命周期。
不要跨进程修改旧 RequestInfo，也不要把旧 socket/pickle 协议直接搬成新的公共 HTTP 接口。

明确区分实验口径：调用方固定估计、预测器、仅诊断使用的 oracle。不得用测试任务最终输出长度
充当在线预测。模型资产或目标口径缺失时报告该缺口；不要宣称完整预测器已经移植。
只有要求完整工作流 JITServe 复现时才把 QRF/graph 纳入相应验收，不把它们写成后端修复缺陷。

验收：预测完成、晚到、版本重复、取消、已结束等路径；更新不阻断正常推理；
多轮/多 line 的 collection 与预算起点一致，并与旧函数或明确记录的接口契约对照。

### P3：压力稳定性与公平对照实验（小规模并发已完成）

短程正确性已通过，尚未完成长时间显存/CPU 压力实验。使用独立实验目录逐步增加并发、
prompt 长度和取消率，观测资源抢占、重复恢复、CPU 水位迟滞、取消后回收及内存是否持续增长。
CPU 容量不足当前会显式失败，不能擅自加静默重算掩盖问题。

stock 与 JITServe 固定模型、GPU/TP、精度、上下文、采样、Tools、数据/到达序列和资源预算。
记录 `capacity_limited_rounds`、`allocation_stopped_rounds`、逻辑批大小、worker 容量及每 rank 完成复制量。
为动态扩批设置并记录合理 headroom；容量饱和意味着物理上限影响结果，不能声称无界扩批。

报告任务正确率、workflow SLO goodput、JCT、TTFT/TBT、吞吐、抢占及真实复制成本；
保存重复实验配置与原始指标。不能从当前 984 个 token 的一致性得出吞吐或 SLO 改善结论。
固定估计、在线预测以及 FlowPilot 不同能力组合使用分开的实验标签。

### P4：可恢复源码归档（固定估计阶段已整理）

归档为 `/home/liyachen/workspace/experiments/jitserve/20261008-fixed-estimate.tar.gz`，
附 SHA-256 与展开目录的 `manifest.json` / `RESTORE.txt`。包含本轮及前序 `.agent_tmp`
验证证据、两仓库所有修改/新增源码、patch、依赖版本与模型/扩展摘要。
模型和第三方 wheel 外置；未恢复原始 wheel URL，环境重建仍需要对应 CUDA 12.9 二进制。

稳定后整理完整补丁/源码清单、依赖与 wheel 信息、模型配置摘要、启动参数、测试命令和实验产物。
未跟踪实现和 `.agent_tmp` 证据必须纳入可恢复快照；不能只记录 upstream HEAD。
按用户意图再决定 commit/PR，不把整理交接文件视为发布授权。

## 7. 快速操作命令

以下命令以新引擎根目录为工作目录。GPU 示例编号必须先核对当前资源；实验目录每轮新建。

```bash
cd /home/liyachen/workspace/vllm-jitserve-0.29.0
git status --short
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# CPU 回归；CUDA 字节复制用例会明确跳过。
CUDA_VISIBLE_DEVICES='' VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 \
  JITSERVE_REFERENCE=/home/liyachen/workspace/JITServe \
  .venv/bin/python -m pytest --confcutdir=tests/v1/jitserve tests/v1/jitserve -q

# 静态检查；按实际修改文件运行全套适用 pre-commit hooks。
.venv/bin/python -m pyright --project .agent_tmp/pyright-jitserve.json
git diff --check
```

`JITSERVE_REFERENCE` 必须设置，否则原函数差分会跳过。
CUDA 全套 JITServe 回归可使用空闲单卡替换上面隐藏设备的设置；当时结果为 66 passed。
原生 69 项测试的准确 node 列表见 `pytest-native-final.log` 前部，另加
`tests/v1/core/test_mamba_align_chunk_split.py`；不要把 `-k test_schedule` 当精确筛选，
因为模块名也参与匹配，会误选整个模块。

当前 HTTP 补验的服务配置示例（前台启动，另开终端跑客户端）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 \
  JITSERVE_MAX_NUM_SEQS=1 \
  JITSERVE_ADDITIONAL_CONFIG='{"jitserve":{"schedule_interval":2,"search_strategy":"length_bin","top_k_selection":2,"max_swaps_per_iter":1}}' \
  bash examples/others/jitserve/serve_qwen35.sh

.venv/bin/python examples/others/jitserve/validate_http.py \
  --base-url http://127.0.0.1:8011 \
  --output .agent_tmp/jitserve-validation/NEXT_RUN/http.json
```

这里 `NEXT_RUN` 是需要替换的新实验目录名。模型服务名为 `Qwen3.5-27B-jitserve`；
OpenAI SDK base_url 为 `http://127.0.0.1:8011/v1`，验证脚本 base-url 则不带 `/v1`。
此容量 1 的配置用于稳定触发抢占，不是性能实验推荐配置。
自然周期和边界矩阵的成对 stock/JITServe 命令见 README；先生成 stock，再传 `--baseline` 跑 JITServe。
验证脚本没有 `--baseline` 时运行 stock，不能仅看文件名判断测试的是哪一后端。

## 8. 已知边界与避免重复踩坑

- 支持同步 V1/MRV1、单实例 TP、文本；async、MRV2、PP/DP/CP、speculative/MTP 组合未移植，明确拒绝。
- Qwen3.5 为全注意力与线性注意力混合模型，必须同时保存 attention KV、卷积和循环状态。
  不能把整页 prefix 命中或旧 GPU 页号当作任意水位可恢复证明。
- `--max-num-seqs` 是固定 worker 数组容量；`jitserve.max_batch_size` 是逻辑初始批大小。
  V1 混合页需求和必要分块边界不同于 V0；这些属于架构边界，不应为“完全一致”破坏原生模型状态。
- 旧 `_cur_rank` 未实际递增、集合扩展分支未写入字典：已按实际源码对齐，别擅自补成论文推测行为。
- 原策略 interval=1 冷启动、很小批大小配合低 TBT 可能停滞或缩到非正上限；当前没有自动放行掩盖。
- 旧 idle abort、预测长度强制结束、错误参数调用、零预算丢失请求没有照搬；停止由新 vLLM 负责。
- 全部原生测试不是全绿：已有 Transformers/Pixtral 导入不兼容；只暴露单卡跑 PP2 也会配置失败。
  不要为文本 JITServe 任务盲目升级模型依赖；如扩展多模态需独立解决并重验。
- 无 GPU 时 native DeviceConfig 自动探测可能失败，JITServe CPU fixture 已显式选择 CPU。
  沙箱缺少 GPU/扩展可见性也不代表安装应重做，先核对获准的真实运行环境。
- 两 token 预留改变了执行水位，历史自然周期快照位置和字节数不可硬套到新版本。
  真实输出一致、正确水位和 worker 复制完成是验收依据。

## 9. 发给新上下文的启动提示

```text
请使用 flowpilot-maintainer 技能，先阅读
/home/liyachen/workspace/vllm-jitserve-0.29.0/examples/others/jitserve/HANDOFF.md，
以及该目录 README.md、POLICY_ALIGNMENT.md、VALIDATION.md，核对当前工作区后继续移植。
后端、策略及执行层修复已通过真实 Qwen3.5-27B / TP4 验证，不要从头重写。
固定估计的 HTTP/SSE/Tool 和 OpenHands 1/2/4 并发闭环已完成；复现入口为 validate_benchmark.py。
后续按用户指定推进公开 benchmark、长期压力或在线 QRF/graph；不能把已完成模块重新列为缺陷。
只使用独立 vllm-jitserve-0.29.0 及其 .venv，保留主动抢占和真实 CPU 换出恢复。
保留全部未提交代码及实验产物，先核对 GPU 占用，不自动提交或发布。
```
