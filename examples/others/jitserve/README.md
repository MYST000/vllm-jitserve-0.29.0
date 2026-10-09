# JITServe → vLLM 0.29.0 移植

**迁移到其他机器：** 见 [MIGRATION.md](MIGRATION.md)，包含 GitHub 上传范围、环境重建、
OpenHands 配套源码和模型资产迁移。最新 QRF 初步接入仍待验收，见 [HANDOFF.md](HANDOFF.md) 顶部；
下文的真实验证结果属于先前固定估计阶段。

目标模型：本机 `/docker/data/HF_MODELS/Qwen3.5-27B`。
2026-10-08 执行层修复已通过 Qwen3.5-27B / TP4 真实推理、主动抢占与 CPU 换出恢复验证。
当前版本已补验 OpenAI HTTP/SSE、Tool Call，并接入默认关闭的 OpenHands benchmark adapter。
固定估计的 stock/JITServe 小规模真实工作流与 1/2/4 并发已通过，详细范围见
[验证记录](VALIDATION.md)。**这些是受控语料接入证据，尚无公开 benchmark 或吞吐/SLO 收益结论。**

## 独立源码与环境

- 目录：`/home/liyachen/workspace/vllm-jitserve-0.29.0`
- 上游：`vllm-project/vllm`，标签 `v0.29.0`，提交
  `98dff2a81d747d1dba01a47f939f48c3526d4206`
- 工作分支：`jitserve-v0.29.0`；迁移修改保留在工作区，尚未提交。
- 环境：本目录 `.venv`，Python 3.12、PyTorch 2.13.0+cu129。
- CUDA Python 绑定：`cuda-python==12.9.7`、`cuda-bindings==12.9.9`。
- 二进制：该上游提交的官方 CUDA 12.9 wheel；未从旧 vLLM 复制扩展。
- 原 `/home/liyachen/vllm`、JITServe、FlowPilot、OpenHands 不参与此安装。

修改后的源码可能带 setuptools-scm 的 dirty/dev 版本后缀；基线以上述提交为准。
重建环境时必须同时固定源码、CUDA wheel 和 PyTorch CUDA 版本。
本机 570 驱动不能使用默认 CUDA 13.0 PyTorch；当前选择 CUDA 12.9。

## 已接入的执行链路

| 部分 | 新位置/行为 |
| --- | --- |
| SLO | `vllm/v1/jitserve/policy.py`，latency/throughput/collective 奖励密度与长度分组 |
| 调度 | SLO 轮次按原版顺序预留预算、停止分配并提交；抢占均保存 CPU 快照 |
| 换出 | `JITServeConnector.handle_preemptions`，在释放页被覆写前同步完成 D2H |
| 恢复 | 分配私有目标页 → 清零 → H2D → Mamba 预处理复制 → 正常前向 |
| 进度 | 使用保存的 `num_computed_tokens`，包括不足一个完整块的进度 |
| 混合状态 | 按每组实际物理页布局复制注意力 KV、卷积状态、循环状态 |
| CPU 预算 | 按真实页字节数乘 TP rank 数预留，加载/丢弃完成后才释放 |
| 错误 | 缺少必要页、容量耗尽、恢复进度不匹配显式报错，无静默重算 |
| 取消 | 删除调度状态与 CPU 副本，完成清理步骤后释放预留 |
| 预测更新 | EngineCore utility 更新版本化预测，不跨进程修改旧 RequestInfo 对象 |

仅启用 JITServe 调度器时要求 SLO 参数；普通 vLLM 请求入口保持原行为。
普通连接器的新增预处理加载钩子默认不执行任何操作。

支持范围为同步 V1 调度、V1 model runner、单实例 TP。
Async scheduling、Model Runner V2、PP/DP/CP 和 speculative/MTP 的状态协议尚未迁移，
相关组合会明确报错。当前 SLO 执行路径只支持文本，带 encoder 输入的请求明确报错。

### 与旧仓库的区别

这是 V1 引擎上的移植，实验中应标记为 `jitserve-v1-port`，不可声称逐调度步复现旧引擎。
奖励公式、集合聚合、top-k 长度窗和 CPU SWAP 是移植对象；
V1 的块分配、chunked prefill、停止判断和实际推理使用 0.29.0 实现。

- 核心策略已按原仓库 `84584a1` 的实际执行路径对齐，见
  [策略对齐记录](POLICY_ALIGNMENT.md)：奖励观测、优先级缓存、SLO 轮次、开放上界、
  长度分组待抢占队列、候选注入/激活和 TBT 动态批大小。
- `jitserve.max_batch_size` 表示逻辑初始批大小，默认等于 `--max-num-seqs`。
  后者是 V1 worker 的固定执行容量；需要动态扩批空间时，将其设得更大。
  饱和时计入 `capacity_limited_rounds`，不能将这样的运行声称为无容量限制的旧版复现。
- 已补齐两 token prefill 预留、`stop_allocate`、按序资源抢占和换入优先提交。
  V1 负责实际混合页布局与必要分块边界，统计 `allocation_stopped_rounds`。
  原代码没有实际调用分组轮转，集合需求扩展分支也未生效。
- 长度预测只影响排序；不移植 `real_output_len` 强制终止或覆盖 EOS 的逻辑。
- 原版 QRF 资产已下载，异步 utility 接入已写入，尚未完成验收，默认关闭；
  当前已验证路径仍由调用方提供固定长度估计。旧 `/generate` 与 graph deadline 尚未迁移。
- CPU 复制当前为同步实现，未宣称旧版吞吐/SLO 效果。
  与原 JITServe 的执行差异需作为实验口径明确记录。

## Qwen3.5-27B 配置检查

```bash
.venv/bin/python examples/others/jitserve/inspect_qwen35.py
```

该命令不导入推理引擎、不加载权重、不使用 GPU。
本地配置为 16 层全注意力、48 层线性注意力，11 个权重分片均存在，
索引记录权重总计 55,562,872,800 字节。检查只证明配置和文件存在，不证明权重校验或推理通过。

本机 RTX 4090 24 GiB 卡已验证 TP4；分片后的权重之外还需要激活、KV 和运行时显存。
启动脚本默认 8192 上下文、BF16、prefix caching、Mamba align、eager execution，
这些是初次验证参数，并非论文实验或模型完整长上下文配置。

只检查模型文件并打印启动命令，不启动服务：

```bash
bash examples/others/jitserve/serve_qwen35.sh --dry-run
```

## 真实推理验证

以下命令会实际使用 GPU。将示例编号替换为已分配的四张空闲卡。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1

.venv/bin/python examples/others/jitserve/validate_swap.py \
  --model /docker/data/HF_MODELS/Qwen3.5-27B \
  --tensor-parallel-size 4 --output /tmp/qwen35-stock.json

.venv/bin/python examples/others/jitserve/validate_swap.py \
  --model /docker/data/HF_MODELS/Qwen3.5-27B \
  --tensor-parallel-size 4 --baseline /tmp/qwen35-stock.json \
  --output /tmp/qwen35-jitserve.json
```

两个进程使用相同模型、TP 和执行参数；后者插入高优先级请求触发主动抢占。
该换出测试使用 `length_bin`、间隔 2、top-k 2、每轮最多一个分组待抢占项。
验收必须同时满足：逐 token 与停止原因一致、每个 TP worker 均实际完成 D2H/H2D、
CPU 快照与预算回收。调度器提交计数与 worker 完成计数分别记录。
该脚本是正确性测试，不是性能基准；失败会直接断言，不能用提交计数代替恢复证据。

扩展场景覆盖重复 decode 抢占、模型实际块大小的前后水位、prefill 中途抢占和换出后取消：

```bash
.venv/bin/python examples/others/jitserve/validate_matrix.py \
  --model /docker/data/HF_MODELS/Qwen3.5-27B --output /tmp/matrix-stock.json
.venv/bin/python examples/others/jitserve/validate_matrix.py \
  --model /docker/data/HF_MODELS/Qwen3.5-27B \
  --baseline /tmp/matrix-stock.json --output /tmp/matrix-jitserve.json
```

两端均加 `--max-num-seqs 2` 可验证另一条序列持续生成时的恢复，另用一对结果文件。
两种配置均保持 TP4；`max_num_seqs` 是每个调度批的序列上限。
脚本观察实际计算水位、worker 完成字节数及重新分配/跨组复用的物理块。
HMA 缓存组覆盖同一物理分配，所以页号相同不等于内容仍然有效。
此边界矩阵会在注入请求时对齐 SLO 调度计数，以隔离指定水位的恢复行为；
自然调度轮次与滑窗扩批使用 `validate_policy.py` 验证，不能混淆两种证据。

```bash
.venv/bin/python examples/others/jitserve/validate_policy.py \
  --model /docker/data/HF_MODELS/Qwen3.5-27B --output /tmp/policy-stock.json
.venv/bin/python examples/others/jitserve/validate_policy.py \
  --model /docker/data/HF_MODELS/Qwen3.5-27B \
  --baseline /tmp/policy-stock.json --output /tmp/policy-jitserve.json
```

然后启动独立 OpenAI 服务：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/others/jitserve/serve_qwen35.sh
```

默认地址 `http://127.0.0.1:8011/v1`，模型名 `Qwen3.5-27B-jitserve`。
`JITSERVE_TP`、`JITSERVE_PORT`、`JITSERVE_MAX_MODEL_LEN`、`JITSERVE_GPU_UTIL` 等变量可覆盖默认值。
`JITSERVE_ADDITIONAL_CONFIG` 可覆盖完整 JSON 调度配置。
HTTP 验证需设 `JITSERVE_MAX_NUM_SEQS=1` 和
`JITSERVE_ADDITIONAL_CONFIG='{"jitserve":{"schedule_interval":2,"search_strategy":"length_bin","top_k_selection":2,"max_swaps_per_iter":1}}'` 后启动服务，
以便两个并发请求可重复触发抢占；它们不是性能推荐参数。

```bash
.venv/bin/python examples/others/jitserve/validate_http.py --output /tmp/jitserve-http.json
```

验证包括 Chat Completions / Responses 流式请求、非法 SLO 的 HTTP 400、
Tool 参数分片与真实本地加法执行、并发请求抢占和断连后停止生成。
上游前端取消路径不增加 `request_success{finished_reason="abort"}`，
因此断连检查使用运行请求归零、生成停止和无正常完成事件；不把该计数当作取消证明。

## 请求参数与后续 adapter

OpenAI 客户端通过 `extra_body={"vllm_xargs": {...}}` 传入：

```python
import time

extra_body = {
    "vllm_xargs": {
        "jitserve_request_type": 1,  # 0 latency / 1 throughput / 2 collective
        "jitserve_deadline": time.time() + 120,  # 绝对 Unix 时间，秒
        "jitserve_output_len": 512,  # 预测值，不是 max_tokens/真实长度
        "jitserve_collection_id": "job-1",
    }
}
```

Latency 请求还必须提供正数 `jitserve_ttft`、`jitserve_tbt`，单位秒。
离线 `SamplingParams` 使用相同字典作为 `extra_args`。
旧 RequestInfo 的相对 deadline 应由 adapter 在明确的请求到达时间上转换，不能直接当 epoch。
OpenHands benchmark 已通过 `RecordedLLM` 注入这些字段，配置默认关闭。
每个 workflow 使用独立 collection 和任务起点计算的固定 deadline，Tool 轮次与重试不续期。
`output_len` 为调用方固定估计，不读取任务真实输出或启用在线 QRF。
只修改 base_url 而不提供必需 SLO 参数仍会返回错误。

### OpenHands 成对验证

在 GPU 空闲时，从本仓库运行：

```bash
.venv/bin/python examples/others/jitserve/validate_benchmark.py \
  --sdk /home/liyachen/openhands/software-agent-sdk \
  --gpus 0,1,2,3 --rounds 2 \
  --output /absolute/new/paired-evidence
```

脚本依次启动并关闭自己的 stock/JITServe 服务，共用 TP4、BF16、MRV1、eager、
16384 上下文、每 rank 6 GiB KV 预算、worker 容量 4 和 token budget 512。
JITServe 逻辑初始批大小 2，CPU 总预算 8 GiB。使用相同的两文档合成任务、
Hotpot adapter、真实 search/read_document、固定官方 scorer 与采样配置；
每个并发等级 1/2/4 重复指定轮数。它不是公开 HotpotQA 准确率实验。

SDK 必须使用自己的虚拟环境并安装 benchmark 的依赖（包括 scorer 的 `ujson`）。
验证脚本检查显式 SDK 基线 `6152623b7b171eb0fe2917d9989425c54c27d789` 的
核心源码和 editable 安装；完整配置、版本记录、请求/Tool 轨迹及评分会落盘。
未启用 FlowPilot admission、retention、reuse、DCS，也未修改 OpenHands 核心。

`serve_qwen35.sh` 可用 `JITSERVE_BACKEND=stock-v1` 启动同参数原生引擎。
显式设置 `additional_config.jitserve.trace_requests=true` 后，EngineCore 仅记录
引擎实际 request ID、外部 request ID 及白名单 SLO 参数。默认关闭，不记录 prompt、
Tool 内容或凭据。`engine-identity.json` 核验参数到达与客户端身份关联；
该记录不表示实现了 FlowPilot KV control/target-prefix 协议。

预测服务可在持有 V1 EngineCoreClient 的进程内调用：

```python
accepted = await engine_core.call_utility_async(
    "jitserve_update_prediction", engine_request_id, predicted_output_len, version
)
```

使用引擎实际 request ID；版本严格递增。请求已结束或版本陈旧返回 False。
这不是新增 HTTP 管理接口，也不引入 FlowPilot 的 RESTORE RPC。

## 不使用 GPU 的回归

```bash
CUDA_VISIBLE_DEVICES='' VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 \
  JITSERVE_REFERENCE=/home/liyachen/workspace/JITServe \
  .venv/bin/python -m pytest --confcutdir=tests/v1/jitserve tests/v1/jitserve -q
```

测试覆盖策略参数、集合奖励、物理页覆盖后恢复、混合块边界、取消、容量峰值、
预测版本、请求校验和 worker 传输序列化。调度器测试使用真实 V1 分配器和给定 token 输出，
不代表模型推理。CUDA 字节复制用例在隐藏 GPU 时明确跳过。
源代码差分测试从 `JITSERVE_REFERENCE` 加载原函数；未设置时明确跳过该组，
不会将手写近似实现当作原版参照。

当前记录见 [VALIDATION.md](VALIDATION.md)。
