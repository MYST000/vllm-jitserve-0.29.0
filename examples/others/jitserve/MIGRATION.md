# JITServe 跨机器迁移（2026-10-08）

推荐用 GitHub 保存源码，用独立文件传输或固定版本下载恢复模型资产，在新机器重建虚拟环境。
当前仓库包含 **已验证的固定估计移植 + 尚未验收的 QRF 初步接入**；QRF 默认关闭。
当前工作区尚未提交，单独复制上游 commit 或推送现有 HEAD 会遗漏全部本地移植。

## 1. 需要迁移的内容

| 内容 | 本机位置 / 恢复依据 | 是否放进普通 Git |
| --- | --- | --- |
| vLLM 源码及本地移植 | `workspace/vllm-jitserve-0.29.0`；上游基点 `98dff2a81d747d1dba01a47f939f48c3526d4206` | 是，包含所有移植新增文件 |
| OpenHands SDK 与 benchmark | `openhands/software-agent-sdk`；HEAD `6152623b7b171eb0fe2917d9989425c54c27d789` 加本地 overlay | 独立仓库，或使用本次配套迁移包 |
| QRF 资产 | `.agent_tmp/qrf-assets/`，模型 1341490597 bytes、向量器 4529414 bytes | 否，下载/rsync |
| Qwen3.5-27B | `/docker/data/HF_MODELS/Qwen3.5-27B`，约 52 GiB | 否，rsync/模型仓库 |
| Python 依赖 | `migration/vllm-installed.txt`、`migration/openhands-installed.txt` | 版本清单放 Git，环境目录不放 |
| 证据与可恢复源码 | `workspace/experiments/jitserve/` 两个归档 | 单独传输；不是启动依赖 |
| 原版 JITServe 差分参照 | `workspace/JITServe`，`84584a119b892cf90c65e2b3f5bac67250d39d27` | 重新 clone，或 rsync |

不需要迁移旧 `/home/liyachen/vllm`。固定估计对照组直连新 JITServe 引擎，
不依赖运行 FlowPilot 服务；两组件分别使用自己的虚拟环境。

## 2. 把当前源码上传 GitHub

在 GitHub 建立目标仓库；下面命令由你在确认目标后执行。文档同步没有执行 commit/push。
保留 LICENSE 与上游历史；使用新的 `migration` remote 避免改写原有 origin。

```bash
cd /home/liyachen/workspace/vllm-jitserve-0.29.0
git status --short
git add .gitignore \
  tools/pre_commit/check_forbidden_imports.py \
  vllm/distributed/kv_transfer/kv_connector/v1/base.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/engine/async_llm.py vllm/v1/engine/core.py \
  vllm/v1/engine/input_processor.py vllm/v1/worker/gpu_model_runner.py \
  vllm/v1/jitserve tests/v1/jitserve examples/others/jitserve
git diff --cached --stat
git diff --cached --check
git commit -m "Preserve JITServe V1 port and unfinished QRF integration" \
  -m "Co-authored-by: OpenAI Codex <noreply@openai.com>" --signoff
# 当前上游 checkout 为 shallow；补齐历史后再向新 GitHub 仓库推送。
if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
  git fetch --unshallow origin
fi
git remote add migration git@github.com:YOUR_ACCOUNT/YOUR_REPO.git
git push -u migration HEAD:jitserve-v0.29.0
git rev-parse HEAD
```

记录最后的 commit，目标机检出该 commit。`.agent_tmp/` 现已加入仓库 `.gitignore`，
其他 `.venv/`、二进制 `.so` 原本已忽略；不要使用 `git add -f` 上传预测资产、环境或临时证据。
GitHub 上的提交包含 QRF WIP，并不代表它已验收。保留旧固定估计归档作为稳定源码恢复点。

## 3. 目标机准备与 vLLM 环境

现有实机证据来自 Linux x86_64、Python 3.12.9、4 张 RTX 4090 24 GiB、TP4、BF16。
当前 PyTorch 为 `2.13.0+cu129`、Triton 3.7.1、Transformers 5.19.0；实际推理 CUDA 为 12.9。
不同 GPU/TP、驱动、glibc 或架构需重新验证；具体源机清单见 `migration/environment.json`。
原 `.venv/bin/python` 指向机器外部的 `/home/zjh/miniconda3/bin/python3`，不能把原 `.venv`
当作可直接解压使用的便携环境。

```bash
git clone --branch jitserve-v0.29.0 git@github.com:YOUR_ACCOUNT/YOUR_REPO.git vllm-jitserve-0.29.0
cd vllm-jitserve-0.29.0
# 将 SAVED_COMMIT 替换为上一节保存的 commit
git checkout SAVED_COMMIT
uv venv --python 3.12.9 .venv

# 恢复源机已安装的第三方包版本；vLLM 本身随后从当前源码安装。
uv pip install --python .venv/bin/python --no-deps \
  -r examples/others/jitserve/migration/vllm-installed.txt \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl/

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION='https://wheels.vllm.ai/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
VLLM_VERSION_OVERRIDE=0.29.0+precompiled \
  uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .

.venv/bin/python -c 'import torch, vllm; print(vllm.__file__); print(torch.__version__, torch.version.cuda); assert torch.__version__ == "2.13.0+cu129"; assert torch.version.cuda == "12.9"'
nvidia-smi
```

这里的 `--no-deps` 用于重放已安装清单，避免重新求解时替换现有 torch/CUDA 版本。
**该清单是源环境快照，不是新求解的依赖锁或“依赖全部兼容”的证明。** 源环境有 CUDA 13
相关可选包及 `torchcodec +cu130`，所以下载源包含 cu130；推理所用 torch 和 vLLM 二进制仍明确固定 cu129。
不要用无版本约束的 `pip install vllm` 或普通 `uv sync` 覆盖此实验环境。
上述 wheel URL 从本机 uv 缓存恢复，未在新机器验证可达性；若需离线迁移，应事先下载该 wheel
及清单中的所有依赖，把 `VLLM_PRECOMPILED_WHEEL_LOCATION` 改为本地 wheel 的绝对路径。

## 4. 恢复模型与 QRF 资产

在源机器执行，替换目标 SSH 地址与路径：

```bash
rsync -a --info=progress2 /docker/data/HF_MODELS/Qwen3.5-27B/ \
  USER@NEW_HOST:/data/models/Qwen3.5-27B/
rsync -a --info=progress2 \
  /home/liyachen/workspace/vllm-jitserve-0.29.0/.agent_tmp/qrf-assets/ \
  USER@NEW_HOST:/data/models/jitserve-qrf/
```

也可以在新 vLLM 工作目录下载相同 QRF 资产：

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'En-2863/jitserve-qrf-length-predictor',
    revision='1e587124a9671f54f4c659f97bccbbe890b3d8af',
    allow_patterns=['metadata.json', 'README.md',
                    'qrf_model/0_qrf_lmsys_chat_llama3_8b.pkl',
                    'qrf_vectorizer/0_qrf_lmsys_chat_llama3_8b.pkl'],
    local_dir='.agent_tmp/qrf-assets',
)
PY
```

核对 `migration/environment.json` 中两个 QRF 文件的 SHA-256。生成模型的记录目前只有配置与索引摘要，
不是 11 个权重分片的完整哈希证明；权重传输可额外用 `rsync --checksum` 复核。
QRF 来自作者发布的 Llama3/LMSYS 资产，应用到 Qwen3.5-27B 属于跨模型使用，暂不重训。

## 5. OpenHands 与历史证据

完整 benchmark 闭环还需要当前 SDK commit 及未提交的 `benchmarks/flowpilot/` 改动。
仅上传 vLLM 仓库不会带走它们。使用配套包，或将 SDK 当前状态另存到你自己的 GitHub 仓库。

```bash
# 在新机器放置配套包及 .sha256 后
sha256sum -c 20261008-qrf-wip-migration.tar.gz.sha256
tar -xzf 20261008-qrf-wip-migration.tar.gz
git clone 20261008-qrf-wip-migration/openhands/HEAD.bundle software-agent-sdk
git -C software-agent-sdk checkout 6152623b7b171eb0fe2917d9989425c54c27d789
cp -a 20261008-qrf-wip-migration/openhands/files/. software-agent-sdk/
cd software-agent-sdk
uv venv --python 3.13.14 .venv
uv pip install --python .venv/bin/python --no-deps \
  -r ../20261008-qrf-wip-migration/environment/openhands-installed.txt
uv pip install --python .venv/bin/python --no-deps \
  -e openhands-sdk -e openhands-tools -e openhands-workspace \
  -e openhands-agent-server -e benchmarks/flowpilot
```

源机 SDK 实际使用 Python 3.13.14，与 vLLM 的 Python 3.12.9 不同。SDK 使用独立环境；
仓库中的 `uv.lock` 是另一份依赖基线，不能把 `uv sync` 后的环境当作与本次
运行快照完全相同。当前验证脚本显式使用上述 SDK pin，并保留源码 provenance 校验。

配套包包含当前两仓库源码 overlay/patch、SDK HEAD bundle、本次日志、旧固定估计验收归档。
它不含 `.venv`、生成模型权重或 QRF 大文件。若 GitHub 不可用，配套包的 vLLM `files/`
可覆盖上游 `98dff2a8...` 的新 checkout；`tracked.patch` 不含新增文件，不能单独使用。
旧固定估计包保持原样，内含当时源码与证据；要复现旧验收版本，应覆盖全新基点 checkout，
不要与当前 QRF overlay 混用。详细恢复清单在包内 `RESTORE.txt` 和 `manifest.json`。

## 6. 新机器验收顺序

先检查模型文件，再验证固定估计配置。替换模型路径，并确认所选四张 GPU 空闲：

```bash
cd /path/to/vllm-jitserve-0.29.0
.venv/bin/python examples/others/jitserve/inspect_qwen35.py --model /data/models/Qwen3.5-27B
JITSERVE_MODEL=/data/models/Qwen3.5-27B bash examples/others/jitserve/serve_qwen35.sh --dry-run

git clone https://github.com/UIUC-MLSys/JITServe.git ../JITServe
git -C ../JITServe checkout 84584a119b892cf90c65e2b3f5bac67250d39d27
CUDA_VISIBLE_DEVICES='' VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 \
  JITSERVE_REFERENCE="$(realpath ../JITServe)" \
  .venv/bin/python -m pytest --confcutdir=tests/v1/jitserve \
  tests/v1/jitserve/test_jitserve.py tests/v1/jitserve/test_scheduler.py \
  tests/v1/jitserve/test_policy_reference.py -q

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 \
  JITSERVE_MODEL=/data/models/Qwen3.5-27B JITSERVE_MAX_NUM_SEQS=1 \
  JITSERVE_ADDITIONAL_CONFIG='{"jitserve":{"schedule_interval":2,"search_strategy":"length_bin","top_k_selection":2,"max_swaps_per_iter":1}}' \
  bash examples/others/jitserve/serve_qwen35.sh
```

另一个终端运行 `.venv/bin/python examples/others/jitserve/validate_http.py --output /tmp/jitserve-http-new-host.json`。
CPU 真实换出/恢复和 token 相等仍按 README 的 `validate_swap.py`、`validate_policy.py` 成对执行。
关闭自己的模型服务后，执行 OpenHands 成对验证：

```bash
JITSERVE_MODEL=/data/models/Qwen3.5-27B \
  .venv/bin/python examples/others/jitserve/validate_benchmark.py \
  --sdk /path/to/software-agent-sdk --gpus 0,1,2,3 --rounds 1 \
  --output /absolute/new/paired-evidence
```

QRF 验收是后续独立步骤，目前不能把它当作迁移后的可用保证：

```bash
CUDA_VISIBLE_DEVICES='' VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 \
  JITSERVE_REFERENCE="$(realpath ../JITServe)" \
  JITSERVE_QRF_ASSETS=/data/models/jitserve-qrf \
  .venv/bin/python -m pytest --confcutdir=tests/v1/jitserve \
  tests/v1/jitserve/test_prediction.py -v -o faulthandler_timeout=30
```

当前实现的启用入口是追加到 `JITSERVE_ADDITIONAL_CONFIG` 的
`jitserve.prediction={"model_path":"绝对模型路径","vectorizer_path":"绝对向量器路径"}`，
同时设置 `jitserve.trace_requests=true` 记录 metadata-only 预测结果。正常请求仍提供初始
`jitserve_output_len`；新预测只更新调度估计，不改 `max_tokens`、EOS、Tool 或 deadline。
当前 benchmark 仍标记 `caller_fixed`，要完成 QRF 实验还需补充标签及实机更新关联证据。
