# JITServe 策略对齐，2026-10-08

参照 `/home/liyachen/workspace/JITServe` 的 `84584a1`，修改仅位于独立的
`vllm-jitserve-0.29.0` 工作区。本记录区分策略决定与 V1 实际执行；实验标签仍为
`jitserve-v1-port`，并记录源码及配置，不宣称旧引擎逐步等价或性能收益。

## 已对齐的策略

| 项目 | 实际行为 | 验证方式 |
| --- | --- | --- |
| 三类奖励 | latency / throughput / collective，预测不足后的剩余成本下限、集合奖励聚合 | 原函数差分，36 请求 × 80 次更新 |
| 增量收益 | 只观测调度后仍运行的请求；无新增输出的观测将收益清零；prompt 增量不重复计入 | 原 SLOTracker 差分 |
| 优先级缓存 | 仅在周期边界清空；预测更新不立即使缓存失效 | 原 BasePolicy/SLOPolicy 差分 |
| 周期耗时 | 首次样本为 20 ms，后续周期耗时除以间隔，保留最近 10 次 | 原 BasePolicy 差分 |
| 周期门控 | 仅第 N、2N…次调度执行 SLO 分支，其余轮次保留运行队列、FIFO 换入并对新请求排序 | 差分及真实 V1 调度器回归 |
| 稳定排序 | 候选顺序为 waiting + swapped + running；选择只比较奖励密度第一分量，平分保留顺序 | 两种策略各 60 轮源函数差分 |
| 滑窗 | top-k 使用初始逻辑批大小；滚动求和；下界包含，上界开放；更长运行请求保留 | 差分及真实 V1 回归 |
| 长度分组 | 邻组补足、最后一组开放上界、待抢占队列跨周期保留；耗尽后重算 | 源函数多轮差分 |
| 注入与激活 | 符合边界的新/换出请求逐个插到队首；按有效待抢占项数量激活 | 源函数多轮差分 |
| 分组抢占 | length_bin 立即 SWAP；滑窗待抢占项只贡献激活额度，不在该阶段立即 SWAP | 差分及调度器回归 |
| 动态批大小 | 当前 SLO budget 先取 max(初始批大小, process 长度)，TBT 超限只减少之后普通轮次的入场上限 | 源函数多轮差分 |
| CPU 迟滞 | 空闲比例 <20% 停止 SLO 分支，之后 >80% 才恢复；改用实际预留字节比例 | 阈值及边界源函数差分 |

原引用代码为 `jitserve/scheduler/{policy,slo_scheduler}.py` 和
`jitserve/slo_tracker/slo_tracker.py`。测试通过 AST 加载这些类的原函数体，
仅替换时钟、日志和旧 SequenceGroup 输入对象，不加载旧 vLLM 二进制。

## 保留的源代码特性

- `_cur_rank` 没有实际递增调用；没有新增论文式或推测的分组轮转。
- 集合需求扩展使用 `collections.get(cid, []).append(...)`，未向字典插入项目；
  原实际执行没有扩展。本移植不擅自修正这个算法行为。
- collective 的成本使用 ingress 时的预测字段，而单请求奖励使用更新后的预测字段。
  版本化 utility 仍拒绝陈旧/已结束请求的预测；其排序效果遵循原缓存周期。
- `schedule_interval=1` 可能导致冷启动无请求入场；非常小的初始批大小结合低/零 TBT
  可能使普通轮次入场上限降至 0 或负数。这些源行为没有用自动放行或下限隐藏。
  实验应使用包含普通轮次的间隔，并显式提供要评估的 TBT。
- 不移植旧引擎的 idle 后自动 abort、预测长度强制停止或错误位置参数调用。
  请求停止与取消继续由 vLLM 0.29.0 执行。

## 执行层一致性修复

SLO 轮次现在使用独立的预算与执行路径，普通轮次继续使用 V1 调度器。
`execution.py` 对照原 `_schedule_process_list`，`scheduler.py` 将结果提交给真实
V1 混合块分配器；两条路径共用原生输出打包、页清零、COW 和进度更新链路。

| 项目 | 当前行为 |
| --- | --- |
| 预算顺序 | 按原 `process_list`，先激活候选，再处理剩余运行请求 |
| prefill 预留 | 每项先预留最多 2 token；运行 prefill 不先耗尽整轮预算 |
| 显存预算 | 按候选完整物理页需求，对照总可用容量；不是只扣当前空闲页 |
| 停止分配 | 首次不能满足预算后锁存 `stop_allocate`，后续小请求不能补位 |
| 资源抢占 | 按列表顺序换出不被接纳且仍有 token 工作的运行请求 |
| 提交顺序 | swapped → running → waiting；新 prefill 按队列顺序消耗剩余 token |
| 运行队列 | 恢复 decode/prefill、原运行 decode/prefill、新 prefill、剩余运行请求 |
| 物理申请失败 | 保留请求，避免另选一个已接纳请求作为受害者 |

240 组确定性资源组合直接执行旧函数体，对照预留、受害者、提交顺序及剩余预算。
真实 V1 分配器回归覆盖 16/64 混合块、首候选超出页预算、按序换出、私有水位恢复，
以及 token 耗尽时保留未执行请求。未复刻旧版 token 耗尽时清空运行队列而丢失请求的缺陷。

## V1 执行边界

`--max-num-seqs` 在启动时决定 worker 数组容量，不能在调度器里无界改大。
新增 `jitserve.max_batch_size` 表示旧算法的初始批大小，默认等于该执行容量。
例如初始 16、worker 容量 32：

```bash
JITSERVE_MAX_NUM_SEQS=32 \
JITSERVE_ADDITIONAL_CONFIG='{"jitserve":{"max_batch_size":16,"schedule_interval":50,"top_k_selection":1,"max_swaps_per_iter":3}}' \
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/others/jitserve/serve_qwen35.sh
```

达到执行容量时，按原 `process_list` 的顺序保留可执行请求，其余运行请求保存 CPU
快照；统计 `capacity_limited_rounds` 并在首次发生时记录日志。动态逻辑入场上限不被
篡改，普通入场仍受实际执行容量约束。比较实验应记录这一计数；非零说明该硬容量
影响了执行，不能仅以“动态批大小已对齐”声称完全复现。

物理页需求按 V1 hybrid/Mamba 实际布局计算，不能等同旧 V0 的页数。
两 token prefill 可以推进未完成块内的私有状态，仍在可缓存块边界、共享前缀和
尾部检查点处停下；原生调度器默认分块行为不变。必要边界可能将实际 token 数缩小。
逻辑选择、固定 worker 容量和物理申请结果分别保留；不能宣称整个旧引擎逐步等价。
CPU 实际复制、私有恢复页、GDN 状态及计算水位仍走真实 V1 路径，不以重算代替恢复。

## 验证范围

策略对齐证据位于 `.agent_tmp/jitserve-validation/20261008-policy/`；执行修复后的新基线
及回归位于 `.agent_tmp/jitserve-validation/20261008-execution/`。
`validate_policy.py` 使用自然调度计数，包含长度分组换出恢复、滑窗保留较长请求、
滑窗 2→3→4 逻辑扩批触及容量 3，以及换出后取消。stock 与 JITServe 使用相同模型、
prompt token、采样和物理执行配置；后者额外断言每个 worker 的实际复制与最终回收，
并记录每个 SLO 轮次的预算预留、停止位置、换出列表和真实 token 分配。

`validate_matrix.py` 用受控的 SLO 周期对齐指定恢复水位，专门检查执行正确性；
它不作为自然策略周期的证据。早先 `20261007-qwen35/` 的 45 项回归和 HTTP 记录属于
对齐前版本，不能代替本次策略代码的验证结果。最终执行结果补充在 `VALIDATION.md`。
