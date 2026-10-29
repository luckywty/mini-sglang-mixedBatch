# C7 台式机实验方案与工具

工具面向 Qwen3-0.6B BF16、TP=1、FlashInfer/page_size=1。阶段 5 准备脚本和实验计划；
笔记本上的小规模运行仅检验工具，不证明性能收益。
当前结论与实际验证见 [阶段 5 报告](../../docs/c7/stage5-results.md)。

## 对照与已知阻断

| 策略 | 配置 | 状态 |
| --- | --- | --- |
| 原生 Prefill-first | 不传 `--enable-mixed-batching` | 原调度、原 Forward、原 Sampler |
| 真正 Mixed | 仅增加 `--enable-mixed-batching` | Decode 先占预算，其余给 Prefill；Mixed Eager，Pure Decode 保留 Graph |
| 独立 Decode-priority | 当前源码没有独立开关 | 不添加虚构对照，不在本阶段新实现该策略 |

**主实验默认双方关闭 Overlap。** 阶段 5 HTTP 联调复现旧 Baseline 的提前 finished：
预热两请求要求输出 9/8 Token，却在 8/7 Token 时结束流。
这与阶段 4 的额外 finished 标记一致，但离线收集全部回复掩盖了 HTTP 首次 finished 的影响。
`--overlap` 可生成对照配置；当前 Baseline 会在工具的预热长度检查中失败，不能发布为公平的吞吐对照。
解决该既有问题应使用独立修复及回归，本阶段不改原生执行逻辑。
只报告关闭 Overlap 的实验，也不能据此推断开启 Overlap 后的收益。

还保留阶段 4 的两个 raw logits FAIL：跨批形状 BF16 差异、不同 Prefix KV 来源的差异。
这里的输出长度/协议检查不替代精度验收；不能把后续吞吐结果写成“全数值一致”。

## 工具组成与运行模式

- `prepare.py`：本地 tokenizer 生成并往返校验精确长度，冻结实际文本、Token IDs、到达时间、输出长度、SLO、配置、种子及模型文件 SHA256。
- `run.py`：默认只打印命令及环境变量；`--execute` 才启动进程。逐 trial 新建服务，预热、测量、排空、退出。
  同一 plan 对比，仅 Mixed 开关不同；奇数轮反转策略及场景顺序。输出路径存在时拒绝覆盖。
- `client.py`：标准库 HTTP 开环发送，时间从连接/请求发送前开始。线程不足显式记录失败，不排入隐形客户端队列。
- `server.py`：独立实验入口，不修改 `python/minisgl/`。所有模式仅在初始化后记录实际服务配置。
- `analyze.py`：核对输入 SHA、UID、实际配置、捕获的 Graph sizes、环境及源码指纹，生成 JSON/CSV。

| `--mode` | 采集内容 | 使用范围 |
| --- | --- | --- |
| `primary` | 客户端端到端指标、启动时的实际配置/版本 | 主性能实验；不安装每请求/每 Forward 诊断 hook，不运行 GPU 轮询 |
| `events` | 另加 backend 收到请求、Forward 提交、原始输出、KV/显存快照及 nvidia-smi 轮询 | 调度与资源诊断；不计算可发布的性能比值 |
| `profile` | events 内容、CUDA Events、C7 NVTX 区间；可选 `--nsys` | GPU/CPU 时间线诊断；不与 primary 混合统计 |

CUDA Events 在 Engine 初始化/Graph 捕获之后安装，在原 shutdown 已同步后读取。
没有每轮 `synchronize()`、没有人为延迟、不更改 GraphRunner 或资源回收时机。
诊断仍有 CPU 记录、事件和轮询开销，因此与主运行分开。

## 迁移与目标 GPU 验证

将本分支及所需本地模型同步到台式机，选择已经配置的解释器；不要复制 WSL 虚拟环境或自动升级核心依赖。
以下路径需换成台式机实际路径：

```bash
cd /path/to/mini-sglang-mixedBatch
export PATH="/path/to/existing/venv/bin:$PATH"
export PYTHONPATH="$PWD/python"
export MINISGL_TEST_MODEL=/path/to/Qwen3-0.6B
mkdir -p artifacts/c7/preflight

git status --short --branch
git rev-parse HEAD
python scripts/c7/audit_env.py > artifacts/c7/preflight/environment.txt 2>&1
uv pip freeze --python "$(command -v python)" > artifacts/c7/preflight/dependencies.txt

CUDA_VISIBLE_DEVICES='' python -m pytest -o addopts='' -p no:cacheprovider -q \
  tests/benchmark/test_c7_benchmark.py tests/core/test_mixed_lifecycle.py \
  tests/core/test_mixed_scheduler.py tests/core/test_cache_allocate.py
python -m pytest -o addopts='' -p no:cacheprovider -q -s tests/attention/test_mixed_flashinfer.py \
  > artifacts/c7/preflight/attention.txt 2>&1
python scripts/c7/summarize_attention.py artifacts/c7/preflight/attention.txt
python -m pytest -o addopts='' -p no:cacheprovider -q -s tests/engine/test_mixed_execution.py
```

`audit_env.py` 沿用本项目已有工具，使用 `/usr/local/cuda/bin/nvcc`；目标安装路径不同，应单独记录实际 `nvcc --version`，
不要把审计脚本路径失败误报成 FI kernel 失败。真实模型 GPU 测试须无 skip；FI kernel、BF16、Graph 支持应在 5060 Ti 上重新验证。
版本不兼容或 OOM 时保留日志，先给出最小配置/兼容修复方案，不自动升级或静默换 backend。

先用小容量协议 smoke 检查迁移环境；生成物不是性能数据：

```bash
python benchmark/c7/prepare.py --model "$MINISGL_TEST_MODEL" --preset smoke \
  --requests 6 --reference-rate 30 --graph 4 --output artifacts/c7/smoke-plan.json
python benchmark/c7/run.py --plan artifacts/c7/smoke-plan.json \
  --output artifacts/c7/smoke --scenarios interference --repeats 1 --mode profile --execute
python benchmark/c7/analyze.py artifacts/c7/smoke
```

## 负载设计与冻结

桌面初始容量：context=2048、8 个请求槽、8192 KV Token、Budget=128、`--graph 8`、radix cache。
它是保守的起始配置，不是已在 5060 Ti 上确认的最优容量。
原生 Graph CLI 在此捕获 `[1,2,4,8]`；没有修改捕获结构。
模型权重固定 BF16，采样 `temperature=0, top_k=1, top_p=1, ignore_eos=True`。

桌面默认冻结 96 请求：约 80% 输入 32–128、输出 64–128；约 20% 输入 512–768、输出 16–64。
使用随机词表 Token 解码后的合成文本，排除 special IDs，按服务端相同默认 `encode()` 检查往返长度。
不模拟真实语言质量；实际输入 IDs 随 manifest 保存，诊断时与后端 tokenization SHA 比较。
多样随机开头尽量减少共享前缀，但 radix 的偶然命中仍属于实测行为，并不宣称绝对零命中。
每 trial 进程重启使初始 Prefix Cache 为空；默认 8 个与测量内容不同的预热请求，覆盖短/长 Prefill，随后固定静置 0.5s。
预热不是所有未知形状的 JIT 完成证明：先查看目标机诊断日志；必要时用 `--warmup-requests` 增加双方共同预热并重新冻结计划。

先只用 Baseline pilot 校准参考请求率 `R`。`R=1` 仅是可以执行的例子，不代表测得容量。
在多个 pilot 速率观察成功率、排队/TTFT、ITL 与资源占用，找出接近 SLO/持续积压转折的速率，然后锁定 R。
不得看 Mixed 的最好结果后挑选 R 或阈值。

| 场景 | 冻结到达过程 | 验证目标 |
| --- | --- | --- |
| low | 同一指数间隔序列，目标率 `0.25R` | 空载/低负载开销与延迟 |
| near | 相同间隔序列，目标率 `0.85R` | 接近容量的 TTFT/ITL 与 Goodput |
| overload | 相同间隔序列，目标率 `1.3R` | 积压、超时、成功率及 SLO 退化 |
| interference | 先并发短输入/长输出，再交错长 Prefill 与持续的短输入流 | Decode 是否被长 Prefill 阻塞；诊断中须确认确有 Mixed Forward |

三档 Poisson 场景的请求内容、UID、输出长度完全相同，只缩放间隔；interference 复用同一批内容并改变到达顺序。
`low/near/overload` 都是候选名称：若 Baseline pilot 未显示对应负载状态，应如实标注未达到，调整后重新冻结。
有限样本的实际 offered rate 为 `N/window_s`，不等于所填参考率的精确倍数。

```bash
export C7_RATE=1.0  # 演示值；替换为台式机 Baseline pilot 校准值
python benchmark/c7/prepare.py --model "$MINISGL_TEST_MODEL" --preset desktop \
  --requests 96 --reference-rate "$C7_RATE" --seed 20261029 \
  --output artifacts/c7/desktop-plan.json

# 默认 dry-run，查看完整命令/环境；此时没有 GPU 服务或流量。
python benchmark/c7/run.py --plan artifacts/c7/desktop-plan.json \
  --output artifacts/c7/main --repeats 3

# 主实验：4 场景 × 2 策略 × 3 轮，顺序驻留 GPU。
python benchmark/c7/run.py --plan artifacts/c7/desktop-plan.json \
  --output artifacts/c7/main --repeats 3 --mode primary --execute
python benchmark/c7/analyze.py artifacts/c7/main

# 相同 plan，独立的诊断目录，不能并入主实验重复样本。
python benchmark/c7/run.py --plan artifacts/c7/desktop-plan.json \
  --output artifacts/c7/events --repeats 1 --mode events --execute
python benchmark/c7/analyze.py artifacts/c7/events
python benchmark/c7/run.py --plan artifacts/c7/desktop-plan.json \
  --output artifacts/c7/profile --scenarios interference --repeats 1 --mode profile --execute
python benchmark/c7/analyze.py artifacts/c7/profile
```

pilot 可通过 `--policies baseline --scenarios near --repeats 1` 单跑；`analyze.py` 仍输出指标，但不生成不存在的策略对照。
正式实验建议扩大请求数/统计窗口并做至少 3 轮，条件允许时采用偶数轮平衡先后顺序。
同一负载窗口内不要执行其他 GPU 任务，记录功耗、温度、时钟及后台负载；工具不改功耗上限或锁频。
容量调整、Budget、Graph、cache、Overlap、SLO、种子都需要生成新 plan，对每个 plan 单独完整做 A/B。

## 指标定义

当前源码每个 backend `DetokenizeMsg` 对应一个 `UserReply`，每个回复产生一个 `finish_reason=None` 的 JSON chunk。
首个回复同时带 role，空 content 也可能是 Token；末尾另有 finish chunk 与 `[DONE]`。
工具按此明确的源码契约计数，**不是按非空文本块计数**，也不使用全为 0 的 `usage` 或重新编码输出文本。
更换服务 API/聚合方式后须重新验收这个契约。

| 指标 | 定义/输出字段 |
| --- | --- |
| Output Token/s | `window_output_token_s`：落在 `[0,W)` 的真实 token ack 数/W；`cohort_output_token_s`：整个请求队列收到的 Token/含排空时长。失败请求已收到的部分输出保留在计数中，不能伪装为成功 |
| Goodput req/s | 唯一请求成功、满足 SLO，且完成时刻位于 `[0,W)` 的个数/W；另给 cohort/drain 版本，不混用分母 |
| TTFT P50/P95 | 实际发送前到首个真实 token ack；另给 `offered_ttft_ms`，从计划到达计算，包含客户端迟发 |
| ITL P50/P95/P99 | 同一成功请求相邻真实 token ack 间隔；所有请求间隔按 Token 加权池化，排除 finish/DONE |
| TPOT | 每成功请求 `(last_token-first_token)/(N-1)`，再给请求间均值/分位数；单输出时为不可用，不编造 0 |
| Success Rate | 完整长度 + 恰好一个 role/finish/DONE + 无错误的唯一 manifest ID/全部 offered ID；失败和 capacity drops 不移出分母 |
| Prefill 等待 | `prefill_wait_backend_ms`：backend 收到 `UserMsg` 到第一次 Prefill Forward 提交，包含 admission/metadata 时间；不含 HTTP/Tokenizer 排队 |
| Decode Forward 间隔 | `decode_submit_interval_ms`：同一 UID 的连续 Decode 行提交间隔，包含 Mixed 中的 Decode 行；是 CPU 提交观察，GPU 实际启动间隔用 Nsight 时间线核查 |
| 单轮 Mixed Forward | `mixed_engine_cuda_ms`：Engine.forward_batch 的 CUDA Events 区间，包含模型、采样和异步 D2H；不含 Scheduler 前置 metadata 或后置 token_pool writeback；不是单 kernel 时间 |
| Graph/Eager 比例 | 按实际 Forward 次数，prefill/mixed/decode 分阶段统计，不按请求数或纯 Decode 的理论可用性猜测 |
| 显存 | 诊断每批的 PyTorch allocated/reserved；`gpu.csv` 为 500ms 整卡采样，含预热、排空及其他进程，不能混称模型专用峰值 |
| KV 占用 | resident pages=`total-free`，包括可驱逐缓存；non-evictable pages=`total-free-evictable`（仅 page_size=1），包含活跃/私有尾部；另记 protected Prefix Token |

`W` 固定为 manifest 的 offered-load window，所有请求到达都在窗口内；计时 0 为客户端该轮启动。
cohort/drain 分母为 `max(W,last_request_end)`，排空数据不会偷渡进入 fixed-window 分子。
另存真实发送延迟分布；P99 超过冻结 `max_dispatch_lag_ms`（默认 20ms）时，不计算性能比值，需降低单客户端负载或扩大客户端能力并重新冻结实验。

默认 SLO 是**演示工程假设**：offered TTFT≤1000ms、每请求 TPOT≤100ms、最大 ITL≤200ms。
可在 prepare 时用 `--ttft-slo-ms/--tpot-slo-ms/--max-itl-slo-ms` 更改；与产品需求确认后再冻结。
单输出请求仅考察 TTFT，ITL/TPOT 不适用。请求 timeout 默认 120s，全部保存在 run 配置中，A/B 不得不同。

`metrics.csv` 是每 trial 的客户端指标；`analysis.json` 保留原始审查结果、调度/资源摘要、每轮配对及重复轮次的描述性分布。
源码/plan/环境不同、协议/长度失败、客户端迟发、诊断运行或 smoke 均不产生吞吐比值。
过载失败仍完整显示 Success Rate/Goodput/部分输出；只是禁止把不完整工作量写成吞吐加速。
重复 finished 在 backend 单独计数，不增加成功请求数。缺失指标为 `null`，不以 0 替代未采集。
每对比值方向为 Mixed/Baseline，Goodput 差为 Mixed−Baseline；没有统计显著性或“必然获益”判断。

## Nsight Systems 与消融

```bash
nsys --version
nsys profile --help
python benchmark/c7/run.py --plan artifacts/c7/desktop-plan.json \
  --output artifacts/c7/nsys --scenarios interference --repeats 1 \
  --mode profile --nsys --execute
python benchmark/c7/analyze.py artifacts/c7/nsys
```

命令包装为 `nsys profile --trace=cuda,nvtx,osrt --sample=none --wait=all --force-overwrite=false`。
`--wait=all` 包括应用产生的重挂父进程；选项依据 [NVIDIA 官方指南](https://docs.nvidia.com/nsight-systems/UserGuide/)。
先在目标版本核对 CLI/子进程 CUDA 追踪能力；本阶段仅验证生成命令，没有运行 Nsight 采集。
采集包含初始化/预热，在 GUI 中用客户端 epoch/trace 时段及 `C7/<phase>/<graph|eager>/bs=...` 标记定位测量区间，
检查 GPU 空洞、长 Prefill 阻塞、Graph replay、CPU/GPU 重叠、计划/写回顺序及不同层的 Kernel。
原有 MHA/MLP/Layer NVTX 保留；Graph replay 的内部范围不能按 Eager 的 Python NVTX 调用次数解释。

消融顺序：先固定 Graph 开启的原生/Mixed 主对照，再为双方同时 `--graph 0`；
Budget 取 64/128/256，每个值均运行两策略；radix/naive 分开记录以隔离 Prefix 来源；
独立修复 Baseline Overlap 后才能做双方同开/同关对照。
每次只改一个维度，用同一内容/种子，保留新 plan SHA 和全部失败；不能用“Mixed 开 Graph、Baseline 关 Graph”等组合推断调度收益。
