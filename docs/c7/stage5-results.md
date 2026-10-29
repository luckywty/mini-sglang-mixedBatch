# C7 阶段 5：Benchmark 工具与台式机实验计划

日期：2026-10-29，Asia/Shanghai。工作区 `/home/wty238388/ai-infra/Serving_V1/mini-sglang-mixedBatch`，
分支 `c7/mixed-batching-stage1`。进入本阶段 SHA 为 `8d4196a`，工作区干净。

本阶段完成了可复现的请求冻结、开环 HTTP 客户端、顺序 A/B 编排、独立调度/GPU 诊断与指标审查工具，
以及[完整台式机实验方案](../../benchmark/c7/README.md)。**没有执行台式机 Benchmark，也没有根据笔记本数字判断性能收益。**
笔记本仅运行小容量协议/采集链路 smoke，结果标为 `smoke_correctness`，分析器拒绝为它计算吞吐加速比。

## 审计与实际变更

- 沿用 RTX 3060 Laptop 6GB、WSL2、Python 3.12.14、Qwen3-0.6B BF16、TP=1。
- PyTorch 2.9.1 / CUDA 12.8、FlashInfer 0.6.16.post3、Transformers 4.57.3、Triton 3.5.1、
  sgl-kernel 0.3.21；完整依赖与阶段 0 冻结清单一致，未安装/升级依赖。
- 本机 Nsight Systems CLI 为 2024.2.3.38，仅检查版本、help 和生成命令，没有执行 Nsight 采集。
- **`python/minisgl/` 与阶段 4 零差异**。生产默认值、调度策略、GraphRunner、Sampler、Prefix Cache、资源回收逻辑均未修改。

证据：[环境与真实 FI API](evidence/stage5-environment.txt)、[依赖](evidence/stage5-dependencies.txt)、
[Nsight 版本](evidence/stage5-nsys-version.txt)、[CLI help](evidence/stage5-nsys-help.txt)。

| 新增/修改文件 | 作用 |
| --- | --- |
| `benchmark/c7/prepare.py` | 本地模型/Tokenizer 文件内容 SHA256；生成精确 Token 长度、固定 UID/到达/输出/SLO 的不可覆盖 plan |
| `benchmark/c7/common.py` | plan 指纹/边界校验、源码指纹、分位数与固定窗口/排空指标定义 |
| `benchmark/c7/client.py` | 标准库开环 HTTP/SSE，发送前开始计时，显式记录失败和客户端容量不足 |
| `benchmark/c7/run.py` | 原生 Prefill-first/Mixed、公平配置、交替先后顺序、独立服务/预热/测量、保存实际命令/版本/源码与失败 |
| `benchmark/c7/server.py` | 独立实验入口；primary 仅记录启动配置，events/profile 才安装诊断 hooks，CUDA 时间在退出时读取 |
| `benchmark/c7/analyze.py` | UID/输入 SHA/配置/环境核对，JSON/CSV、配对结果与描述性重复分布；拒绝不完整/不公平的吞吐比值 |
| `benchmark/c7/README.md` | 台式机迁移、预检、校准、全部指标口径、Nsight 与消融方案 |
| `tests/benchmark/test_c7_benchmark.py` | 30 项 CPU 测试，含本地真实 HTTP mock 服务验证 SSE 统计与容量处理 |
| `.gitignore` | 忽略 `artifacts/c7/` 生成的请求、服务日志与实验结果；关键证据另存本文目录 |

没有独立 Decode-priority 开关：当前 `_select_next_batch` 只有原生 Prefill-first 与 Mixed 两条分支。
工具只提供这两组对照，显式拒绝 `decode-priority`，未额外引入调度优化。

## 新确认的原生 Baseline HTTP 阻断

首轮工具 smoke 开启双方 Overlap、Graph=4，预热阶段发现 Baseline HTTP 提前结束：

| 请求 | 约定输出 | HTTP 收到 | backend 实际输出 | backend finished 的输出序号 |
| --- | ---: | ---: | ---: | --- |
| warmup-0 / UID 1 | 9 | **8** | 9 | **8、9** |
| warmup-1 / UID 0 | 8 | **7** | 8 | **7、8** |

服务端原路径使用共享 Req 的可 Decode 状态判断 finished，Overlap 中该状态可能已被下一轮推进；
HTTP 看到第一个 finished 后停止流。阶段 4 离线收集完整 backend 回复仍能取得全部 Token，
当时已报告重复 finished；本阶段确认它会影响真实 HTTP 输出长度。
这不是“相同 Token 数下更快”，不能拿提前结束的时间计算收益。

工具行为：Baseline 预热长度检查失败，**不启动其测量负载**；保留退出码和部分输出，
继续记录另一个策略的独立结果，但整轮 exit 1，分析器不生成缺失的对照比值。
同配置 Mixed 完成协议检查，不能据此推断性能优于失败的 Baseline。

证据：[初轮执行结果](evidence/stage5-smoke-profile-initial.txt)、
[原始预热回复/后端 Token 与 finished](evidence/stage5-overlap-failure.txt)、
[初轮分析退出记录](evidence/stage5-analysis-initial.txt)。

**最小处理是正式计划默认双方关闭 Overlap**，用原项目已有环境开关，不修改原生实现。
prepare 提供 `--overlap`，保留可复现失败配置；需要 Overlap 开启的公平 Benchmark 时，应先单独修复旧路径并补回归。
本阶段没有通过修改输出长度、忽略预热错误或补造 Token 让它通过。

## 已实现的公平性与统计口径

同一 plan 固定模型文件内容、实际 Prompt 文本/IDs、输出长度、采样、请求 UID、绝对到达间隔、种子、服务容量和 SLO。
所有请求在预设窗口内开放到达，不因服务器完成速度改变后续发送时间；线程不足会显式 drop 并留在成功率分母中。
两策略只有 Mixed flag 不同；每 trial 重启相同配置的服务，使用相同预热和静置间隔。
奇数轮反转策略及场景顺序，原始顺序写入 schedule。

主实验 `primary` 没有每批/每请求诊断 hooks，也不运行 GPU 轮询。
`events/profile` 独立重复相同 plan，分别收集 CPU/资源及 CUDA 时间，不能并入主实验样本。
本阶段没有每 Forward 强制同步或额外模型 Forward，也没有阶段 4 HTTP 测试入口的人工延迟。

本 API 的 `usage` 为 0、末尾有独立 finish chunk；每个模型 Token 的 ack 也可能没有文本。
客户端按本地源码的“一 Token ack 一 JSON”契约计数，包含首个携带 role 的 Token 和空 content Token，排除 finish/DONE。
成功必须满足实际输出长度、恰好一个 role/finish/DONE、无错误；UID/manifest ID 不能重复计为多个成功请求。
后端重复 finished 在诊断中另计，不增加 Success Rate。

已提供原任务要求的全部指标字段：

- 客户端：Output Token/s、Goodput req/s、TTFT P50/P95、ITL P50/P95/P99、TPOT、Success Rate、迟发和失败统计。
- 调度：backend 收到请求到首次 Prefill 提交的等待、每 UID 连续 Decode 提交间隔（含 Mixed 的 Decode 行）、实际 Graph/Eager 比例。
- GPU/资源：单次 Mixed Engine CUDA Events 时间、PyTorch allocated/reserved、nvidia-smi 整卡采样、resident/non-evictable KV pages、protected Prefix Token。

统计窗口严格区分 `[0,W)` 与整个 cohort+drain：窗口分子不包含排空后的完成/Token，cohort 分母至少为 W。
Goodput 使用包含客户端迟发的 offered TTFT、每请求 TPOT 和最大 ITL 的冻结 SLO；默认阈值只是演示工程假设。
单输出请求的 TPOT/ITL 为 null。原始客户端实际发送时间与 token ack 时间保存在 artifacts 中。

分析器拒绝为 smoke、诊断、缺失/失败请求、P99 迟发超限、模型配置/plan/源码/环境不一致的 trial 计算吞吐比值；
仍保留过载下的成功率、Goodput、部分输出与失败数据。重复轮次只给描述性分布，不声称统计显著性。
详细公式、采样边界和不可混用的指标见[工具方案](../../benchmark/c7/README.md)。

## 实际验证

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 新 Benchmark CPU 套件 | 初版 22 passed；最终 30 passed，并入下行回归 | [初版](evidence/stage5-benchmark-tests-initial.txt)、[最终](evidence/stage5-benchmark-tests-final.txt) |
| 新套件 + 原 CPU 调度/采样/缓存/生命周期 | **103 passed，0 skipped** | [最终日志](evidence/stage5-cpu-tests.txt) |
| 原生 Overlap HTTP 预热 | **FAIL，保留失败；不产生对照收益** | 上述原始预热证据 |
| 双方无 Overlap，Graph=4，profile smoke | 两策略各 6 个测量请求和 2 个预热请求，全部完整输出；Mixed 记录 8 次真正 Mixed | [执行](evidence/stage5-smoke-no-overlap-profile.txt)、[分析](evidence/stage5-smoke-no-overlap-profile-analysis.txt) |
| 同 plan 的 primary smoke | 两策略各 6 个测量请求和 2 个预热请求，协议检查通过，无每轮诊断 hooks | [执行](evidence/stage5-smoke-no-overlap-primary.txt)、[分析](evidence/stage5-smoke-no-overlap-primary-analysis.txt) |
| smoke 比值门禁 | 四个完整 trial 均不可作为性能对照，比值全部 null；24 个测量请求无协议/长度错误 | [汇总](evidence/stage5-smoke-summary.txt)、[冻结 smoke plan](evidence/stage5-smoke-frozen-plan.txt) |
| 桌面完整计划与 dry-run | 4 场景各 96 请求，内容集一致；8 个共同预热，24 次主试验，仅生成命令 | [计划校验](evidence/stage5-desktop-plan-checks.txt)、[命令](evidence/stage5-desktop-dry-run-final.txt)、[冻结 plan](evidence/stage5-desktop-frozen-plan.txt) |
| Nsight 包装 | CLI/help/两条命令生成通过；**未执行采集** | [生成命令](evidence/stage5-nsys-dry-run-final.txt) |
| 静态/隔离检查 | Black、Ruff、Python 编译、依赖一致性、生产源码零差异均通过 | [命令与退出码](evidence/stage5-checks.txt) |

小容量 smoke 为 context=128、3 槽、768 KV Token、Budget=7，Graph CLI=4（实际 `[1,2,4]`）、FI/page_size=1。
profile 组 Baseline 有 31 个测量相关 Forward（12 Prefill/19 Decode），Mixed 为 28（5 Prefill/8 Mixed/15 Decode）。
各有 19/15 次真实 Graph replay，Mixed Graph 违规为 0；CUDA Events 与显存采集字段实际有值，
阶段结果只用它们确认采集工作，不报告或比较笔记本速度。

本阶段没有重新跑未修改的完整 Attention/真实 Engine 回归套件，沿用阶段 4 的 22/12 项结果；
新增实验入口则由上述实际新服务进程验证。所有 GPU smoke 顺序执行，未发生 OOM。
生成的完整原始实验日志在本地 `artifacts/c7/stage5-*`，可随实验归档；表中关键命令、冻结请求、失败及协议/采集摘要已纳入 Git。

## 失败和修正记录

- 第一次合并 CPU 回归漏设 `PYTHONPATH`，误导入相邻项目，2 项 collection error；
  [原日志](evidence/stage5-cpu-tests-import-failure.txt)保留。
- 修正导入后，命令仍漏将 venv/bin 加入 PATH，TVM FFI 找不到已有 `ninja`，7 failed / 96 passed；
  [原日志](evidence/stage5-cpu-tests-path-failure.txt)保留。完整 PATH/PYTHONPATH 后 103 passed，无需安装或改代码。
- 新脚本最初有 dict 字面量/未使用导入的 lint 项，按固定 Ruff/Black 版本修正；没有因此改变运行指标定义。
- 初版桌面示例使用 2 个预热，最终默认增加至 8 个以覆盖更多并发形状；保留初版 dry-run，正式命令引用 `*-final`。
  这不是根据性能结果选择参数：桌面负载没有执行。
- 原始日志仅规范化行尾空白；输入 plan/指纹数据保持相同解析内容。

## 笔记本实际复现命令

```bash
cd /home/wty238388/ai-infra/Serving_V1/mini-sglang-mixedBatch
export PATH="/home/wty238388/ai-infra/Serving_V1/mini-sglang/.venv/bin:$PATH"
export PYTHONPATH="$PWD/python"
export MINISGL_TEST_MODEL=/home/wty238388/ai-infra/models/Qwen3-0.6B

CUDA_VISIBLE_DEVICES='' python -m pytest -o addopts='' -p no:cacheprovider -q \
  tests/benchmark/test_c7_benchmark.py tests/core/test_mixed_lifecycle.py \
  tests/core/test_mixed_scheduler.py tests/core/test_cache_allocate.py

# 新路径，避免覆盖已归档结果；这是小容量工具 smoke。
CUDA_VISIBLE_DEVICES='' python benchmark/c7/prepare.py --model "$MINISGL_TEST_MODEL" \
  --preset smoke --requests 6 --reference-rate 30 --graph 4 \
  --output artifacts/c7/reproduce-smoke-plan.json
python benchmark/c7/run.py --plan artifacts/c7/reproduce-smoke-plan.json \
  --output artifacts/c7/reproduce-profile --scenarios interference --repeats 1 --mode profile --execute
python benchmark/c7/analyze.py artifacts/c7/reproduce-profile
python benchmark/c7/run.py --plan artifacts/c7/reproduce-smoke-plan.json \
  --output artifacts/c7/reproduce-primary --scenarios interference --repeats 1 --mode primary --execute
python benchmark/c7/analyze.py artifacts/c7/reproduce-primary

# 复现原生 Overlap 失败：单独 plan，仅验证阻断，预期整轮 exit 1。
CUDA_VISIBLE_DEVICES='' python benchmark/c7/prepare.py --model "$MINISGL_TEST_MODEL" \
  --preset smoke --requests 6 --reference-rate 30 --graph 4 --overlap \
  --output artifacts/c7/reproduce-overlap-plan.json
python benchmark/c7/run.py --plan artifacts/c7/reproduce-overlap-plan.json \
  --output artifacts/c7/reproduce-overlap --scenarios interference --repeats 1 --mode profile --execute
```

桌面 plan 的 `reference-rate=1` 是输入例子而非容量测量。它的完整生成/运行、Baseline-only pilot、诊断和 Nsight 命令
已写入[实验方案](../../benchmark/c7/README.md)，本阶段没有在笔记本执行这些桌面负载。

## 未验证范围与下一步

- RTX 5060 Ti 尚未实际运行；其 BF16/FI kernel/Graph/显存容量、依赖和 Nsight 采集必须重新预检。
- low/near/overload 尚未根据台式机 Baseline 校准，三档负载状态、SLO Goodput、稳定态与收益没有实测结论。
- Overlap 开启的原生 Baseline HTTP 正确性仍阻断公平对照。工具保留并拒绝失败，没有修复这个既有问题。
- 阶段 4 的两个 raw logits FAIL 未解决；仍只支持已限定的数值条件，不声称跨形状/缓存来源普遍一致。
- 没有长时间稳定性、真实业务请求、随机采样质量、其他模型/backend/TP、功耗锁定或统计显著性结论。

下一轮可直接续接：

> 阅读 `docs/c7/stage5-results.md` 和 `benchmark/c7/README.md`，核对分支、SHA、工作区与目标 GPU。
> 在 RTX 5060 Ti 上先做环境/FI/Graph 和小容量协议预检；不自动升级依赖。
> 只用 Baseline pilot 校准 R 和窗口，冻结模型/请求/到达/采样/SLO/服务参数后，
> 依方案顺序运行原生 Prefill-first 与 Mixed。当前主对照双方关闭 Overlap；
> 若要验证同开 Overlap，先单独修复并回归旧 Baseline 的提前 finished。
> 主运行、诊断、Nsight 各用独立目录，保留失败和原始数据，不预设任何性能收益。
