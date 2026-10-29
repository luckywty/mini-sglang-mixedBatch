# C7 阶段 4：笔记本端到端验证

日期：2026-10-29，Asia/Shanghai。工作区：`/home/wty238388/ai-infra/Serving_V1/mini-sglang-mixedBatch`。

阶段 4 完成了当前笔记本、Qwen3-0.6B BF16 的有界正确性验证：功能矩阵、匹配输入/KV 的数值参考、
HTTP/SSE 和既有回归均通过。**这不代表通用跨批形状或不同 KV 来源的 logits 对照通过**：
两个原始数值诊断仍 FAIL，关闭 Mixed 的旧 Overlap 路径还观察到重复 finished 标记，详见下文。
未运行吞吐、延迟、Nsight 或阶段 5 实验，不推断性能收益。

## 源码、环境与改动范围

- 进入阶段 4 的 SHA：`ad593601b35b5ab8264f2af343747db559ef5510`，当时工作区干净。
  审计日志写入后，状态中的唯一 untracked 是审计日志本身。
- 分支：`c7/mixed-batching-stage1`；`main` 仍为原始 `9a91cfafe754aa85daee49998176275667eb58f2`。
- **本阶段没有修改 `python/minisgl/` 生产代码、默认配置、GraphRunner、Sampler 或 Prefix Cache 实现。**
  原始 Prefill-first 与 Mixed 开关均保留。
- Python 3.12.14、PyTorch 2.9.1 / CUDA 12.8、FlashInfer 0.6.16.post3、Transformers 4.57.3、
  Triton 3.5.1、sgl-kernel 0.3.21、apache-tvm-ffi 0.1.14.post1。
- RTX 3060 Laptop 6GB，SM 8.6，TP=1；WSL2，CUDA toolkit 12.8。
  依赖冻结清单与阶段 0 **逐字一致**，未安装或升级推理依赖。
- 沿用相邻原项目的 `.venv`，所有模型命令显式设置 `PYTHONPATH="$PWD/python"`，
  避免相邻项目 editable install 被错误导入。

证据：[环境/API 审计](evidence/stage4-environment.txt)、[本阶段依赖](evidence/stage4-dependencies.txt)、
[冻结基线依赖](evidence/dependencies.txt)。

实际代码变更：

| 文件 | 用途 |
| --- | --- |
| `tests/engine/test_mixed_execution.py` | 让原真实模型 Driver 接受开关、预算、Graph sizes、容量参数；支持保留 Prefix Cache；默认行为不变 |
| `scripts/c7/stage4_matrix.py` | 单配置真实 Engine/Scheduler 验证，记录请求指纹、输出、Graph/Mixed 路径与资源守恒 |
| `scripts/c7/run_stage4_matrix.py` | 顺序运行 8 组配置及额外 80 轮 churn，检查请求集合和自然 greedy 输出一致 |
| `scripts/c7/stage4_numeric.py` | 真实 28 层 Attention 的 FP32 参考、匹配状态的完整 Forward 对照及 FP32 LMHead 参考 |
| `scripts/c7/stage4_http.py` | 标准库 HTTP/SSE 客户端，验证并发输出、流与 JSON 一致、中断后的后端 Abort 与复用 |
| `scripts/c7/stage4_http_server.py` | 测试专用服务入口，记录真实 Forward/Abort，提供可控取消窗口 |
| `scripts/c7/summarize_stage4.py` | 只读审查证据；缺失结果、跳过、失败或容差变化不能产生有界验证 PASS |

## 数值验收条件与结果

容差沿用[阶段 1 预定条件](stage1-results.md)：Attention `atol=0.02, rtol=0.02`；
logits `atol=0.1, rtol=0.01`。判断公式为 `abs(actual-reference) <= atol + rtol*abs(reference)`，
没有根据本轮结果调整容差。

验收顺序如下：

1. 每个请求的绝对 position 由 `cached_len .. device_len-1` 独立生成；新写位置唯一，不能覆盖任何请求历史 KV。
   从请求页表独立构造逻辑物理页序列，与 FI packed indices 比较。
   LMHead 的预期采样 UID/行由完整原始 Prompt 长度决定，不直接复用 sample_mask 作为参考答案。
2. 在每层真实 Attention 后捕获 post-RoPE Q、该层真实逻辑 K/V 和实际输出。
   独立 FP32 CPU 实现 GQA、缩放、按请求右对齐 causal mask、softmax 与 V 聚合。
   参考直接按请求页表取 KV，不复用 FI 的 packed 索引展开或 causal 实现。
3. 对每次实际 Forward，**只在测试中**增加一次关闭 BatchLayout/压缩 head 路径的完整模型参考 Forward。
   保持 packed inputs、positions、逻辑历史 KV 与写位置相同；不推进 Req、不额外调用 Sampler、不产生用户输出。
   这样比较原算子路径和 Mixed 路径，隔离不同批形状或 KV 来源造成的 BF16 差异。
4. 对真实 BF16 hidden 与真实 BF16 head 权重再做 FP32 CPU 线性投影参考。
   这是 head 的独立算术参考，**不是独立 FP32 全模型**。完整模型参考复用已有模型算子，
   Attention 数学由上一项独立校验。
5. 比较自然 greedy 输出；功能矩阵中的 EOS/生命周期触发另用确定性 logits，不能拿强制 Token 作为数值证据。

数值运行覆盖 cold Mixed、多个 Prefill、Prefix seed、真实 Prefix hit（13 个已缓存输入 Token）。
共 15 次实际 Forward，含 Prefill/Mixed/Decode；每次额外的参考 Forward 仅存在于测试脚本。
生产 Engine 仍然每批只执行一次模型 Forward。

| 比较 | 次数 | 最大绝对误差 | 超差元素 |
| --- | ---: | ---: | ---: |
| 28 层 Attention × 15 次实际 Forward，FP32 CPU 参考 | 420 | 0.1186180115 | 0 |
| FP32 CPU LMHead 投影 | 14 | 0.0576190948 | 0 |
| 匹配 packed 输入/逻辑 KV 的完整模型 logits | 14 | 0.03125 | 0 |
| 合计 | **448** | 分别按对应 atol/rtol 验收 | **0** |

一次中间 Chunk 没有需要采样的行，因此没有 logits/head 数值比较。
Attention 最大绝对误差大于 0.02 仍可能满足相对误差项，不能将表格解释为绝对误差全部小于 0.02。
逐比较误差分位数、容差、层号、阶段与超差数均保存在[数值日志](evidence/stage4-numeric-initial.txt)。

数值探针有同步，使用 Eager/无 Overlap；异步生命周期正确性由下述独立功能矩阵验证。
未用这些探针测性能；未对 Graph replay 内部做逐层 FP32 捕获。

## 开关 × Overlap × Graph 功能矩阵

8 组普通配置覆盖三个布尔开关的笛卡尔积，每组另起进程，顺序运行。
统一参数：本地 Qwen3-0.6B、BF16、TP=1、FI/page_size=1、context=128、3 个请求槽、768 个 KV Token、
Token Budget=7（不做对齐）。Graph 开启组使用测试参数 `[1,2,3]`，关闭组用 `[]`；
这些是测试输入，没有修改生产 Graph sizes、捕获结构或 memory pool。

每组的 8 个场景：

| 场景 | 主要断言 |
| --- | --- |
| cold 单请求自然 greedy | 4 个输出，核对资源与实际输入 |
| cold 短/长多请求自然 greedy | 6/3 个输出；Mixed 开启时确有真正混合 Forward |
| 两请求共享 19 Token Prompt，cold | 多轮 Chunk、重复 Prefix、不同输出长度 |
| 保留上述缓存的 warm 请求 | 首批命中 18 Token，3 个有效输出 |
| 127 Token 输入 + 1 输出 | 边界长度、多轮中间 Chunk、空采样 |
| 125 Token 输入，请求 9 输出 | context=128 时合法截断为 3 输出 |
| 两个 Decode，Budget 临时降为 1 | 两个活跃 Decode 都保留，新 Prefill 等待可用预算 |
| 交错到达/重复 Abort/完成/复用 | 12 轮，每轮 3 请求，1 个在中间 Chunk 后 Abort，其余输出 2/3 Token |

每个场景/每轮 churn 排空后检查：pending 为空、3 个请求槽归还、Mixed 活跃代际为空、
Prefix ref_count 与 protected_size 为 0、空闲页唯一、空闲页与缓存页不相交、总页数守恒。
实际 packed 输入逐 Token 对照原 Prompt 和已接受输出；Mixed 请求恰好完成一次，中间 Chunk 不写入假生成 Token。
Mixed 永不走 Graph，开启 Graph 的配置确实有 Pure Decode replay，关闭组没有 replay。

普通 8 组每组 **47 个 UserMsg**；请求类型、UID、完整输入、采样参数、Abort 顺序的 SHA256 均为：

`e9107a761b08a28095eff642a9c6b221498f83cb9683a9dedce737671f15fea5`

到达按相同调度步注入，不主张不同策略下的墙钟到达时间相同。普通 8 组的自然 greedy 输出全部一致：

- 单请求：`[12095,13,576,6722]`。
- 多请求 UID 1：`[12095,13,576,6722,315,9625]`。
- 多请求 UID 2：`[44801,279,3409]`。

另在 Mixed/Overlap/Graph 全开配置做 **80 轮 churn**，240 个交错请求，80 个中间 Chunk Abort，
并重跑其余 7 个场景。合计 **9 个进程、72 个场景、627 个 UserMsg**；其中 churn 528 请求，
176 个被 Abort 的请求，每个还收到重复 Abort。矩阵合计 163 次 Mixed Forward、324 次 Graph replay。
所有约定的功能断言通过。

矩阵最高 PyTorch allocated **1,625,347,072 bytes**，reserved **1,723,858,944 bytes**。
这是 PyTorch 进程内统计，不是整卡显存或外部进程统计；本轮未发生 OOM。
不从测试运行时间推断 Serving 性能或持续稳定性。

证据：[逐配置退出状态/命令](evidence/stage4-matrix-runs.txt)、[矩阵汇总](evidence/stage4-matrix-summary.txt)、
[80 轮详细日志](evidence/stage4-matrix-m1-o1-g1-stress.txt)。各配置详细日志位于同目录 `stage4-matrix-m*-o*-g*.txt`。

**关闭 Mixed 的旧 Overlap 路径共观察到 42 个额外 finished 标记**（Token 没有重复，普通两组 Overlap 各 21 个）；
关闭 Overlap 没有该现象，Mixed 各组均为 0。功能脚本显式记录该既有行为，
并没有把原始基线判为“完成语义全部正确”，也没有为通过对照修改 Baseline。
阶段 3 已保留原基线回归证据，本阶段生产源码完全未变。后续指标不能按 finished 标记条数直接统计请求成功率。

## HTTP/SSE 与回归

真实服务使用上述小容量配置、Mixed 开启、默认 Overlap、Graph 关闭：

- 同时提交短 Prompt/长输出和长 Prompt/短输出，SSE 均恰好一个 role chunk、一个 finish_reason、一个 `[DONE]`。
- Prefix 命中后，相同短请求的 SSE 文本与非流式 JSON 文本相同。
- 流式请求收到首个 JSON chunk 后关闭连接；日志确认实际后端 `AbortBackendMsg` 处理时请求仍活跃，
  处理后活跃数为 0、空闲槽为 3；随后非流式请求成功，输出仍相同。
- 服务日志记录 **11 次 Mixed Forward**，不是仅通过 API 文本猜测调度路径。

为确定性覆盖前端约 100ms 的 Abort 延迟，**仅测试入口**在每次 Forward 返回后施加 10ms CPU 延迟，
使断连发生时请求仍活跃。延迟不在生产代码中，HTTP 测试不产出延迟/吞吐指标。
HTTP 未覆盖 Graph；Graph/Overlap 组合由离线真实 Engine 矩阵覆盖。没有验收原 API 的 usage 计数或 finish_reason 分类。

证据：[HTTP 结果及实际命令](evidence/stage4-http-tests.txt)、[服务端 Forward/Abort 日志](evidence/stage4-http-server.txt)。
结束时测试进程组正常退出，没有遗留本次服务。

| 既有套件重新运行 | 结果 | 证据 |
| --- | --- | --- |
| CPU 调度/缓存/采样/生命周期 | **73 passed，0 skipped** | [日志](evidence/stage4-cpu-tests.txt) |
| Qwen3-0.6B 真实执行/生命周期 | **12 passed，0 skipped** | [日志](evidence/stage4-execution-tests.txt) |
| FI 小张量 Attention | **22 passed，39 次比较无超差，0 skipped** | [日志](evidence/stage4-attention-tests.txt)、[汇总](evidence/stage4-attention-summary.txt) |

12 项 GPU 回归还覆盖 EOS/ignore_eos、最后一个输出的完成、5 类在飞 Abort、槽位复用、
Prefix 去重、真实非 greedy Sampler 参数与 Decode 超预算。确定性触发用例在真实 Sampler 前设置 logits，
模型 Forward 仍真实执行；不将这些用例当作自然输出或统计随机采样质量证明。

## 原始数值失败仍然存在

本轮重新运行两个独立诊断，得到与阶段 3 相同的结果；容差仍是 `0.1 + 0.01*abs(reference)`：

1. **Mixed 关闭的 Baseline 单请求 vs 三请求批大小：FAIL。**
   首输出 max_abs=0.28125，6,780 个元素超差；后两步也超差，greedy Token 一致。
   该脚本仅测量，exit 0 不能解释为数值通过。
2. **原始 Baseline 无 Overlap vs Mixed 有 Overlap 的 Prefix-hit 对照：FAIL，exit 1。**
   前两个输出 logits max_abs=0；第三个 max_abs=0.21875，1,512 个元素超差。
   Token 仍同为 `[198,13874,19324]`。
   KV 诊断复现 Baseline 保留重算 Prompt 页、Mixed 使用 canonical 页的区别；
   仅在诊断控制组把 Baseline 页表改为相同 canonical KV，三步 logits 才逐位一致。
   该控制没有修改服务端 Baseline，也不能使原始失败转为 PASS。

证据：[跨批形状测量](evidence/stage4-baseline-logits-diagnostic.txt)、
[Prefix/KV 来源失败与控制](evidence/stage4-prefix-kv-diagnostic.txt)、[实际退出码](evidence/stage4-diagnostic-exits.txt)。

本阶段新增匹配 packed 形状/逻辑 KV 来源的严格验收，证明该条件下执行、采样行与 Attention 数学符合参考。
跨批形状或不同 KV 来源的通用 logits 数值验收仍未建立，不能声称“所有 A/B logits 均通过”。
如果后续需要将此作为发布前的硬性门槛，应先继续研究 BF16 的形状/缓存来源差异，不应直接放宽阈值。

## 失败、修正与验证边界

- HTTP 首轮：[客户端失败日志](evidence/stage4-http-initial.txt)、[对应服务日志](evidence/stage4-http-server-initial.txt)。
  原因是现有环境没有 `httpx` 模块。改为标准库 `urllib.request` 与线程并发后通过；未安装包。
- 格式工具最初尝试在线解析时索引连接超时，后续使用已有缓存的 `uvx --offline`，未升级工具或推理环境。
- GPU 回归保留 10 条第三方警告，Attention 回归保留 7 条，没有隐藏或改动依赖消除警告。
- 原始日志保留；仅规范化行尾空白供 Git whitespace 检查。
- 80 轮 churn 是有界压力测试，不是 24 小时稳定性测试；没有随机采样分布的统计检验。
- 只验证当前 3060 Laptop、Qwen3-0.6B BF16、TP=1、FI/page_size=1；
  没有验证 5060 Ti、TP>1、其他模型/dtype/backend、page_size>1。
- 未验证完整 FP32 独立模型、Graph 内逐层 FP32 参考，也未进行性能实验。

汇总脚本输出[有界验收汇总](evidence/stage4-summary.txt)：`bounded_acceptance=PASS`；
同时保留两个 raw logits `FAIL`、通用 logits `NOT_ESTABLISHED` 和旧路径额外 finished 的计数。
这是一组有明确条件的验证结果，不是无条件的全通过声明。
Black 24.10.0、Ruff 0.11.7、Python 编译检查、生产源码零差异和依赖一致性检查均通过，
实际命令与退出码见[静态检查记录](evidence/stage4-checks.txt)。

## 复现命令

以下模型命令逐个运行，不并行驻留 GPU；默认参数即本轮使用值。脚本会覆盖同名证据文件，
如需保留旧实验，应先复制 `docs/c7/evidence/stage4*`。

```bash
cd /home/wty238388/ai-infra/Serving_V1/mini-sglang-mixedBatch
export PATH="/home/wty238388/ai-infra/Serving_V1/mini-sglang/.venv/bin:$PATH"
export PYTHONPATH="$PWD/python"

CUDA_VISIBLE_DEVICES='' python -m pytest -o addopts='' -p no:cacheprovider -q \
  tests/core/test_mixed_lifecycle.py tests/core/test_mixed_scheduler.py tests/core/test_cache_allocate.py \
  > docs/c7/evidence/stage4-cpu-tests.txt 2>&1
python -m pytest -o addopts='' -p no:cacheprovider -q -s tests/engine/test_mixed_execution.py \
  > docs/c7/evidence/stage4-execution-tests.txt 2>&1
python -m pytest -o addopts='' -p no:cacheprovider -q -s tests/attention/test_mixed_flashinfer.py \
  > docs/c7/evidence/stage4-attention-tests.txt 2>&1
python scripts/c7/summarize_attention.py docs/c7/evidence/stage4-attention-tests.txt \
  > docs/c7/evidence/stage4-attention-summary.txt

python -u scripts/c7/stage4_numeric.py > docs/c7/evidence/stage4-numeric-initial.txt 2>&1
python -u scripts/c7/run_stage4_matrix.py --waves 12 --stress-waves 80 \
  > docs/c7/evidence/stage4-matrix-runs.txt 2>&1
python -u scripts/c7/stage4_http.py > docs/c7/evidence/stage4-http-tests.txt 2>&1

# 原始诊断：分别保存实际退出码；第二个在本环境为 exit 1，不应忽略。
python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

runs = []
for name, command in [
    ('baseline-logits', [sys.executable, '-u', 'scripts/c7/diagnose_logits.py',
                         '--model', '/home/wty238388/ai-infra/models/Qwen3-0.6B']),
    ('prefix-kv', [sys.executable, '-u', 'scripts/c7/diagnose_prefix_logits.py']),
]:
    with Path(f'docs/c7/evidence/stage4-{name}-diagnostic.txt').open('w') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=120)
    runs.append(dict(name=name, command=command, exit_code=result.returncode))
Path('docs/c7/evidence/stage4-diagnostic-exits.txt').write_text(json.dumps(runs, indent=2) + '\n')
PY

uv pip freeze --python "$(command -v python)" > docs/c7/evidence/stage4-dependencies.txt
diff -u docs/c7/evidence/dependencies.txt docs/c7/evidence/stage4-dependencies.txt
python scripts/c7/summarize_stage4.py > docs/c7/evidence/stage4-summary.txt

uvx --offline --from black==24.10.0 black --check scripts/c7/stage4_*.py \
  scripts/c7/run_stage4_matrix.py scripts/c7/summarize_stage4.py tests/engine/test_mixed_execution.py
uvx --offline --from ruff==0.11.7 ruff check scripts/c7/stage4_*.py \
  scripts/c7/run_stage4_matrix.py scripts/c7/summarize_stage4.py tests/engine/test_mixed_execution.py
git diff --check
git diff --exit-code ad59360 -- python/
```

## 下一阶段续接提示词

> 工作区为 `/home/wty238388/ai-infra/Serving_V1/mini-sglang-mixedBatch`，分支 `c7/mixed-batching-stage1`。
> 开始阶段 5：先阅读 `docs/c7/stage4-results.md`、数值失败与证据汇总，核对当前 SHA/工作区/依赖。
> 准备台式机可复现的 Benchmark 脚本与实验计划；不在笔记本上以性能数字判断成功。
> 保留原生 Prefill-first 与 Mixed 的独立配置，只有存在可独立配置的 Decode-priority 才增加该对照。
> 请求集合、到达、长度、随机种子、统计窗口必须可比；覆盖低负载、近饱和、过载及长 Prefill/持续 Decode。
> 指标按原任务记录，Success Rate 按唯一 UID 统计并检测重复 finished；先核对 API usage/finish_reason 的现有语义。
> 不预设收益，不升级核心依赖。5060 Ti 上重新检查 FI kernel/Graph 支持；保留当前 raw logits FAIL 的数值边界。
