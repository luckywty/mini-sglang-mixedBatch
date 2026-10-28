# C7 阶段 1：选批实现、验证结果与续接

本轮完成阶段 0、阶段 1，停在模型执行适配之前。**还不能运行真正的 Mixed Forward**。
阶段 0 的环境、源码审计、phase 分支枚举、不变量和原执行流见 [stage0-audit.md](stage0-audit.md)。

## 已实现的行为

1. 默认 `enable_mixed_batching=False`，保持原有 Prefill-first 选批表达式和原 Forward。
2. 新 CPU 选批入口 `schedule_mixed_batch` 首先取得全部活跃 Decode（沿用 UID 排序），
   再将 `max(0, total_token_budget - num_decode)` 交给现有 PrefillManager。
3. Decode 在前、Prefill 在后，返回**一个** Packed Batch；没有调用模型、没有两次 Forward 冒充混批。
   单阶段时保留 `prefill` 或 `decode`，两者同时出现才是 `mixed`。
4. Decode 数达到或超过预算：保留全部 Decode，Prefill 为零；有效预算仅该轮允许达到 Decode 数，
   配置预算不变。非法非正预算、Decode Query 长度不等于 1 会在 Prefill 准入之前报错。
5. 原 PrefillAdder 继续负责完整输入/输出 KV 预留、请求槽、Prefix lock 后容量重检、Chunk 续算和队列次序。
   多个 Prefill 共享剩余预算，不做任何 16/32/128 对齐。
6. 仅新路径生成不可变 `BatchLayout`，按请求记录 UID、槽位、阶段、Q/KV 长度和 sample_mask。
   `qo_indptr`、`sample_req_indices`（压缩输出行到原请求行）、`sample_token_indices`（packed hidden 行）
   均以调度快照为准，不受后续 `complete_one()` 或输出长度耗尽影响。
7. `Req.needs_sampling=True`；`ChunkedReq.needs_sampling=False`。
   **这是采样契约，尚未接入原 Sampler**。基线原有的中间 Chunk 虚假采样/-1 写回保持原样。
8. CLI `--enable-mixed-batching` 默认关闭；开启配置只接受显式 fi、TP=1、page_size=1、正预算。
   阶段 1 中 `Scheduler(config)` 在创建 Engine 前明确抛出 NotImplementedError；
   `_prepare_batch` 也拒绝 Mixed phase 或任何新 layout，包含新策略选出的纯阶段 Batch。
   因而不能靠偶然走通纯 Prefill 误以为 Mixed 服务已经可用。

例：3 Decode + 尚未完成的长 Prefill，预算 128：

```text
phase                = mixed
req_phases           = (decode, decode, decode, prefill)
query_lens           = (1, 1, 1, 125)
qo_indptr            = (0, 1, 2, 3, 128)
sample_mask          = (True, True, True, False)
sample_req_indices   = (0, 1, 2)
sample_token_indices = (0, 1, 2)
```

若这个 Prefill 的完整剩余输入恰好只有 125，最后一行也采样，hidden 行索引为 127。

## 修改文件及关键位置

| 文件 | 关键代码/职责 |
| --- | --- |
| `python/minisgl/core.py` | `Req.needs_sampling`、`BatchLayout`、`Batch.layout`、`Batch.is_mixed` |
| `python/minisgl/scheduler/mixed.py` | `schedule_mixed_batch`，独立于 Engine/GPU 的统一选批 |
| `python/minisgl/scheduler/prefill.py` | `ChunkedReq.needs_sampling` |
| `python/minisgl/scheduler/config.py` | 开关及受支持组合校验 |
| `python/minisgl/scheduler/scheduler.py` | `_select_next_batch`、启动/prepare 阶段执行门禁 |
| `python/minisgl/server/args.py` | 明确的 CLI 开关和阶段 1 提示 |
| `tests/core/test_mixed_scheduler.py` | 51 项新 CPU 用例（包含参数化用例） |
| `scripts/c7/audit_env.py` | 当前解释器/源码位置、环境和 wheel API 记录 |
| `scripts/c7/baseline_smoke.py` | 小内存 BF16 基线验证及选批/Graph trace |
| `docs/c7/` | 阶段审计、结果、失败记录、原始日志和续接提示词 |

没有修改 Attention、LMHead、Engine、Sampler、Graph、DecodeManager 或 CacheManager 的实现。

## 已执行的验证

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 原始基线 CPU cache tests | 6 passed | [baseline-cpu-tests.txt](evidence/baseline-cpu-tests.txt) |
| 新调度/布局 + 原 CPU cache tests | **57 passed，0 skipped** | [stage1-cpu-tests.txt](evidence/stage1-cpu-tests.txt) |
| Qwen3-0.6B Eager / 无 Overlap，修改前和修改后 | 均 PASS | [修改后日志](evidence/stage1-baseline-eager-no-overlap.txt) |
| Qwen3-0.6B Decode Graph [1,2] / Overlap，修改前和修改后 | 均 PASS，Decode 确实使用 Graph | [修改后日志](evidence/stage1-baseline-graph-overlap.txt) |
| 两组基线的修改前后 trace、Token、峰值分配和完成检查 | MATCH | [对照结果](evidence/stage1-baseline-comparison.txt) |
| Black 24.10.0、Ruff 0.11.7、`git diff --check` | 全部 PASS，核心环境依赖清单未变 | [检查日志](evidence/stage1-checks.txt) |

CPU 测试覆盖全部要求的十类调度场景，以及真实 radix 命中、Prefix 锁后容量失败回滚、
累计 KV 预留、槽位用尽、非对齐 Chunk、最终一次输出的采样映射、不可变快照和 Graph 排除 Mixed。
对 Q=[1,1,1,125] / KV=[501,1001,2001,125] 检查了位置、token_pool 输入映射、真实 allocate_paged 的
物理页索引范围及不重叠性；**未对 Attention 输出做数值比较**。

CPU 测试仅将 pinned Tensor 分配/拷贝替换为普通 CPU 存储，给 rank-aware logger 设置 TP=1 元数据，
关闭 TVM 可选 CUDA DLPack 探测；没有替代准入、Prefix 匹配、页分配或选批逻辑。
每个新用例结束都断言 `not torch.cuda.is_initialized()`；最终运行时还设置了 `CUDA_VISIBLE_DEVICES=''`。
Abort 用例只覆盖无 GPU 在飞的 Pending/Chunk 释放和槽位复用，不能证明 overlap 的 Abort 安全。

基线 smoke 限制为 256 上下文、2 请求、512 KV Token、16 Token Chunk、每请求 greedy 生成 4 Token，
运行后检查活跃集合清空、请求槽回收和 CacheManager 总页数。两组结果均为
`[12095,13,576,6722]`、`[44801,279,3409,23811]`。
Eager 和 Graph 的 PyTorch 峰值分配分别为 1,423,837,184 / 1,432,887,296 bytes（不是总显存）。
这些数值只说明本次 smoke 资源规模，不用于判断性能收益。

### 失败及修复记录

- 原始基线首次运行找不到 ninja：补全已有环境的 PATH 后通过，未安装依赖。
  [完整失败](evidence/baseline-initial-path-failure.txt)。
- 第一轮新 CPU 用例在实际 radix 匹配导入 TVM 的可选 torch-c-dlpack 扩展时，
  `torch.cuda.is_available()` 触发 native abort（exit 134）；不是通过/跳过。
  [堆栈](evidence/stage1-cpu-tests-initial.txt)。
  CPU fixture 用安装的 TVM 自带 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` 关闭此可选模块，仍执行真实 C++ radix 比较。
  该 fixture 只在测试生效，未修改依赖包或 GPU runtime。
- 第二轮为 52 passed、2 failed：Abort 测试缺少 Engine 正常会设置的 TP 元数据，logger 报
  `TP info has not been set`；[失败记录](evidence/stage1-cpu-tests-retry.txt)。
  补齐 CPU fixture 的 TP=1 元数据，随后全部通过；之后又补了 3 个边界用例，最终 57 项通过。
- 首次 Ruff 发现 3 处导入排序和 1 处 dict 构造风格，均已修正。
- 失败日志中的 pytest 行尾空白已去除，以通过 Git whitespace 检查，错误信息和堆栈完整保留。
- 首次 git commit 因本机未配置作者身份失败。提交使用命令级
  `-c user.name=Codex -c user.email=codex@localhost`，没有修改仓库或全局 Git 配置，没有冒用用户身份。

## 可复现命令

从本仓库根目录运行；不要直接用系统 Python，也不要省略 PYTHONPATH。

```bash
export PATH="/home/wty238388/ai-infra/Serving_V1/mini-sglang/.venv/bin:$PATH"
export PYTHONPATH="$PWD/python"
python scripts/c7/audit_env.py
CUDA_VISIBLE_DEVICES='' python -m pytest -o addopts='' -p no:cacheprovider -q \
  tests/core/test_mixed_scheduler.py tests/core/test_cache_allocate.py
python scripts/c7/baseline_smoke.py --model /home/wty238388/ai-infra/models/Qwen3-0.6B --disable-overlap
python scripts/c7/baseline_smoke.py --model /home/wty238388/ai-infra/models/Qwen3-0.6B --graph-bs 1 2
```

`-o addopts=''` 关闭仓库默认全量 coverage 输出，避免生成 htmlcov；不是跳过用例。
格式工具通过隔离的 uv tool cache 调用，核心 Python 环境未变：

```bash
uvx --from black==24.10.0 black --check python/minisgl/core.py \
  python/minisgl/scheduler/{config,prefill,scheduler,mixed}.py python/minisgl/server/args.py \
  tests/core/test_mixed_scheduler.py scripts/c7
uvx --from ruff==0.11.7 ruff check python/minisgl/core.py \
  python/minisgl/scheduler/{config,prefill,scheduler,mixed}.py python/minisgl/server/args.py \
  tests/core/test_mixed_scheduler.py scripts/c7
git diff --check
```

## 未解决风险和阶段 2～4 计划

### 阶段 2：独立接入和验证 Mixed Attention

1. 保持服务端门禁，先做隔离的 FlashInferBackend 数值测试；只使用现有 Paged Prefill Wrapper、NHD、fa2。
2. 将 layout 的 Query/KV 长度转换为所需 int32 indptr，核对 page_table 按请求完整历史拼接、last_page_len=1、
   positions/out_loc 的范围，以及每层先 store 本层 KV 再 attention。Mixed 不补 Graph dummy。
3. 使用 Q=[1,1,1,125]、KV=[501,1001,2001,125]；分别随机打乱各请求的物理页位置，
   历史 KV 与新 KV 使用可区分内容；另覆盖 cached prefix、Query 全为 1 的 Mixed、多层 KV 和计划复用。
4. 对每请求显式构造右下对齐的 causal mask：Query 行 j 只能看到 `cached_len+j` 及之前的 KV。
   参考实现使用 FP32 PyTorch attention 计算并逐请求拼接，与 BF16 wrapper 输出比较。
   **预定容差** `atol=2e-2, rtol=2e-2`；记录 max abs/relative error、误差分布，失败不得事后静默放宽容差。
5. 通过扰动一个请求的历史 KV 验证其他请求输出不变，确认没有跨请求 Attention。
   验证连续不同批次的 plan/workspace/event 不被提前复用，不删除现有同步。

### 阶段 3：接通 Sampling 和请求生命周期

1. 仅在 Mixed 开启的路径消费 layout：LMHead 用 `sample_token_indices` gather；Sampler.prepare 只读取
   `sample_req_indices` 对应参数；Engine 采样数由有效行数确定；CPU/GPU 输出共用同一显式 UID 映射。
2. `_make_write_tuple` 只覆盖有效采样行；中间 Chunk 不生成输出、不写 token_pool 的 -1 位置。
   区分中间 Chunk 的 KV 前进和生成 Token 的 device_len 前进，不机械为中间 Chunk 生成占位输出。
   保留现有基线路径，避免顺带重构其 Sampler/Prefix Cache。
3. `_process_last_data` 按压缩输出映射遍历，按 layout 的**每请求阶段**决定 Prefix 插入；
   确认最后 Prefill 加入 Decode、中间 Prefill 留在 Pending、max_tokens/EOS/ignore_eos 行为。
4. 重点审计 overlap：下一轮可已提交才处理上一轮 EOS，旧批次不可读取被覆盖的 layout/槽位，
   在飞 Abort 不能提前复用仍被 kernel 访问的页。保持同步和预留，必要时设计最小的延迟回收。
5. 增加请求完成、EOS、Abort、页释放、锁引用、槽复用和重复完成的集成测试；
   `RadixPrefixCache.check_integrity` 本身为空，需用可检验的显式 ref_count/页集合断言补足验证。
6. 只有 Attention、Sampling、写回、生命周期全部通过后才移除两处阶段门禁。

### 阶段 4：6 GB 笔记本端到端验证

1. 固定本次模型/依赖，TP=1、BF16、fi、page_size=1；从 256 上下文、2 请求、512 KV Token 起步。
2. 同一请求集做开关关闭/开启对照；覆盖单/多请求、Prefix 有/无命中、多轮 Chunk、交错到达、完成、Abort。
   Overlap 开/关与 pure Decode Graph [1,2] / Eager 分别在独立进程运行，Mixed 始终 Eager。
3. 先比较确定性布局/状态/索引，再比较关键层 Attention 和有效采样行 logits，最后比较 greedy Token。
   logits 预定 `atol=1e-1, rtol=1e-2`，记录 argmax margin；近似平局的输出差异单独调查，不能把随机采样
   顺序改变造成的 Token 差异直接判为错误，也不能只靠 Token 相同宣称 Attention 数值正确。
4. 本轮**未运行任何 Mixed GPU Attention/logits/Token 测试**；未验证在飞 Abort、EOS 全组合、
   TP>1、其他 attention backend 或 RTX 5060 Ti 支持情况。

### 阶段 5 留待之后

本轮未新增或运行 Serving 性能实验。以后在 5060 Ti 重新核对依赖/API/kernel，准备同一 seed/请求集/
到达时间/配置/统计窗口的 Prefill-first 与 Mixed 对照；本分支没有独立 Decode-Priority 配置，不虚构第三基线。
覆盖低负载、近饱和、过载及长 Prefill + 持续 Decode。采集 Output Token/s、SLO Goodput、TTFT/ITL 分位数、
TPOT、成功率、Prefill 等待、Decode Forward 间隔、Mixed Forward 耗时、Graph/Eager 比例、显存和 KV 占用，
提前固定 warmup、窗口、SLO、有效请求分母及重复次数；不预设性能收益。

## 下一轮可直接使用的提示词

```text
工作区：/home/wty238388/ai-infra/Serving_V1/mini-sglang-mixedBatch
分支：c7/mixed-batching-stage1，原始基线为 9a91cfafe754aa85daee49998176275667eb58f2。
先读 docs/c7/stage0-audit.md、docs/c7/stage1-results.md 和实际 git status/log，不覆盖已有修改。
阶段 0 和 1 已完成：schedule_mixed_batch、BatchLayout、默认关闭开关及 CPU 用例；
Scheduler.__init__ 和 _prepare_batch 仍有阶段门禁，禁止未适配的 Mixed 执行。
本轮只做阶段 2：基于已安装 FlashInfer 0.6.16.post3 的 Paged Prefill Wrapper 接入/验证 Mixed Attention。
复用现有环境 /home/wty238388/ai-infra/Serving_V1/mini-sglang/.venv（PATH 加其 bin，PYTHONPATH=$PWD/python），
不升级核心依赖。按文档使用确定性 Q/KV、不同物理页/历史 KV、FP32 参考、预定 BF16 容差比较真实输出，
验证 causal 对齐、Prefix、每层 KV、无跨请求串扰和 plan 生命周期。
纯 Decode Graph 保留，Mixed 走 Eager，首版只支持 TP=1、显式 fi、page_size=1。
阶段 3 前不要撤销执行门禁，不要顺带修改全部 Sampler/Engine/输出处理。
运行阶段 1 CPU 回归（当前 57 项），报告真实 GPU 数值结果、失败、未验证项并建立阶段提交记录，然后停止。
```
