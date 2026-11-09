# 小组成员主要贡献

| 姓名 | 学号 | 主要贡献 |
| --- | --- | --- |
| Zhuoxi Zeng | 2501213387 | 确认作业需求与实验目标，协调实验资源及账户访问，提出正确性回归、差异修复与提交材料整理要求。 |
| Steven Walxim | 2501112130 | 新增 HW2 流式延迟测量模式及离线复核器：按请求索引处理并发交错的 token 数据，记录原始到达事件，计算可观测的 TTFT/TPOT，识别首包合并导致的不可测情况，并校验输出一致性及测量完整性；新增 25 项本地测试，全 HW2 suite 221 项通过。该增量已完成本地验证及本人账户下真实 Ascend TP1 实测：两档输入×两档并发×三次重复，共 18 个测量请求及 2 个 smoke 请求通过；TP2 新测量待同卡空闲双 chip。此前完成 HW1 离线证据工具、macOS 测试路径修复、已有 HW2 结果复核及本人账户下 HW1 单 chip 前置复现；不将小组已有 TP2 适配和硬件结果归为本人独立完成。 |
| 成员 3（姓名待确认） | 待确认 | 待确认 |

Steven 的贡献证据：[新增流式测量实现、测试与实验方法](docs/streaming-metrics.md)、[离线复核工具与方法](docs/offline-reproduction.md)、[2026 年 11 月 6 日单 chip 复现记录](../homework-1/rebuild/README.md)、[独立 HW1 报告](../homework-1/rebuild/report.md)。1,058 项联合测试包含小组已有的 1,057 项参数化用例及新增的 1 项初始化回归，不代表本人新编写了全部测试。

# Homework 2：双 chip TP 与通信分析

## 1. 当前结论与实验边界

**四个模型均已完成同卡双 chip TP2 连续推理，并各有通过严格判据的 TP1/TP2 配置。4B 的精度问题已修复，并经两轮独立启动复核。** 0.8B 完成两输入长度×两并发×TP1/TP2 性能及每-chip HBM 矩阵；27B 完成 TP1 正确性基线与 TP2 性能/容量测量；完成双 rank 计算/通信分析和 batch 策略实验。结论适用于所记录模型、配置和输入。

[作业要求核对表](REQUIREMENTS.md) 将原始要求与证据逐项对应；[最新实验索引](docs/final-validation.md) 是结果入口。正确性采用相同输入下 greedy IDs exact 与 selected-logprob absolute error ≤0.15，这是本项目的工程验收判据，不是课程指定容差。失败尝试保留在 [历史索引](docs/run-evidence.md)，不替代本报告的通过配置。

HW1 技术 baseline 与 [frozen patch](../homework-1/patches/ascend-qwen3.5.patch) 不变；本次改动通过 HW2 增量 patch 交付。实验使用课程服务器的私有项目/venv，未修改共享环境、终止他人作业或绕过内存均衡检查。

## 2. 双 rank、典型层切分与通信

原始 [inventory](results/server-inventory.txt) 证明 physical0/1 属于 card0 的 chip0/1；最新 0.8B/4B/9B TP2 使用该对设备。新 27B 使用 **card2 physical4/5**，[inventory](results/final-27B-inventory.txt) 提供同卡证据。每个结果文件记录配置、进程树、设备请求和源码指纹；完整日志记录两个 rank，HBM 只按本运行 child PID 匹配。Visibility 不是独占预约。

TP2 的 world/attention/dense/MoE/LM-head TP degree 均为2，PP=1，同一模型按权重/计算切分，**不是两个独立副本**。job-local `HCCL_NPU_SOCKET_PORT_RANGE=41000-41031` 避开既有设备 socket 冲突，与 rendezvous `--port 31889` 不同。FP32/BF16 AllReduce、AllGather、ReduceScatter 在两 rank 均通过 [HCCL smoke](results/hccl-port-range/result.json)；[实际 backend 回归](results/hccl-backend-tests.txt) 也通过。Torch2.9 适配使用 `all_gather_into_tensor/reduce_scatter_tensor`。

[典型层分析](docs/parallelism-plan.md) 的核心形状如下。令 T 为当前调度的 token 行数，P 为状态池槽数，R 为送入 logits head 的行数；权重按 `[out,in]`，列出逻辑形状，不含 padding。0.8B 含 18 层 GDN、6 层 full attention，每层都有 MLP。**两个 rank 采用相同的局部形状，但持有不同权重切片；这些是源码推导形状，不是 profiler 实测 shape。**

| 结构 | Rank 0 / Rank 1 各自权重 | 各 rank 的输入 → 中间/输出与状态 | 通信与复制关系 |
| --- | --- | --- | --- |
| Embedding | 词表分片 `[124160,1024]` | 复制 IDs `[T]` → 局部 `[T,1024]` | shard 外 ID 掩零，AllReduce 后得到复制 hidden |
| GDN 输入投影与 conv | 投影 `[4112,1024]`；conv `[3072,1,4]` | 复制 hidden `[T,1024]` → 本地 Q/K/V/Z 各 1024 通道及 b/a 各 8；history `[P,3072,3]` | 列并行；conv 按 Q/K/V 的 head/channel 分片，无跨-rank recurrence 通信 |
| GDN recurrence 与输出 | 输出投影 `[1024,1024]` | Q/K/V `[T,8,128]`；FP32 state `[P,8,128,128]`；本地输出 `[T,1024]` | state 的 head 维切分，slot 身份保持一致；行并行部分和 AllReduce |
| Full attention | Q/gate/K/V 合并投影 `[2560,1024]`；输出投影 `[1024,1024]` | Q/gate 各 `[T,4,256]`，K/V 各 `[T,1,256]`；各 K/V cache `[pages,page_size,1,256]` | Q/K/V、gate、KV 按 head 切分；attention 本地计算；输出部分和 AllReduce |
| MLP | gate/up 各 `[1792,1024]`；down `[1024,1792]` | 复制 `[T,1024]` → SwiGLU `[T,1792]` → 部分 `[T,1024]` | 中间维切分；down 后 AllReduce |
| Norm / residual | hidden norm `[1024]`，GDN norm `[128]` | residual `[T,1024]`；head-local grouped norm | norm 权重复制；完整 residual 在必要规约后复制 |
| LM head / sampling | 词表分片 `[124160,1024]` | 选中行 `[R,1024]` → 局部 logits `[R,124160]` | 输出路径聚合词表 logits/选择信息；实测存在 AllGather 和 token/control Broadcast |

本配置 attention TP=dense TP=2，`CommManager` 为行并行部分和选择 AllReduce；部分 reduction 在 manager 中执行，不一定紧跟在线性层函数内。不同 TP group 宽度的 ReduceScatter 路径不是本实验配置。每 rank 的 BF16 hidden AllReduce 逻辑输入为 `T×1024×2 = 2048T bytes`，不是实际链路字节数。推理状态由各 rank 的 cache allocator 管理；conv history、GDN state、MHA KV 随 head/channel 切分，不是复制整个模型状态。

## 3. 连续推理与数值结果

每组对照保持同一模型 checkpoint/cases 指纹、相同权重/激活精度和任务配置；FP32 recurrent state、greedy、禁止 prefix reuse、5/32/256 input tokens 各生成16 tokens。不同模型允许使用各自启动配置。

| 模型 | 权重/激活精度；memory fraction | 最新 TP1/TP2 对照 |
| --- | --- | --- |
| 0.8B | BF16；0.12 | c1/c2 三个 case 全通过；最大 selected-logprob error **0.0576621294** |
| 4B | FP16；**最终词表投影FP32**；0.25 | c1 三个 case 全通过，两次独立启动确认；最大误差 **0.0059497356** |
| 9B | FP16；0.45 | c1 三个 case 全通过；最大误差 **0.0058279037** |
| 27B | BF16；**0.9** | c1 三个 case 全通过；最大误差 **0.0773688555** |

严格判据仍是 greedy IDs exact，且共同 token/history 下 selected-logprob absolute error ≤**0.15**，不是 full-logit 等价。最终源码的 c1 回归证据：[0.8B](results/head-fix-regression/0.8B-validation.json)、[4B初次](results/4B-fp32-head-initial-validation.json)、[4B独立确认](results/4B-fp32-head-confirmation-validation.json)、[9B](results/head-fix-regression/9B-validation.json)、[27B](results/head-fix-regression/27B-validation.json)。[0.8B c2 对照](results/final-matrix/tp-comparison-c2.json) 与性能矩阵来自较早的匹配源码批次，不是最终源码的新采集。每组 TP1/TP2 pair 内均核验源码树、runtime/environment、模型/cases 指纹及除 TP/port 外的配置一致；不同批次的源码版本和证据保持各自绑定。

### 4B 精度修复与验证

4B 显式对 **TP1 和 TP2 同时**设置 `--dtype float16 --lm-head-compute-dtype float32`：最终词表投影在乘法前将操作数转 FP32，并保留 FP32 logits；权重存储、transformer 激活与 KV 仍为 FP16，SSM 为 FP32。该修改消除了已捕获的最终 FP16 舍入 tie。两轮独立启动均通过三个完整 16-token case，最大 selected-logprob error 为 **0.0059497356**；同一源码的 native-head 对照仍复现原失败。没有更改输入、采样规则或容差。

代价是临时 FP32 权重和更宽的词表通信：post256 child-owned HBM 从 TP1 **9581→14175 MB**、TP2 **6267/6155→8439/8327 MB**。这是含 allocator 效应的快照，不是峰值或精确 tensor 分摊；不主张性能加速。详细诊断保留为 [复现附件](docs/4B-discrepancy.md)，接口约束见 [runtime contract](../homework-1/tokenspeed/docs/lm-head-precision.md)。

### 其他配置选择与容量

FP16 路径将 conv history 改为 activation dtype，并增加显式 FP16 KV 配置；`auto` 仍为 BF16，SSM 精度独立，27 项 cache/config 测试通过。旧 9B BF16 和旧 27B 参考比较失败仍记录在 [历史审计](docs/prefix-audit.md)，不回改为成功。当前 9B 采用匹配的 FP16 TP1/TP2；27B 采用直接 TokenSpeed TP1/TP2 对照，不依赖替代参考获认可。

27B 的 fraction=0.8 失败只说明当时 cache budget 不足（−4.23 GiB），不代表物理装不下。显式 fraction=**0.9** 后成功完成 TP1/TP2；TP1 child-owned HBM 为 **53,385 MB loaded / 53,603 MB post-generation**。各阶段源码与配置按 [最终实验索引](docs/final-validation.md) 区分，性能数据不被重标为最终精度修复后的新测量。

## 4. 性能矩阵与较大模型结果

以下均为**未启用 profiler**、warm-up 后3次测量的描述性中位数；固定16 outputs/request。指标为 batch wall latency / aggregate output-token throughput，**不是 TTFT/TPOT 或纯通信时间**。共享服务器 contention 未受控，无置信区间。

| 模型 | TP | 并发 | 输入32：秒 / output tok/s | 输入256：秒 / output tok/s |
| --- | --- | --- | --- | --- |
| 0.8B | 1 | 1 | 1.232503 / 12.981709 | 1.911833 / 8.368932 |
| 0.8B | 1 | 2 | 1.429913 / 22.378982 | 3.040349 / 10.525109 |
| 0.8B | 2 | 1 | 1.447189 / 11.055917 | 2.135147 / 7.493628 |
| 0.8B | 2 | 2 | 1.606095 / 19.924098 | 3.264206 / 9.803304 |
| 27B | 2 | 1 | **3.638593 / 4.397304** | **6.038791 / 2.649537** |

[新0.8B摘要](results/final-matrix/benchmark-summary.json) 验证完整矩阵的 **36 measured request outputs** 均匹配对应 correctness IDs；源码/配置/模型/环境指纹一致。四种条件下 TP2 都较慢，不能仅据此断言全部差值都是通信时间。并发提高 aggregate throughput，同时增加 batch latency。

[新27B摘要](results/final-27B/validation-summary.json) 将成功 TP1/TP2 correctness 对照与 **6个** TP2 benchmark outputs 绑定到同一源码、BF16/fraction0.9配置，并核对吞吐公式。这支持 TP agreement 与重复性，**不是独立 full-logit accuracy certification**。旧fraction0.8 benchmark 及其 reference failure 仍单独保留，不混入新表。

## 5. 每个 chip 的归属 HBM

[新矩阵 HBM](results/final-matrix/hbm-summary.json) 与 [各模型 HBM](results/final-model-hbm-summary.json) 精确匹配 `process_snapshot_after_load.children` 与 npu-smi process rows，排除 parent/runner 和其他作业。下表各 TP2 snapshot 均找到两个 scheduler。单位为工具标注的 **MB**，仅快照、非峰值；同一 child 跨 chip 的 context 分配保留，不按进程顺序推断 rank。

| Benchmark | 设备 | Loaded chip0 / chip1 | Post-input32 | Post-input256 |
| --- | --- | --- | --- | --- |
| 0.8B TP1 c1 | card0 physical1 | — / 2305 | — / 2529 | — / 2529 |
| 0.8B TP1 c2 | card0 physical1 | — / 2305 | — / 2533 | — / 2533 |
| 0.8B TP2 c1 | card0 physical0,1 | 2400 / 2288 | 2625 / 2513 | 2627 / 2515 |
| 0.8B TP2 c2 | card0 physical0,1 | 2400 / 2288 | 2627 / 2515 | 2629 / 2517 |
| 27B TP2 c1 | card2 physical4,5 | 28166 / 28054 | 28391 / 28279 | 28391 / 28279 |

4B修复后（FP16+FP32最终投影）的 [TP2 post256](results/4B-fp32-head-hbm-summary.json) 为 **8439/8327MB**；9B既有FP16快照为 **10841/10729MB**。新矩阵消除了交付所缺的两组内存证据，但**旧快照仍为 unknown/partial**，见 [旧JSON](results/owned-hbm-summary.json)；未以新值填旧行，也没有用设备总占用冒充模型内存。

## 6. 通信路径、时间线与 batch 实验

进程内显式 `activities=["NPU"]` 成功采集 **input32→2 output tokens、c1/c2、各两个 rank** 的 device traces；完整采集/失败历史见 [记录](docs/npu-profile-run.md)。16-token 和256-input导出仍有 vendor parser 失败，因此分析仅适用于成功短窗口；profile latency 不替代第4节的无profiler结果。

**假设：** 增大 batch 可让同一 collective 传输更多 token 的数据，摊薄启动开销，但不保证同步时间减少。 **改变因素：** 并发1→2，模型0.8B/TP2/BF16及input32保持不变。 **测量：** 每 rank、每request窗口均为 **102 collectives（98 AllReduce、2 AllGather、2 Broadcast）**；实际 decode-sized BF16 AllReduce input **2048→4096bytes**，prefill input 65536→131072bytes，是逻辑payload而非链路流量。

| 并发 | Rank | Compute union ms | Communication-envelope union ms | Communication ms/output token |
| --- | --- | --- | --- | --- |
| 1 | 0 | 92.499580 | 26.079120 | 13.039560 |
| 1 | 1 | 94.123420 | 63.765240 | 31.882620 |
| 2 | 0 | 161.496960 | 89.631500 | 22.407875 |
| 2 | 1 | 160.328520 | 20.709020 | 5.177255 |

依据 [通信分析](docs/communication-results.md) 与 [JSON](results/npu-communication-analysis.json)，采用 request-clipped interval unions，所选 compute 与 HCCL envelope overlap=0；**不把各 rank 或重叠类别时间相加当端到端耗时**。等待/控制相关的可观测量如下，便于区分设备工作与未覆盖时间：

| 并发 / rank | Request wall ms | Compute ∩ comm ms | 控制事件 union ms | 控制独占 ms | 未分类间隙 ms |
| --- | --- | --- | --- | --- | --- |
| 1 / 0 | 315.556720 | 0 | 24.611200 | 0.559940 | 196.418080 |
| 1 / 1 | 315.556720 | 0 | 68.441320 | 0.407620 | 157.260440 |
| 2 / 0 | 453.437730 | 0 | 99.037920 | 0.681780 | 201.627490 |
| 2 / 1 | 453.437730 | 0 | 19.448580 | 0.496240 | 271.903950 |

控制事件包括 device EVENT/NOTIFY WAIT/RECORD 等；完整 control union 可与其他流的工作重叠，只有“控制独占”进入无重叠分区。分区恒等式为 `wall = compute-only + comm-only + overlap + control-only + gap`。未分类间隙还含未分桶的辅助活动、传输和主机调度，不能称为纯等待。HCCL envelope 也不穷尽所有通信辅助 kernel 的忙碌时间。每 rank 的 51 个 post-request tail collectives 均排除。

**结论：** batch2 将固定 collective 次数摊到更多输出，但 rank0 normalized communication time 变差、rank1 变好。每并发仅一个 warmed profile window，加上共享 contention，不能声称普遍通信加速或因果解释。四份 gzip trace 共 10,952,799 bytes，可离线复算。

## 7. 交付与复现

[HW2 patch](patches/hccl-torch29.patch) 包含 HCCL API、NPU profiling、FP16 cache plumbing及显式FP32最终投影，共 **15 files /89,738bytes**；[manifest](patches/manifest.json) SHA256为 `76071724b56e1f7922dd3fd9a4c2c5b359f3422252ac10c74daa5e5d0f2350f6`。[验证](patches/patch-verification.json) 在固定base+frozen HW1+HW2上重建 **2972 files byte/mode-exact**，三个原有可执行源文件均保留100755，HW1 patch不变。历史3/6/11文件版本均归档。

[README](README.md) 给出服务器启动和顺序应用patch；[最新索引](docs/final-validation.md) 与 [4B诊断](docs/4B-discrepancy.md#5-reproduce-and-test) 给出离线复算命令。最终投影修复批次的历史本地suite **178 passed**，[日志](results/local-tests-head-fix.txt)；该批次LM-head40项与cache27项共 **67 passed**，[日志](results/lm-head-precision-tests.txt)；原logits-processor suite **50 passed/1 skipped**，[日志](results/head-fix-native-regression-run.txt)。既有真实NPU profiler tests为10 passed，profile兼容子集29 passed/1 deselected；不声称完整upstream suite通过。包含 Steven 后续工具的当前本地测试范围见第 8 节。

**原始作业要求的对应关系：** 任务1（四模型双 chip 适配）见第2–3节；任务2（并行正确性）见第3节；任务3（通信路径及耗时）见第2、6节；任务4（性能/HBM 与通信策略对比）见第4–6节。4B 修复已经完成，详细诊断仅为可追溯附件，不是未解决事项。

结果保留有限输入、共享服务器、短 profile 与 HBM 非峰值等边界。对应源码、原始结果与复现脚本随附件交付。

## 8. Steven 新增的流式测量工具与 TP1 硬件验证

既有第 4 节的 batch wall latency 无法区分首 token 等待与后续生成速度。新增 `--mode streaming` 保留每个请求的原始 token 到达事件，支持 TP1/TP2 和并发 1/2；按请求索引及 ID 解复用交错的数据，逐项校验 token 数、终止状态、无缓存条件和 warmup 输出一致性。离线复核器重新计算指标，拒绝缺失样本、被改动的摘要和旧的非流式结果。

TTFT 仅在首个非空 chunk 含一个 token 时提供；若首包已合并多个 token，则记为不可测，不以首包时间冒充精确 TTFT。TPOT 使用 `(末 token 到达时间 − 首 token 到达时间)/(N−1)`；空的结束帧不改变末 token 时间。各条件的指标中位数要求全部请求均可观测，否则只报告覆盖数量。这些是包含调度、IPC 和观测开销的客户端指标，不等同于纯 prefill/decode 或通信耗时。

新增 25 项合成测试通过，包括 CLI 到结果文件的完整模拟路径；与已有 196 项一起构成当前 221 项本地通过测试。详见[实现与运行方法](docs/streaming-metrics.md)和[本地验证记录](results/steven-streaming-local-validation.json)。这些合成测试不需要 NPU；真实硬件证据单独列于下方。旧实验表和 178 项历史测试记录保持原始时间及范围。

### 2026-11-09 实机结果（TP1，physical chip 1）

使用独立 HW2 runtime、Qwen3.5-0.8B/BF16/FP32 recurrent state、无 prefix reuse、每请求 16 个输出 token，在本人账户依次完成两输入长度×两并发×三次重复。18 个正式测量请求和另行保留的 2 个 smoke 请求全部通过事件重算、warmup 一致性及已保存 Transformers 参考的输出 ID 核对。另有 9 个新 correctness 请求通过精确 ID 与 selected-logprob 容差检查，最大绝对误差 0.1255553961 ≤ 0.15。该数值判据仍是项目自定标准。

| 输入 token 数 | 并发 | 测量请求数 | TTFT 中位数 ms | TPOT 中位数 ms | Batch wall 中位数 s |
| --- | --- | --- | --- | --- | --- |
| 32 | 1 | 3 | 251.028 | 68.057 | 1.274354 |
| 256 | 1 | 3 | 1011.539 | 66.672 | 2.023089 |
| 32 | 2 | 6 | 342.099 | 75.906 | 1.482909 |
| 256 | 2 | 6 | 1461.425 | 107.698 | 3.143219 |

并发 2 的 TTFT/TPOT 中位数汇总三个 batch 中的六个请求；同 batch 的请求并非独立实验重复。全部正式样本均收到单 token chunk，TTFT/TPOT 观测覆盖完整。单 worker HBM 快照为加载后 2299 MB，生成后并发 1 为 2525 MB、并发 2 为 2527 MB，按本次 scheduler PID 匹配，非峰值。

[硬件复核及指标](results/steven-streaming-20261109/hardware-verification.json)、[HBM 归属](results/steven-streaming-20261109/hbm-summary.json)与[原始证据清单](results/steven-streaming-20261109/evidence-manifest.json)保留源码/输入指纹、状态、压缩日志和原始流事件。共享服务器的其他 chip 有既存任务；本次只使用当时无列出进程的 chip 1，逐次检查且结束后释放。所有同卡双 chip 对均至少有一个 chip 被其他任务占用，因此没有运行新 TP2 流式实验。以上支持新增测量工具的 TP1 实机验证及客户端延迟分析，不构成 TP2 加速或通信耗时结论；小组原有 TP2 证据保持独立。
