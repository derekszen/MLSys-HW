# 小组成员主要贡献

| 姓名 | 学号 | 主要贡献 |
| --- | --- | --- |
| Zhuoxi Zeng | 2501213387 | 确认作业需求与实验目标，协调实验资源及账户访问，提出正确性回归、差异修复与提交材料整理要求。 |
| Steven Walxim | 2501112130 | 补充离线证据复核工具及 18 项回归测试、修复 macOS 路径比较问题；在本人账户重建单 chip 环境并核验模型文件，重新运行 Transformers 参考、1,058 项算子与初始化测试、三个输入的连续生成及请求隔离检查、12 次无缓存性能测量；新增模型初始化回归并整理独立复现报告。复用小组冻结适配与实验脚本，不将原有实现或全部测试归为本人新编写。 |
| 陈子睿 | 2601112045 | 完成 Qwen3.5-0.8B 模型在 Ascend 单 chip 上的适配与正确性验证，分析关键算子支持及与 Transformers 参考实现的精度差异。开展不同输入、输出长度下的单 chip 延迟与吞吐对比，统计 HBM 占用快照并分析主机侧 profile，整理实验结果。 |

Steven 的新实测见 [独立复现报告](rebuild/report.md) 与 [运行记录](rebuild/README.md)。联合测试由小组已有的 1,057 项用例加新增的 1 项初始化回归组成。本文后续技术表格保留小组原实验数据，11 月 6 日的新运行另行归档，不覆盖历史结果。

# HW1 实验报告：Qwen3.5-0.8B 在 Ascend 单 chip 上的适配与正确性基线

## 1. 阶段结论与证据范围

**单 chip 技术 baseline 与实验材料已完成。** 已有真实 NPU 日志支持：纯文本 Qwen3.5-0.8B 能在单 chip、BF16、FP32 recurrent state、greedy eager 配置下完成权重加载、prefill、decode 及连续生成；关键新增/修改路径的五组 NPU 测试最终通过，三个输入各连续 16 步 greedy token 与未修改 Transformers reference 完全一致，并建立了输入 32/256 × 输出 8/16、每组 3 次的客户端延迟基线。

此结论**不是**完整 logits 等价、任意输入正确、整个模型家族/多模态/MTP/量化/多卡已支持，也不是优化完成或独占 NPU 的性能结果。逐样本 benchmark、进程 HBM snapshots 和 CPU-only profile 已归档；未测 NPU kernel 时间分解或瞬时 HBM 峰值。

主要证据入口：

| 内容 | 已记录文件 |
| --- | --- |
| 作业要求、结构与模型来源 | [assignment.md](assignment.md)、[config.json](results/model/config.json)、[source-manifest.json](results/model/source-manifest.json) |
| 算子支持与调用链 | [operator-audit.md](docs/operator-audit.md)，含全部主要算子的功能/形状/dtype/来源/Ascend/测试矩阵与源码路径 |
| 最终关键算子结果 | [GDN](results/integrated-gdn-tests-2.txt)、[conv](results/integrated-conv-tests.txt)、[gated norm](results/runtime-import.txt)、[Gemma norm](results/integrated-gemma-tests.txt)、[MHA](results/integrated-mha-tests-3.txt) |
| 模型参考与连续推理 | [model-reference.json](results/model-reference.json)、[model-validation.json](results/model-validation.json)、[model-validation-run-3.txt](results/model-validation-run-3.txt) |
| 性能与启动内存 | [benchmark.json](results/benchmark.json)、[benchmark-run-2.txt](results/benchmark-run-2.txt)、`results/hbm-*.txt` |
| 主机 profile | [原始压缩 trace](results/profile/hw1-host-256-8-TP0.trace.json.gz)、[汇总](results/profile-summary.json)；CPU-only，不是 NPU kernel 时间 |

源码基线为 TokenSpeed 子模块 `52bdb9763bb2da3d61336398072671885cc0a33e` 加 [适配 patch](patches/ascend-qwen3.5.patch)，[manifest](patches/manifest.json) 保存 base commit 和 patch SHA256；不能只凭上游 commit 复现本实验。完整启动及 patch 应用步骤见 [README](README.md)。

## 2. 环境、固定模型与运行配置

### 2.1 实际环境与模型可追溯性

实验运行在课程服务器 aarch64 环境，设备 API 报告 `Ascend910_9362`，面向作业的 Ascend 910C 单 chip 目标。记录选择物理设备 1、映射逻辑 `npu:0`，world size/TP/PP 均为 1。**可见性选择不是设备预约**；服务器存在其他任务，没有停止其他作业，也不宣称无争用。

| 项目 | 已记录配置及其出处 |
| --- | --- |
| Python / PyTorch / torch_npu / CANN | Python 3.11.16、PyTorch 2.9.0+cpu + torch_npu 2.9.0、CANN 9.0.0；见 [早期环境快照](results/environment.json)。`+cpu` base build 搭配 NPU extension，实际 NPU 运算由后续测试验证，不意味着模型在 CPU 运行 |
| 私有实验环境 | `homework-1/.venv` 继承系统 site-packages；pytest 8.4.2、Transformers **5.12.0**、Triton-Ascend distribution **3.2.1**（其 Python module 自报 3.2.0）。见 [最终环境](results/runtime-environment.json)、[私有包清单](results/private-requirements.txt)、[约束](scripts/npu-constraints.txt)。早期快照保留，不当作最终依赖版本 |
| 模型来源 | ModelScope `Qwen/Qwen3.5-0.8B`。manifest 按文件记录 revision/hash，不是所有文件共用一个 revision；权重/config 的 revision 为 `05f056712f8c582c8708395156518d963b46fb29` |
| 权重与配置 hash | 权重 SHA256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`，config SHA256 `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`；均来自 manifest，不把 manifest 内容写成此次重新下载/哈希校验 |

### 2.2 实际模型与启动参数

实际 `text_config`：hidden=1024，24 层，**18 GDN + 6 MHA**（每四层一层 full attention）；GDN 为 16 key/value heads、K=V=128、**6144 通道 width=4 causal conv**；gated RMSNorm 是 16 组、每组 128。MHA 为 8 Q heads/2 KV heads、D=256、**partial RoPE=64/256**、sigmoid output gate。每层 dense MLP intermediate=3584，vocab=248320，embedding/head tied。配置虽有 vision/MTP 字段，本实验没有启用相应计算路径。

模型验证使用 [engine_config](scripts/engine_validate.py)：

```text
device=npu, dtype=bfloat16, language_model_only=True, trust_remote_code=False
max_model_len=512, max_num_seqs=2, max_total_tokens=16384
chunked_prefill_size=128, max_prefill_tokens=128, gpu_memory_utilization=0.12
mamba_ssm_dtype=float32, sampling_backend=greedy, seed=42
enforce_eager=True, disable_prefill_graph=True, disable_pdl=True, disable_autotune=True
enable_output_logprobs=True
```

基线测量再显式设置 `enable_prefix_caching=False`、`disable_kvstore=True`、`enable_output_logprobs=False`、`stream_interval=1`，单并发串行请求；`max_num_seqs=2` 是容量上限，不代表性能测试并发为 2。生成固定 `temperature=0, ignore_eos=True`，严格检查生成长度与 token IDs；不通过提前 EOS 或省略层降低耗时。

复现入口如下（是已有脚本的调用方式，**不是此次文档任务执行记录**）。需要先由执行者选择适当物理设备，并使用服务器已有私有环境和源码路径：

```bash
# 从解压后或 Git checkout 的项目根目录开始；记录的服务器根目录是 ~/MLSys-HW。
ROOT="$PWD"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd "$ROOT/homework-1"
source .venv/bin/activate
export PYTHONPATH="$PWD/tokenspeed/python:$PWD/tokenspeed/tokenspeed-kernel/python:$PWD/tokenspeed/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"
# 设置 NPU_PHYSICAL_DEVICE 和 MODEL_DIR 后：
python scripts/model_reference.py --physical-device "$NPU_PHYSICAL_DEVICE" \
  --model "$MODEL_DIR" --output results/model-reference.json
python scripts/engine_validate.py --physical-device "$NPU_PHYSICAL_DEVICE" \
  --model "$MODEL_DIR" --reference results/model-reference.json \
  --output results/model-validation.json
python scripts/benchmark.py --physical-device "$NPU_PHYSICAL_DEVICE" \
  --model "$MODEL_DIR" --reference results/model-reference.json \
  --output results/benchmark.json --repeats 3
```

## 3. 适配内容与关键障碍

### 3.1 哪些复用，哪些新增或修改

**直接复用：** 已有 Qwen3.5 模型结构和 checkpoint loaders、embedding、dense GEMM/projections、Ascend SwiGLU、partial RoPE/composite attention prologue、MHA BSH decode、Triton-Ascend sigmoid gate 与 prefill GDN gate、hybrid backend/LCM cache/scheduler/greedy sampling。并未从零重写整个模型，也未将独立数学原型直接称作 runtime 集成。

**本次补齐的工作树路径：**

| 类别 | 适配内容 | 设计边界 |
| --- | --- | --- |
| GDN 与 conv | 增加 Ascend public registry leaf 与 vendor-package eager 实现，接通 prefill/单步 decode、FP32 K-last state、caller-owned checkpoint、独立读写槽与 padding | 保持 runtime → `tokenspeed-kernel` → `tokenspeed-kernel-npu` 边界；无 activation/state CPU fallback；不是融合高性能 kernel |
| Gated / Gemma RMSNorm | Ascend grouped gated RMSNorm；Gemma runtime 走 vendor-neutral facade，`1+w` 用 FP32，保留 residual buffer 与输入精度舍入契约 | 不复写已有 QK norm 的 Ascend 精度例外；拒绝未支持 FP8/FP4 copy，而非静默忽略 |
| D256 MHA | prefill/extend 由 packed TND 改为逐请求 BSH native attention，保留 GQA、causal suffix mask、pages 与 custom scale | D256 prefill/extend 明确 eager-only；原 decode 与其他 D 的 TND 路径保留 |
| 模型设备初始化 | 仅 NVIDIA 构造 `torch.cuda.Stream()`；Ascend 保持同一模型 stream 顺序 | 避免在 NPU 上构造 CUDA stream，不声称新增多流优化 |

完整算子矩阵与实现文件见 [operator-audit.md](docs/operator-audit.md)。模型能调用并不自动证明每个复用算子已单独对照：embedding、projection、RoPE、sigmoid gate 等目前主要由有限 E2E 测试覆盖。

### 3.2 GDN 从模型到执行的链路

`Qwen3_5GatedDeltaNet.forward` 投影 `[T,1024]→[T,8224]`，拆 Q/K/V/Z/b/a → hybrid 中 `GdnAttnBackend` 的共享 `state/mamba.py` → causal conv public facade/Ascend registry/vendor helper → Q/K/V `[1,T,16,128]` → `gdn_chunk_prefill` 或 decode 的 `gdn_decode_step` registry → `ascend_gdn_*` → `tokenspeed_kernel_npu.ops.gdn` device-local eager 运算。最后 `[T,16,128]` 展平，做 grouped gated RMSNorm 与输出投影。

递归状态始终是 **FP32 `[pool,HV,V,K]`，K-last**。先 decay，再从 decay 后 state 预测，按 β 更新，再从更新后的 state 读出：

```text
g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)
D = exp(g) * S
r = beta * (v - D @ k)
S_next = D + outer(r, k)
y = (S_next @ q) / sqrt(K)
```

Q/K 按 `x/sqrt(sum(x*x)+1e-6)` 归一化，prefill 按 shared contract 将 normalized Q/K 舍入到输入 dtype，而 decode 保留 FP32，故低精度 chunk/step 不是 bitwise parity。state 无需在此 Ascend 路径转成 K-major；decode 仅把**输出** `[B,1,HV,V]` 转成 backend 约定的 `[1,B,HV,V]`。

LCM 拥有 conv history、recurrent state 和请求槽。prefill 保留 initial state，backend 将 final state 写到 live 目标槽；decode 在写任何目标前快照所有源，避免交叉槽写入污染另一请求，负 source 作为 padding 不写状态。conv history 只保存最近 3 个原始输入，不能保存 SiLU 后输出。独立请求不共享可写状态，也没有新增全局私有 state。

主要障碍不是更换模型名，而是**补齐缺失的厂商 dispatch 并保持布局、精度与状态 ABI**。另一个实际障碍是 native attention 在本栈拒绝 TND D256：早期 [模型启动日志](results/engine-smoke-5.txt) 确有失败，最终用 BSH 适配后才通过。最初 [GDN public-registry 测试](results/integrated-gdn-tests.txt) 也出现 Triton-Ascend 3.2.0/CANN header 编译兼容问题，私有安装 3.2.1 后有后续成功日志；没有为了消除 warning 修改共享软件所有权。

### 3.3 同步与 eager 限制

数学计算保持在 NPU；但是 conv/GDN 的 sequence offsets、slot indices、checkpoint starts/program maps 等元数据通过 **`.tolist()` 同步到 host**，Python 再逐 token/请求发射 Torch 运算。D256 MHA extend 也读取 host 元数据、逐请求 native BSH 调用。因此“无 CPU math fallback”不等于“无 CPU 调度或无 device synchronization”。本实验关闭 graph/PDL/autotune，既不宣称全模型 graph capture，也不宣称已有优化性能。

## 4. 分层正确性实验

### 4.1 关键算子/模块

| 模块 | 独立参考与覆盖要点 | 实际成功结果 |
| --- | --- | --- |
| GDN | CPU FP64 K-major per-head oracle；分组头、非零初态、变长/空序列、checkpoint/resume、chunk/step、strides、交叉/共享 source、padding、public registry | **82 passed**，81 deselected；[最终日志](results/integrated-gdn-tests-2.txt) |
| Conv | CPU FP64 grouped conv1d oracle；多宽度/dtype、两种 history layout、reset、分块/逐步、交叉槽、空序列、错误拒绝及真实 public API | **227 passed**，220 deselected；[日志](results/integrated-conv-tests.txt) |
| Gated RMSNorm | 独立 FP64 scalar oracle；group/shared weight、gate 顺序/种类、mixed dtype、strides、empty、dispatch | **144 passed**，93 deselected；[日志](results/runtime-import.txt) |
| Gemma RMSNorm | 真实 runtime layer 与 FP64 oracle；residual 舍入、buffer identity、BF16 `1+w`、weight reload、量化请求拒绝 | **59 passed**；[日志](results/integrated-gemma-tests.txt) |
| MHA | FP64 GQA oracle；8Q/2KV D256、变长/空请求、shuffled 多页 cache、causal suffix/noncausal、request/future isolation；加既有 D64/decode 回归 | **151 passed**；[最终日志](results/integrated-mha-tests-3.txt) |

这些是实际 NPU-selected 测试日志，不是 CPU 通过或 skip 冒充 NPU；early failed logs 保留，不能替代最终成功证据。最终五个 suite 联合执行 **1,057 passed、无 skip**，其中 663 个 NPU case、394 个 CPU case，见 [联合日志](results/integrated-all-tests.txt)。独立 `hw1/state_ops.py` 的 67 CPU/14 NPU 早期原型结果不加进本表充当集成测试。模型形状来自 config 与 E2E，参数化 unit suite 的小型/边界形状不冒称全部都是完整模型尺寸。

误差使用 `abs(diff) <= atol + rtol*abs(ref)`，参考读取**相同的已量化输入**。BF16 输出的 `(atol,rtol)` 分别为 GDN `(4e-3,8e-3)`、conv `(5e-4,8e-3)`、gated/Gemma norm `(2e-3,8e-3)`、D256 MHA `(3e-3,8e-3)`；GDN FP32 state 用 `(2e-6,2e-5)`，history 与非目标槽等用 exact。最终舍入允许比 FP32 state 更宽的输出误差；这些是有限数值案例的界，不是任意长序列误差保证。FP16/FP32 完整阈值和来源见 [审计第 4 节](docs/operator-audit.md#4-已记录的分层测试与误差界)。

### 4.2 完整模型连续生成

可信参考为 **Transformers 5.12.0 的未修改 `Qwen3_5ForConditionalGeneration`，eager attention，BF16，运行于同一 NPU 环境**；[reference 脚本](scripts/model_reference.py) 无 TokenSpeed import/patch。两端使用同一模型目录与保存的 input IDs；seed=42、不采样、强制生成 16 tokens、忽略 EOS。短输入是 `The capital of France is` 的 5 tokens；32/256 输入为固定英文实验语句重复后 tokenizer 输出的前缀，不是用字符数估算 token 数，不使用 chat template。

判据是 **16 步 greedy IDs 全部 exact，且每步所选 token 的 logprob 绝对误差 ≤0.15**。reference 的 score 转 FP32 后算 log_softmax。0.15 是此 BF16 跨执行路径实验选用的工程验收界，反映 reduction/舍入及 norm/RoPE 实现差异；未给出普适误差推导，不能把它提升为全词表 logits 或所有输入的等价保证。

| 案例 | 输入 / 输出 tokens | 16 步 greedy IDs | 最大所选 logprob 绝对误差 | 该次 cached tokens |
| --- | --- | --- | --- | --- |
| smoke | 5 / 16 | exact | **0.1255554**（约 0.12556） | 0 |
| length_32 | 32 / 16 | exact | **0.0408112**（约 0.04081） | 0 |
| length_256 | 256 / 16 | exact | **0.0022228**（约 0.00222） | **47** |

数据来自 [model-validation.json](results/model-validation.json)；参考 IDs、各步所选 logprob 与 top5 在 [model-reference.json](results/model-reference.json)。虽然验证脚本在案例前调用 flush，256 案例仍报告 47 cached tokens，故不能称该次为“全量 uncached 256-token prefill”。后续 benchmark 关闭 prefix cache/kvstore，对 32/256 各做一次 **uncached 16-token warmup 且 exact IDs**；四组正式测量也逐次检查 reference IDs、精确输出长度、`cached_tokens==0`，见 [脚本](scripts/benchmark.py) 与 [原始运行日志](results/benchmark-run-2.txt)。这补充了 uncached 生成证据，但没有补充 uncached logprob 对照。

有状态验证还包括同一 engine 中 **A→其他较长请求→A 重复**，重复 A 的 16 IDs 不变；flush 后 **batch=2** 同时提交 5/32-token 输入，两请求各 16 IDs 都与参考一致。它们结合算子交叉槽、非零初态和 checkpoint 测试支持有限状态隔离结论；不声称覆盖全部调度、抢占恢复、长会话或压力并发情况。

**没有进行 full-logit equivalence 验证。** reference 保存 top5，但 TokenSpeed 没有输出并对照完整词表 logits/top5；本报告只声称已测的 greedy ID 与 selected-logprob 判据通过。

## 5. 性能与 HBM 基线

### 5.1 方法：客户端 TTFT / TPOT，不是纯 kernel 时间

采用 [benchmark.py](scripts/benchmark.py)，engine 启动后对输入 32 与 256 各 warmup 一次、每次生成 16 tokens 并检查 reference。随后按输入 32/256、输出 8/16 四组合，**单并发串行、每组 3 次**，prefix cache 与 kvstore 关闭；输出 logprob 关闭以避免比较接口开销。日志中长输入由两个 128-token chunk prefill，无 cached tokens。

客户端以 `perf_counter` 记录请求开始和流式 token 到达，首个 chunk 必须恰好 1 token：

```text
TTFT = first_token_arrival - request_start
TPOT = (last_token_arrival - first_token_arrival) / (output_tokens - 1)
total = 请求开始至流式迭代结束
```

TTFT 包含 prefill、首步输出、排队/调度、IPC 与客户端交付；TPOT 是首 token 之后的客户端平均每 token 间隔，也含调度/IPC，不是 NPU event 测到的纯 decode kernel 时间。没有把 reference 的 seconds、pytest wall time 或冷启动时间当模型吞吐。由于各指标分别求 median，median total 不必恰好等于 median TTFT 加 `(N-1)*median TPOT`。

### 5.2 实测结果

下表直接抄录 **[benchmark-run-2.txt](results/benchmark-run-2.txt) 的四条 JSON summary**，单位 ms，每项为 3 次中位数；括号是该项 min–max，不是置信区间。首轮 `benchmark-run.txt` 不作为本表完成依据。

| 输入 tokens | 输出 tokens | 客户端 TTFT median（min–max） | 客户端 TPOT median（min–max） | 客户端 total median |
| --- | --- | --- | --- | --- |
| 32 | 8 | **244.514**（242.910–245.903） | **67.722**（67.668–68.886） | 719.972 |
| 32 | 16 | **237.392**（187.330–241.155） | **67.020**（65.765–67.864） | 1242.700 |
| 256 | 8 | **1009.342**（968.350–1063.655） | **67.497**（62.429–68.934） | 1446.362 |
| 256 | 16 | **978.786**（962.829–1030.778） | **66.916**（63.913–67.111） | 1982.557 |

这些数据表明本次配置下长输入 TTFT 更大，而这两档输入的 TPOT 中位数约 67 ms；仅 3 次、共享服务器条件，不能据此推导可扩展性、硬件峰值利用率或性能优化收益。[benchmark.json](results/benchmark.json) 已归档全部 12 次样本的 arrivals、IDs 和 summary，与运行日志一致。

### 5.3 HBM 口径

同一 [第二轮 benchmark 启动日志](results/benchmark-run-2.txt) 记录：

| 指标 | 已记录值 | 含义与限制 |
| --- | --- | --- |
| 启动 Torch allocated | **1.92 GiB** | runtime 表头虽写 GB，源码使用二进制口径；是当时 allocator live allocation，不是模型峰值或进程总 HBM |
| 启动 Torch reserved | **2.06 GiB** | PyTorch allocator pool，已包含 allocated，不能二者相加 |
| cache arena | **213909504 bytes = 204 MiB** | 包含该 LCM/cache plan 的设备 arena，不是另加在 allocated 之外的额外测量 |
| 进程级 `npu-smi` HBM | worker **2299 MB（启动）/2525 MB（各工作负载之后）**，frontend **112 MB** | `npu-smi` 报告单位 MB；worker PID 2588245、frontend PID 2588138，物理 card 0/chip 1。见 [启动](results/hbm-startup.txt)、[32→8](results/hbm-32-8.txt)、[32→16](results/hbm-32-16.txt)、[256→8](results/hbm-256-8.txt)、[256→16](results/hbm-256-16.txt)。这是采样时刻的进程占用，不是瞬时峰值 |

启动表还记录约 5.3 GiB driver-view device used 与 non-Torch 差值；共享设备上它们含其他进程，不能当作本模型 HBM。`gpu_memory_utilization=0.12` 是配置参数，不是 HBM 测量。allocator、cache arena 和进程统计有重叠，不能相加。未记录的峰值、prefill/decode 分阶段 HBM 不填估算值。

## 6. 基于主机 profile 的瓶颈分析

用 [profile_engine.py](scripts/profile_engine.py) 通过 TokenSpeed 自带 profiler API，在相同配置、warmup 后记录一次 **256 输入→8 输出**。prefix cache/L2 关闭，输出仍与参考一致。原始 trace 与 [summary](results/profile-summary.json) 已归档；trace 的 10 个 annotation name 含未转义 JSON，解析器仅在内存修复这些名字，原始文件保持不变并保存 SHA256。四个 [parser 单元测试](results/profile-summary-tests.txt) 通过。

**这是 CPU-only、带 profiler 开销的记录，不是 NPU kernel 时间。** 其中两个 128-token EXTEND 的 host forward span 合计 **1772.943 ms**，8 个 DECODE span 合计 **774.001 ms**。生成 8 tokens 的记录含 8 次 decode forward；当前配置启用 overlap schedule，但没有单独追踪额外 forward 的归属，不能直接用 774/7 替代无 profiler 的 TPOT。profile 会放大开销，因此性能基线只用第 5 节的无 profiler 测量。

| 主机事件 | 调用次数 | exclusive CPU 时间（ms） |
| --- | ---: | ---: |
| `empty_tensor` | 81,193 | 232.464 |
| `aten::mul` | 31,754 | 187.704 |
| `aten::as_strided` | 89,312 | 129.268 |
| `aten::select` | 49,474 | 126.602 |
| `aten::copy_` | 7,939 | 96.994 |
| `npu::_npu_dtype_cast` | 14,254 | 73.254 |
| `aten::matmul` | 970 | 10.392 |
| `npu::npu_fused_infer_attention_score` | 60 | 3.473 |

exclusive 时间按线程减去直接子调用，避免层层累计重复计算；所有 CPU op exclusive 合计约 **1569.907 ms**。矩阵乘/attention 的主机调用时间小，**不代表 NPU 计算时间也小**，因为 kernel 异步执行。本记录不提供实际 HBM 带宽、AI Core 利用率或 device compute/memory 占比。

**优先优化目标：GDN eager 路径的逐 token 小算子、临时张量与主机提交/同步开销。** 18 层递归，每次 prefill 含 Python token 循环；源码与实测数万次 mul/select/cast/临时张量事件共同支持这个方向，区别于仅凭较长 TTFT 猜测。metadata `.tolist()` 与 D256 的逐请求 native attention 也有同步/调度成本，但此 trace 没有完整模块标签，不能精确把全部 CPU 事件分摊到 GDN、conv、norm，也不能宣称已锁定最大 NPU 硬件瓶颈。

后续可融合 recurrence、复用 workspace 并减少 metadata 往返，但必须保留 FP32 state、K-last layout、读写槽隔离与相同误差检查。当前没有实现或测得此类优化 speedup；BSH 改动是支持性修复，不是性能提升实验。

## 7. 实验范围与局限

- **已完成技术部分：** 单 chip 模型计算路径、关键算子分层验证、有限输入的连续推理/状态隔离、2×2 基线、进程 HBM snapshots、主机 profile、算子审计与可重放 patch。
- **未覆盖：** 完整 logits 等价、更多非重复输入/长会话、任意上下文、graph、量化、多模态、MTP、瞬时 HBM 峰值及独占设备上的 NPU kernel 级性能分解。TP=2 属于独立的 Homework 2，未混入本报告的冻结 baseline。

**原始要求对应关系：** 环境与启动见第2节；计算路径、关键障碍、状态与同步见第3节；分层验证与连续推理见第4节；两输入×两输出长度的 TTFT/TPOT/HBM 见第5节；基于测量的优化方向见第6节；主要算子支持矩阵见附录A。源码、原始结果和复现脚本随附件交付。

## 附录 A：主要算子支持矩阵

令 T 为当前 token 行数、B 为请求数、P 为状态池槽数、R 为送入 logits head 的行数；下表均为本作业 TP1 逻辑形状，权重按 `[out,in]`。激活/权重为 BF16，另有说明的累加/state 为 FP32。详细源码位置、数值约定和测试日志见 [完整算子审计](docs/operator-audit.md)。

| 功能 | 主要输入 → 输出 / 状态 | dtype | 来源 / Ascend 状态 | 测试方法 |
| --- | --- | --- | --- | --- |
| Embedding | IDs `[T]`，权重 `[248320,1024]` → `[T,1024]` | 整数 IDs；BF16 输出 | 复用 vocab embedding；NPU 可运行 | 有限 E2E 覆盖 |
| GDN 投影 | `[T,1024]` → `[T,8224]`；输出投影 `[T,2048]` → `[T,1024]` | BF16；gate 参数 FP32 | 复用 merged/row linear 与 loader | 有限 E2E 覆盖 |
| Causal conv + SiLU | prefill `[6144,T]` / decode `[B,6144]` → 同形；history `[P,6144,3]`，width=4 | BF16 history；FP32 累加 | 新增 public Ascend dispatch + vendor eager 实现 | FP64 grouped conv oracle；227 NPU cases，含分块/逐步与槽隔离 |
| GDN gate / recurrence | Q/K/V `[1,T,16,128]`；decode `[B,1,16,128]`；state `[P,16,128,128]` | BF16 激活；FP32 state | 复用 prefill gate；新增 GDN dispatch / eager recurrence | FP64 K-major oracle；82 NPU cases，含非零状态、checkpoint、交叉槽 |
| Gated RMSNorm | `[T,2048]`，16 组×128，weight `[128]` → 同形 | FP32 统计，BF16 输出 | 新增 vendor helper；复用 runtime facade | FP64 scalar oracle；144 NPU cases |
| Gemma RMSNorm / residual | hidden/residual `[T,1024]` → 同形；weight `[1024]` | FP32 统计和 `1+w`；BF16 输出 | 修改 runtime 为 vendor-neutral facade | FP64 oracle、buffer/舍入契约；59 NPU cases |
| MHA 投影 / QK norm | `[T,1024]` → `[T,5120]`；Q/gate 各 `[T,8,256]`，K/V 各 `[T,2,256]` | BF16，norm 保留既有舍入语义 | 复用 QKV loader / prologue / Ascend QK norm | E2E；不冒称独立逐投影验证 |
| Partial RoPE / KV store | Q/K 仅旋转前64/256维；K/V cache 各 `[pages,page_size,2,256]` | BF16 | 复用 Ascend rotary / composite prologue | 文本 E2E；非多模态验证 |
| Causal MHA | Q `[T,8,256]`，K/V `[T,2,256]` 或 pages → `[T,8,256]` | BF16；测试亦含 FP16 | D256 prefill/extend 改为逐请求 native BSH；复用 decode | FP64 GQA oracle；151 NPU cases，含多页、mask、request/future isolation |
| Sigmoid output gate | `[T,2048]` 与 `[T,8,256]` → `[T,2048]` | BF16 | 复用 Triton-Ascend sigmoid multiply | 有限 E2E 覆盖 |
| MLP / SwiGLU | `[T,1024]` → `[T,7168]` → `[T,3584]` → `[T,1024]` | BF16 | 复用 GEMM 与 Ascend `npu_swiglu` | 有限 E2E 覆盖 |
| LM head / greedy | `[R,1024]` × tied `[248320,1024]` → `[R,248320]` → token | BF16；reference log-softmax FP32 | 复用 logits/sampling 路径 | 3输入×16步 exact IDs 与 selected-logprob 对照 |
| Cache / scheduler | conv history、recurrent state、MHA pages；offset `[B+1]` 与槽索引 `[B]` | NPU cache/state；整数 metadata | 复用 hybrid backend / LCM；禁止 Ascend 构造 CUDA stream | A→B→A、batch2 与算子槽隔离；metadata host 同步已说明 |

独立 oracle 的小型/边界形状与真实模型 E2E 互补；没有把所有复用模块声称为独立验证。完整模型的数值与性能结论均保持上述有限范围。

