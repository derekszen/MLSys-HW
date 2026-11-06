# HW1 实验报告

## Qwen3.5 0.8B 在 Ascend 单 chip 上的复现与正确性验证

**Steven Walxim　2501112130**　　实验日期：2026 年 11 月 6 日

### 小组成员主要贡献

| 成员 | 本报告对应的贡献 |
|---|---|
| Steven Walxim（2501112130） | 在本人课程账户建立私有环境、核验模型文件，重新运行 Transformers 参考、TokenSpeed 算子测试、完整生成与性能矩阵；复现 CUDA stream 初始化错误，补充真实模型初始化回归；整理新结果及其适用范围。另完成已有证据的离线核验工具与路径可移植性修复。 |
| Zhuoxi Zeng（2501213387） | 根据小组既有贡献记录，负责需求与实验目标确认、资源及账户访问协调，并提出正确性回归、差异修复和材料整理要求。小组仓库提供本次复用的冻结适配、实验脚本及历史证据。 |

本文是基于本人账户新实验记录撰写的复现报告。原有 Ascend 适配与实验框架来自小组共享仓库；不将复用实现表述为本人从零开发。模型初始化的设备判断也已存在于小组冻结 patch，本次新增工作是重现其必要性、补充回归并在独立 checkout 中验证。

### 1 实验目标与主要结论

目标是在一张 Ascend 910C 的单个 chip 上运行纯文本 Qwen3.5-0.8B，完成权重加载、prefill、decode 与连续推理，并建立可复核的正确性及性能基线。

本次联合测试 **1,058 项通过、无失败或跳过**；模型在 5、32、256 token 输入下均连续生成 16 tokens，greedy IDs 与新运行的未修改 Transformers 参考完全一致，所选 token 的 logprob 误差通过既定判据。三项请求状态隔离检查通过；32/256 输入与 8/16 输出的四组合、每组三次、共 12 次无缓存测量全部通过输出检查。[E2–E5]

该结论限定于本文模型、精度、输入与 eager 配置，不代表全词表 logits 等价、任意长会话、多模态、量化或多 chip 支持。

### 2 环境与运行配置

| 项目 | 本次运行值 |
|---|---|
| 设备与隔离 | 物理 chip 1，映射为逻辑 `npu:0`；TP=1。每次启动前确认该 chip 无已列出的进程；未终止其他用户作业。 |
| 软件栈 | Python 3.11.16；PyTorch 2.9.0+cpu 配合 torch-npu 2.9.0；CANN 9.0.0；Transformers 5.12.0；triton-ascend 3.2.1；pytest 8.4.2。 |
| 模型与精度 | Qwen3.5-0.8B；权重和激活 BF16，递归 state FP32；greedy，seed=42；纯文本，禁用 speculative decoding。 |
| 调度与缓存容量 | max_model_len=512，max_num_seqs=2，max_total_tokens=16384；prefill chunk=128；gpu_memory_utilization=0.12。 |
| 执行方式 | eager；关闭 prefill graph、PDL 与 autotune。性能测试关闭 prefix cache、kvstore 和输出 logprob。 |

私有 venv 继承课程系统包，未更改共享环境。保存的软件版本可运行本实验，但依赖清单不是无冲突 lock：`pip check` 仍报告厂商依赖及版本元数据问题；实际导入、算子测试和模型执行通过。[E1]

<div class="page-break"></div>

## 3 源码边界与主要算子支持

上游固定为 `52bdb9763bb2da3d61336398072671885cc0a33e`。在独立 checkout 上先应用本次初始化回归 patch，再应用小组冻结 HW1 patch，并排除其中已单独处理的 `qwen3_5.py`。没有混入 HW2 的 TP 改动。模型来自 ModelScope 的 `Qwen/Qwen3.5-0.8B`；本次下载后核验了 manifest 中各文件的大小与 SHA256。[E1、E6]

模型为 hidden=1024、24 层，其中 18 层 GDN、6 层 full attention；MLP intermediate=3584，词表 248320，embedding 与 head 绑定。下表为 TP1 的逻辑形状。T 为 token 数，B 为请求数，P 为状态槽数；权重按 `[out,in]` 表示。“小组适配”指冻结 patch，“复用”指原有实现。本次重新运行集成测试；详细实现来源沿用小组算子审计，并与模型配置核对。[E2、E7]

| 功能 | 主要形状与精度 | 实现来源及 Ascend 支持 | 验证方式 |
|---|---|---|---|
| Embedding 与 GDN 投影 | IDs `[T]` → `[T,1024]`；GDN merged 输出 `[T,8224]`；BF16 | 复用 embedding、linear 和权重 loader；实际 NPU 模型运行通过 | 有限端到端覆盖 |
| Causal conv 与 SiLU | 6144 通道，width=4；prefill `[6144,T]`；history `[P,6144,3]`；BF16 输入与 history、FP32 运算 | 小组新增公共 Ascend dispatch 和 eager helper | FP64 conv oracle；分块/逐步、stride、交叉槽与错误输入 |
| GDN recurrence | Q/K/V `[1,T,16,128]`；state `[P,16,128,128]`；激活 BF16、state FP32 | 小组新增 GDN registry 实现与 eager recurrence；仅本文 eager 范围 | FP64 K-major oracle；非零状态、checkpoint、变长、padding、槽隔离 |
| Gated RMSNorm | `[T,2048]`；16 组×128；共享 weight `[128]`；统计 FP32 | 小组新增 Ascend helper，复用 runtime facade | FP64 scalar oracle；gate 顺序、mixed dtype、stride |
| Gemma norm 与 residual | hidden/residual `[T,1024]`；统计及 `1+w` 为 FP32，输出 BF16 | 小组修改 runtime facade；保留 residual 舍入契约 | FP64 oracle、buffer identity、权重重载与舍入 |
| MHA 投影与 QK norm | Q/gate 各 `[T,8,256]`；K/V 各 `[T,2,256]`；BF16 | 复用 loader、attention prologue、QK norm | 端到端覆盖；不冒称每个投影独立验证 |
| Partial RoPE 与 KV 写入 | Q/K 的前 64/256 维旋转；KV 各 `[pages,page_size,2,256]` | 复用 Ascend rotary 与 cache prologue | 本文文本位置的端到端覆盖 |
| Causal MHA | 8 Q heads、2 KV heads、D=256；BF16 | 小组将 D256 prefill/extend 适配为逐请求 native BSH 调用；复用 decode | FP64 GQA oracle；mask、多页、future/request isolation |
| 输出 gate 与 MLP | sigmoid gate `[T,2048]`；MLP `1024→7168→3584→1024`；BF16 | 复用 sigmoid multiply、GEMM 与 Ascend SwiGLU | 有限端到端覆盖 |
| LM head 与调度状态 | logits `[R,248320]`；LCM 管理 conv、GDN 与 KV 状态 | 复用 head、greedy sampler、scheduler/LCM；仅 NVIDIA 创建 CUDA stream | 生成与 logprob 对照、A 重复和 batch2 隔离、初始化回归 |

上述复用模块实际参与了 NPU 推理，但端到端通过不等于每个模块都完成独立数值证明。小型及边界形状的算子测试与真实模型形状的端到端测试相互补充。

<div class="page-break"></div>

## 4 关键路径与适配障碍

### 4.1 GDN 从模型到厂商算子的调用路径

`Qwen3_5GatedDeltaNet.forward` 将 hidden 投影为 Q/K/V/Z/b/a，经 hybrid attention backend 的 `GdnAttnBackend` 进入共享 `state/mamba.py`。LCM 提供 conv history、递归 state、序列边界和源/目标槽。

Prefill 将 QKV 转为 `[6144,T]`，经公共 `causal_conv1d_fn` 选择 Ascend conv helper，再将结果恢复为 Q/K/V `[1,T,16,128]`。随后公共 `gdn_chunk_prefill` 经 registry 调度到 `tokenspeed_kernel_npu/ops/gdn.py`。Decode 先更新 conv window，再经 `gdn_decode_step` 完成逐步递归。模型最后将输出展平为 `[T,2048]`，做 16 组 gated RMSNorm，再投影回 1024。[E7]

对每个 head，state 的布局为 `[V,K]`，K 在最后；整个池为 `[P,HV,V,K]`，保持 FP32。先归一化 Q/K，再依次完成衰减、误差更新与读出：

```text
q_hat, k_hat = x / sqrt(sum(x*x) + 1e-6)
g = -exp(A_log) * softplus(a + dt_bias)
beta = sigmoid(b)
D = exp(g) * S_prev
r = beta * (v - D @ k_hat)
S_next = D + outer(r, k_hat)
y = (S_next @ q_hat) / sqrt(K)
```

Prefill 按共享接口约定将归一化 Q/K 舍入到输入 dtype；decode 保留 FP32 normalized Q/K。因此低精度 chunk/step 的比较使用明确误差界，不要求逐位一致。Decode core 的输出由 `[B,1,HV,V]` 转为 backend 使用的 `[1,B,HV,V]`；这不是对 recurrent state 做转置。

### 4.2 状态生命周期与设备同步

Conv history 保存最近三个**原始投影输入**，不能保存 SiLU 后的值。GDN decode 在写入前快照全部 source state，再更新 active destination，避免交叉槽污染。请求状态仍由 scheduler/LCM 拥有，不引入私有全局可变状态；prefill 的最终 state 由 backend 写回目标槽。

权重、激活与递归算术在 NPU 上执行，但整数边界和路由索引等元数据通过 `.tolist()` 读回 host，会产生同步。Python 循环还会逐 token/请求提交多个 eager Torch 算子；D256 MHA extend 也有逐请求 BSH 调用。因此“主要数学运算在 NPU”不等于“无主机调度成本”。

### 4.3 本次复现的初始化失败与修复验证

干净上游在构造模型时尝试创建 `torch.cuda.Stream()`，在 Ascend 环境报 `Tried to instantiate dummy base class Stream`，尚未进入完整权重加载。将创建条件限定为 NVIDIA 后，本次新增的真实 `Qwen3_5ForCausalLM` 最小配置回归由失败转为通过。[E6]

该回归在 meta device 上检查模型可构造、embedding 与 norm 等形状，不是完整推理替代品。设备 guard 已存在于小组冻结 patch；本次工作验证了它的必要性，而不是声称新发明此适配。原小组另记录过 native TND D256 拒绝与 Triton-Ascend 编译兼容问题；本次复用其 BSH 路径及版本配置，不把历史失败写成本次重新发生的故障。

<div class="page-break"></div>

## 5 分层正确性验证

### 5.1 算子测试与判据

本次运行五个小组算子 suite，加上新增模型初始化回归，共 **1,058 passed，6 warnings，10.81 秒**。总数包含 CPU 参考/校验用例和 NPU 用例，不能描述成 1,058 个纯 NPU kernel 测试；没有将 skip 算作通过。警告涉及共享安装目录所有权和厂商 tensor format，未通过更改共享环境来消除警告。[E2]

数值比较采用 `abs(actual-reference) <= atol + rtol*abs(reference)`，参考读取相同的已量化输入。BF16 GDN 输出使用 `(4e-3,8e-3)`，conv 使用 `(5e-4,8e-3)`，gated/Gemma norm 使用 `(2e-3,8e-3)`，D256 MHA 使用 `(3e-3,8e-3)`；FP32 recurrent state 使用 `(2e-6,2e-5)`。History 保留、非目标槽与输入不修改等契约使用 exact 断言。低精度输出界用于容纳舍入和运算顺序差异，不允许省略状态更新或降低 FP32 state 精度。[E7]

### 5.2 新运行的模型参考与连续生成

参考为未修改的 Transformers 5.12.0 `Qwen3_5ForConditionalGeneration`，在同一 NPU 软件栈以 BF16、eager attention、greedy 运行；使用相同模型文件和保存的 input IDs。三个输入长度为 5、32、256，均强制生成 16 tokens、忽略 EOS。短提示为 `The capital of France is`，长输入按 tokenizer token 截取，不以字符数代替 token 数。[E1、E3]

本项目采用“16 步 greedy IDs exact，且各步所选 token logprob 绝对误差 ≤0.15”的工程检查。**0.15 是项目选择，不是教师给定的评分阈值，也不是所有输入的误差保证。** 两种执行路径在归约、norm、RoPE 与舍入上有差异；本实验只判断这三个案例的有限结果。

| 输入 tokens | 输出 tokens | Greedy IDs | 最大所选 logprob 误差 | cached tokens |
|---:|---:|---|---:|---:|
| 5 | 16 | 全部一致 | 0.1255553961 | 0 |
| 32 | 16 | 全部一致 | 0.0408112407 | 0 |
| 256 | 16 | 全部一致 | 0.0022227764 | 47 |

以上为本人账户 11 月 6 日的 TokenSpeed 实测。[E4] 虽然验证脚本调用 flush，256-token 案例仍报告 47 cached tokens，不能将该条称为完全无缓存 prefill。随后性能脚本显式关闭 prefix cache 与 kvstore；两个输入的 16-token warmup 及全部 12 次正式请求均检查 reference IDs，正式请求 `cached_tokens=0`。这补充了无缓存生成证据，但没有补充无缓存 selected-logprob 或全词表 logits 对照。[E5]

### 5.3 多步状态与请求隔离

验证同一 engine 先处理短请求 A、再处理较长请求、之后重复 A，输出 16 IDs 仍与参考一致；flush 后同时提交 5/32-token 两请求，batch2 的两条输出也各自一致，三项检查全部通过。结合算子级非零状态、交叉槽和 checkpoint 测试，支持本文范围内的状态隔离结论，不扩展为全部抢占恢复、长会话或压力并发行为均已验证。

<div class="page-break"></div>

## 6 性能与内存基线

### 6.1 测量方法与新实测结果

engine 启动后分别以 32/256 输入、16 输出 warmup 并核对输出。正式测量为单并发，输入 32/256 × 输出 8/16，每组三次。用客户端 `perf_counter` 记录流式 token 到达，首个 chunk 必须恰含一个 token；关闭 prefix cache、kvstore 及输出 logprob。[E5]

`TTFT = 首 token 到达时间 - 请求开始时间`；`TPOT = (末 token 到达时间 - 首 token 到达时间)/(输出 tokens - 1)`。这些量包含调度、IPC 和交付，不是 NPU event 测得的纯 kernel 时间。表中单位均为毫秒，括号为三次测量的 min–max，不是置信区间。

| 输入→输出 | TTFT 中位数（范围） | TPOT 中位数（范围） | 总时长中位数 |
|---|---:|---:|---:|
| 32→8 | 266.383（264.696–266.902） | 73.711（73.580–74.856） | 782.885 |
| 32→16 | 260.957（203.118–268.696） | 70.334（70.046–73.551） | 1315.975 |
| 256→8 | 997.488（976.347–1074.446） | 64.787（64.393–70.477） | 1451.007 |
| 256→16 | 1000.399（954.623–1013.665） | 66.912（66.249–67.882） | 1994.145 |

全部 12 次输出长度、reference IDs 与零缓存检查通过。长输入 TTFT 约 1 秒，短输入约 0.26 秒；TPOT 中位数约 65–74 ms。结合 128-token chunk 的配置，这说明长输入首 token 阶段工作更多，但单凭客户端时间不能分解计算与访存占比，也不能因长输入 TPOT 较低就认定存在优化收益。各指标分别取中位数，不要求中位数之间严格可加。

### 6.2 内存指标与证据来源

| 来源与口径 | 数值 | 解释 |
|---|---|---|
| **本次 TokenSpeed benchmark** 启动 Torch allocated / reserved | **1.92 / 2.06 GiB** | 日志表头写 GB，runtime 使用二进制口径。分别为 allocator 已分配与保留池，不能相加，也不是运行峰值。 |
| **本次 TokenSpeed benchmark** cache arena | **213,909,504 bytes = 204 MiB** | 实际计划与分配记录；可能已包含在 allocator 统计中，不另行累加。 |
| **本次 Transformers 参考** peak allocated | **1,838,871,040 bytes** | 仅参考实现的 Torch allocator 峰值；不替代 TokenSpeed 峰值或进程总 HBM。 |
| **小组原实验** `npu-smi` 进程 HBM | worker **2299 MB**（启动）、**2525 MB**（工作负载后）；frontend **112 MB** | 历史同模型基线快照，按 worker PID 2588245、frontend PID 2588138 识别；不是本人本次进程的测量。 |

新启动记录见 E8，参考峰值见 E3；历史进程 HBM 见 E9。共享设备总 used 包含其他用户进程，不能归因给本模型。没有本次瞬时峰值 HBM 或按 prefill/decode 分阶段的进程峰值。新 benchmark 在服务器生成的 `hbm-*.txt` 尚未收录到本报告证据包，因此本文不据其推测数值。

<div class="page-break"></div>

## 7 计算瓶颈分析与后续优化

本节使用**小组原实验的 CPU-only profile**，不是本人 11 月 6 日新采集的 trace。原实验对 warmup 后的 256 输入→8 输出进行记录，并核验输出；两个 128-token EXTEND 的 host forward span 合计 **1772.943 ms**，8 个 DECODE span 合计 **774.001 ms**。[E10]

| 历史主机事件 | 调用次数 | exclusive CPU 时间 ms |
|---|---:|---:|
| `empty_tensor` | 81,193 | 232.464 |
| `aten::mul` | 31,754 | 187.704 |
| `aten::as_strided` | 89,312 | 129.268 |
| `aten::select` | 49,474 | 126.602 |
| `aten::copy_` | 7,939 | 96.994 |
| `npu::_npu_dtype_cast` | 14,254 | 73.254 |

exclusive 时间按线程扣除直接子调用，避免层层累计。Trace 带 profiler 开销，不应把上述 span 当作无 profiler 的性能基线。8 个输出对应记录中的 8 次 decode forward；不能用 774.001/7 替代第 6 节的 TPOT。矩阵乘和 attention 的主机调用时间较短也不能证明设备计算较快，因为 NPU 异步执行。

源码中的 18 层 GDN、prefill 的逐 token Python 循环、FP32 临时张量和 metadata `.tolist()`，与原 profile 中大量细粒度算子/分配事件相符。由此，**优先优化方向是 GDN eager 路径的提交、同步与临时张量开销**。这是有源码和历史测量支持的方向判断，尚不是最大 NPU 硬件瓶颈的定量证明；原 trace 没有完整模块标签，无法将全部 CPU 事件精确分配到 GDN、conv 和 norm。

下一步可设计融合 recurrence 或 workspace 复用实验，固定模型、输入、dtype 和缓存设置，先重新通过本文的 FP32 state、槽隔离及输出检查，再比较相同客户端矩阵，并补充 NPU 时间线。当前没有执行该优化，也没有声称获得 speedup。D256 的 BSH 改动解决支持性问题，不能仅凭运行成功称为性能优化。

## 8 结论与实验边界

本次在本人账户重建单 chip HW1 环境，并重新获得算子、完整生成、有限请求隔离和 2×2 性能矩阵的通过结果。直接观察到的首个障碍是上游无条件创建 CUDA 辅助 stream；在复用小组 guard 与 Ascend 适配后，初始化及模型计算路径通过验证。主要复用内容为模型主体、公共调度框架、小组厂商 dispatch/算子与实验脚本；本人新增贡献集中于环境重建、可追溯复现、初始化回归和证据核验。

结果不覆盖完整 logits 等价、任意长上下文、全部 scheduler 行为、图执行、量化、多模态或多 chip。服务器是共享资源，选择空闲 chip 不等于设备预约或全机隔离。HBM 与 profile 的新旧来源已明确标注；本报告不以历史结果冒充新运行，也不将 pytest 时间当作模型速度。HW2 的 TP2 源码与结果不进入此 HW1 结论。

<div class="page-break"></div>

## 附录 复现与证据索引

### A 源码与运行步骤

在干净的固定上游 checkout 上应用 `rebuild/patches/01-device-initialization.patch`，再应用 `patches/ascend-qwen3.5.patch`，第二步排除 `python/tokenspeed/runtime/models/qwen3_5.py`。不要在已经适配的源码上重复应用，也不要用 HW2 工作树覆盖冻结 HW1 patch。文件相对于 `homework-1/`。

设置课程 CANN 环境，使用私有 venv，令 `PYTHONPATH` 包含该 checkout 的 `python`、`tokenspeed-kernel/python` 和 `tokenspeed-kernel-npu/python`。先检查可用 chip，再显式设置 `ASCEND_RT_VISIBLE_DEVICES`。本次使用 1 不代表设备以后仍空闲。模型按 manifest 下载，权重不放入提交附件。

依次执行 `model_reference.py`、五个算子测试及新增初始化回归、`engine_validate.py`、`benchmark.py --repeats 3`。三个模型脚本都显式传入 `--physical-device`、`--model` 和新 `--output`；验证与 benchmark 还需 `--reference`。原始精确命令保存在各 run 的 `command.sh`。本次执行设 600 秒 timeout、15 秒终止宽限，并记录退出码；结束后确认 chip 无本次残留进程、port 31887 已释放。

### B 证据索引

以下路径均相对于本报告的 Markdown 文件。PDF 正文已包含主要表格；需要逐样本或代码核查时，可在仓库中打开对应材料。

| 标记 | 材料与用途 |
|---|---|
| E1 | [参考运行 provenance](results/2026-11-06/reference-provenance.json)：模型文件 hash 与运行版本；[rebuild README](README.md)：环境与源码重建说明。 |
| E2 | [本次测试汇总](results/2026-11-06/operator-tests/summary.json)、[退出状态](results/2026-11-06/operator-tests/status.json)：1,058 项联合测试；原始日志 SHA256 保存在汇总中。 |
| E3 | [新 Transformers reference](results/2026-11-06/reference.json)：input/output IDs、selected logprobs 与 allocator 峰值。 |
| E4 | [新 TokenSpeed validation](results/2026-11-06/engine-validation/validation.json)、[运行状态](results/2026-11-06/engine-validation/status.json)：三个案例、cache 记录和三项状态隔离。 |
| E5 | [新 benchmark](results/2026-11-06/benchmark/benchmark.json)、[运行状态](results/2026-11-06/benchmark/status.json)：12 个样本、流式到达、零缓存与统计。 |
| E6 | [初始化回归 patch](patches/01-device-initialization.patch)、[早期 checkpoint](results/2026-11-06/checkpoint.json)：失败与修复；checkpoint 的“未开始”状态是早期记录，以 E2–E5 的后续结果为准。 |
| E7 | [小组算子审计](../docs/operator-audit.md)、[冻结 HW1 patch](../patches/ascend-qwen3.5.patch)、[模型 config](../results/model/config.json)：来源、ABI、形状、误差界及实现。 |
| E8 | [新日志的内存摘录](results/2026-11-06/report-evidence.json)：启动内存行及原始日志 hash；只摘录实际记录，不推算峰值。 |
| E9 | 小组原实验 [startup HBM](../results/hbm-startup.txt)、[32→8](../results/hbm-32-8.txt)、[32→16](../results/hbm-32-16.txt)、[256→8](../results/hbm-256-8.txt)、[256→16](../results/hbm-256-16.txt)。 |
| E10 | 小组原实验 [CPU profile 汇总](../results/profile-summary.json)：历史瓶颈证据，非本次新测；文件内记录原始 trace 的 SHA256。 |

### C 原始要求对应关系

贡献声明与环境见首页；计算路径、支持矩阵、状态/同步和适配障碍见第 3–4 节；分层数值与请求隔离见第 5 节；两输入×两输出、TTFT/TPOT/HBM 见第 6 节；瓶颈证据、优化方向和局限见第 7–8 节。原题见 [HW1 assignment](../assignment.md)。
