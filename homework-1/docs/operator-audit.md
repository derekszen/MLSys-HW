# HW1 算子与计算路径审计

## 1. 范围、证据与记号

本审计针对 **Qwen3.5-0.8B 的纯文本、单 chip、TP=1、BF16、greedy eager 推理**，不代表整个 Qwen3.5 系列、视觉编码器、MTP、量化或图执行已经适配。模型结构以 [实际 config](../results/model/config.json) 为准；执行范围以 [model-validation.json](../results/model-validation.json)、[benchmark-run-2.txt](../results/benchmark-run-2.txt) 和下文逐项日志为准。独立原型文档记录早期阶段，不能把原型测试当作运行时集成测试。

- `T`：当前 packed token 数；`B`：请求数；`P`：cache/state pool 槽数；`R`：送往 logits head 的 token 行数；形状均按 TP=1 写出，权重按 `[out,in]`。
- 实际模型：hidden=1024，24 层，**18 层 GDN + 6 层 MHA**。MHA 层号（从 0 开始）为 3/7/11/15/19/23。所有层都有 dense MLP，intermediate=3584。词表 248320，embedding/head 权重绑定。
- “复用”指当前改动之前已有模型/Ascend 实现；“新增/修改”指本次工作树中补齐的路径，不归因给未确认身份的学生。源码基线为子模块 HEAD `52bdb9763bb2da3d61336398072671885cc0a33e` **加 [保存的适配 patch](../patches/ascend-qwen3.5.patch)**，不是声称该 commit 本身已有这些能力。
- “端到端覆盖”只表示已记录输入经过该模块且最终生成通过所述判据，不等于该模块已经单独数值对照。未记录独立测试的复用算子明确保留此边界。

源码路径下文相对于 `homework-1/tokenspeed/`；`K/` 表示 `tokenspeed-kernel/python/tokenspeed_kernel/`，`N/` 表示 `tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/`，`RT/` 表示 `python/tokenspeed/runtime/`。

## 2. 功能 / 形状 / 精度 / 来源 / Ascend / 测试矩阵

| 算子或模块、功能 | 本模型主要输入 → 输出 / 状态 | 数据类型与数值约定 | 当前实现来源、复用或新增 | Ascend 状态及已记录测试 |
| --- | --- | --- | --- | --- |
| Token embedding：查表 | 整数 IDs `[T]`；权重 `[248320,1024]` → `[T,1024]` | 权重/输出 BF16；索引为整数 | **复用** `RT/layers/vocab_parallel_embedding.py`，unquantized `F.embedding`；TP=1 不需要跨 rank 聚合 | NPU 执行的模型路径；无此次独立 embedding 数值日志，只有 E2E 覆盖 |
| GDN 输入/输出投影（18 层） | `[T,1024]` → merged `[T,8224]`，拆成 Q/K 各 2048、V/Z 各 2048、b/a 各 16；最后 `[T,2048]` → `[T,1024]` | BF16 dense，`A_log,dt_bias[16]` 单独为 FP32；未使用量化 | **复用** `RT/models/qwen3_5.py::Qwen3_5GatedDeltaNet` 的 merged/row linear、packed weight loaders；`RT/layers/dense/unquant.py` 经 GEMM facade 调度 | 已实际加载并 E2E 运行；不额外声称每个 GEMM 形状独立对照或已调优 |
| Causal depthwise conv + SiLU（18 层） | QKV **6144 通道、width=4**；prefill `[6144,T]` → 同形状；decode `[B,6144]` 原地输出；history `[P,6144,3]`，也支持 width-major `[P,3,6144]` | BF16 输入/history；乘加、bias、SiLU FP32 后转回输入 dtype；history 保存原始输入，最旧在前 | **新增** `K/ops/conv/ascend.py` 注册 prefill/update → `N/ops/conv.py` eager tap 运算；不是调用独立 `hw1/state_ops.py` | 支持普通 eager prefill/单步更新；227 项 NPU-selected 测试通过（C）；含独立 FP64 grouped conv oracle、分块/逐步、交叉槽、负索引、stride、错误拒绝 |
| GDN gate + QK L2 norm + recurrence（18 层） | prefill Q/K/V `[1,T,16,128]`，g/β `[1,T,16]` → `[1,T,16,128]`；decode `[B,1,16,128]`；state **`[P,16,128,128] = [P,HV,V,K]`** | 激活 BF16；g、递归乘积/累加及 state FP32。`g=-exp(A_log)*softplus(a+dt_bias)`，`β=sigmoid(b)`；QK 用 `sqrt(sum(x²)+1e-6)`，读出乘 `1/sqrt(128)` | **新增** `K/ops/attention/gdn/ascend.py` → `N/ops/gdn.py`；prefill 的 `RT/layers/attention/linear/gdn.py::fused_gdn_gating` 为**复用** Triton-Ascend 路径，β 由 device Torch 求得；decode 在新 vendor helper 求 gate | 82 项 NPU-selected 测试通过（G）；FP64 K-major oracle、checkpoint/resume、非零状态、chunk/step、分组头、布局/stride、padding、独立读写槽与 public registry；仅 eager、非融合 baseline |
| GDN 输出 gated RMSNorm（18 层） | recurrence/Z 展平为 `[T,2048]`；**16 组/头，每组 128**，共享 weight `[128]` → `[T,2048]` | BF16 输入/输出；统计、weight 乘法、SiLU(z) FP32；`norm_before_gate=True`，不是 MHA 的纯 sigmoid gate | **新增** `ascend_gated_rmsnorm`（`K/ops/layernorm/ascend.py`）→ `N/ops/gated_rmsnorm.py`；复用 `RMSNormGated` facade | 144 项 NPU-selected 测试通过（N）；FP64 scalar oracle、group/shared weight、两种 gate/顺序、混合 dtype、strides、empty、public dispatch；FP8 output 不支持 |
| 层间与最终 Gemma RMSNorm / residual add | 各层 attention 前后 `[T,1024]`，最终 `[T,1024]`；可带同形状 residual；weight `[1024]` | BF16 residual add 先按输入 dtype 舍入；统计与 **`1+weight.float()` 保持 FP32**，最终转回 BF16；eps=1e-6 | **修改** `RT/layers/layernorm.py::GemmaRMSNorm` 走 vendor-neutral `gemma_rmsnorm/gemma_add_rmsnorm`，Ascend 复用新增 no-gate grouped RMSNorm helper | 59 项真实 NPU runtime-layer 测试通过（R）；FP64 oracle、buffer identity、残差舍入、BF16 weight offset、重载、empty、拒绝 FP8/FP4 copy；测试宽度非穷举，1024 的实际模型路径另由 E2E 覆盖 |
| MHA Q/gate/K/V 与输出投影（6 层） | `[T,1024]` → `[T,5120]`，Q/gate 各 `[T,8,256]`、K/V 各 `[T,2,256]`；attention `[T,2048]` → `[T,1024]` | BF16，GQA 8 Q / 2 KV，D=256；gate 与 Q 按每头交错拆分，不能把整个 Q/gate 当两个任意连续块 | **复用** `Qwen3_5AttentionDecoderLayer`、`QKVParallelLinear/RowParallelLinear` 与 checkpoint loaders | 已加载并 E2E 运行；新增 MHA suite 验证 core，不等同于单独验证全部投影 |
| MHA per-head Q/K RMSNorm | Q `[T,8,256]`、K `[T,2,256]`（接口可 flatten）→ 同形状；各 weight `[256]` | BF16；复用 Ascend 的 QK 特例：`1+w` 先转回 weight dtype，FP32 统计的 scale 也会转回激活 dtype；**不同于上行新增 Gemma facade 的一次最终舍入** | **复用** attention prologue → `K/ops/layernorm/ascend.py::qk_rmsnorm` → `N/ops/layernorm.py::qk_rmsnorm`；不声称本次改变其精度语义 | E2E 覆盖；未记录此次独立 QK norm 测试。与 reference 的差异须由模型判据约束，不能宣布逐位相同 |
| Partial RoPE / KV cache 写入 | Q/K D=256 中仅 **前 64 维旋转**（factor=0.25），余 192 维透传；position 索引；cache K/V `[pages,page_size,2,256]` | Q/K/cache BF16；Ascend cos/sin 表先按激活 dtype 舍入；theta=10000000，配置含 mrope section `[11,11,10]` | **复用** rotary embedding 与 composite attention prologue、`K/ops/embedding/ascend.py` → `N/ops/rotary_embedding.py` 的 `npu_mrope`；prologue 负责 cache store | 仅文本位置的已记录 E2E 覆盖；不是视觉/M-RoPE 输入测试，也没有新增独立 partial-RoPE 日志 |
| Causal MHA prefill / paged extend / decode | packed Q `[T,8,256]`，K/V `[T,2,256]` 或上述 pages；输出 `[T,8,256]`；decode 每请求一个 query | BF16（新测试亦覆盖 FP16）；softmax scale `1/sqrt(256)`；extend causal mask bottom-right 对齐已有前缀 | **修改** `N/ops/mha.py`：D256 prefill/extend 按请求转 **BSH** 后调用 `torch_npu.npu_fused_infer_attention_score`，规避本栈 TND D256 拒绝；**复用**原 BSH decode、registry/cache ABI | 最终 151 项真实 NPU 测试通过（A，新 suite 含 D256 与既有路径回归）；FP64 GQA oracle、变长/空序列、shuffled pages、causal/noncausal、padding/future isolation、自定义 scale、capture 拒绝及 D64/decode 回归；D256 prefill/extend eager only |
| MHA sigmoid 输出 gate（6 层） | attention `[T,2048]`，gate strided view `[T,8,256]` → 原地 `[T,2048]` | BF16 输入/输出；`x *= sigmoid(gate)`，不是 SiLU | **复用** `K/ops/activation/triton.py::sigmoid_mul`，Triton-Ascend 编译执行，PDL 禁用 | E2E 已经过；没有此次独立 gate 数值日志，不把 gated-RMSNorm 的 sigmoid 分支测试冒充本实现测试 |
| Dense MLP / SwiGLU（24 层） | `[T,1024]` → gate/up `[T,7168]` → SiLU(gate)×up `[T,3584]` → down `[T,1024]` | BF16、无量化；具体 GEMM 累加由选中 provider 决定，不凭配置假定全部 FP32 | **复用** `Qwen3_5MoeMLP` 的 dense 分支、GEMM facade；`K/ops/activation/ascend.py` → `N/ops/activation.py::silu_and_mul` → `torch_npu.npu_swiglu` | 纯文本 E2E 覆盖；类名含 MoE 不代表本模型走 MoE，也不证明 INT8/FP8/FP4 路径 |
| LM head / logits / greedy | final `[R,1024]` × tied weight `[248320,1024]` → logits `[R,248320]` → 每请求整数 token；校验可返回所选 token logprob | 权重/hidden BF16；reference 将 score 转 FP32 再 log_softmax；不要据此假定两端全部 logits dtype/值一致 | **复用** conditional-generation wrapper 的 lm_head/logits processor、vocab embedding/head 与 greedy sampling；权重绑定由已有 loader 完成 | 3 输入×16 步 exact IDs + 所选 token logprob 判据（E）；没有保存两端全词表 logits 差值，不能宣称 full-logit equivalence |
| 状态/请求调度与元数据 | LCM 管理 conv windows、FP32 recurrent state 与 MHA pages；prefill offsets `[B+1]`，decode source/destination `[B]` | state/cache 在 NPU；offset/indices 为整型；小型元数据 `.tolist()` 会同步到 host | **复用** hybrid/state backend、scheduler、LCM pool；新 helper 不创建私有全局请求状态；**修改**模型 alt stream，仅 NVIDIA 创建 `torch.cuda.Stream()` | 新算子路由/交叉写测试 + E2E A-after-B/batch2；没有覆盖抢占恢复、多 rank、并发修改同 pool 或全部 scheduler 行为 |

详细 ABI 文档：[GDN](../tokenspeed/tokenspeed-kernel-npu/docs/gdn.md)、[conv](../tokenspeed/tokenspeed-kernel-npu/docs/causal-conv.md)、[gated RMSNorm](../tokenspeed/tokenspeed-kernel-npu/docs/gated-rmsnorm.md)、[Gemma RMSNorm](../tokenspeed/tokenspeed-kernel-npu/docs/gemma-rmsnorm.md)、[D256 MHA](../tokenspeed/docs/mha-d256.md)、[attention prologue numerics](../tokenspeed/docs/design/attention-prologue.md)。

## 3. 关键模块追踪：模型 → backend → registry → vendor GDN

### 3.1 Prefill

1. `RT/models/qwen3_5.py::Qwen3_5GatedDeltaNet.forward` 将 `[T,1024]` 投影为 Q/K/V/Z/b/a。当前配置每种头都是 16×128，QKV 为 6144 维、Z 为 2048 维。拆出的 view 可能有 row stride，不能默认 contiguous。
2. 模型经 `ctx.attn_backend.forward(...)` 交给 hybrid backend 中的 `GdnAttnBackend`（`RT/layers/attention/backends/state/gdn.py`），实际共享执行在 `state/mamba.py::forward_extend`。LCM 提供本层 conv/recurrent pool、初始槽/目标槽及 sequence boundaries。
3. backend 将 QKV 转为 `[6144,T]`，调用 `causal_conv1d_fn` → `K/ops/conv/ascend.py` → `N/ops/conv.py::launch_causal_conv1d_prefill`，再转回 `[T,6144]` 并 split/reshape 为 `[1,T,16,128]`。history 为最近 3 个**原始投影输入**，不是激活结果。
4. `state/mamba.py::_prefill_scan` 先求 `beta=b.sigmoid()`，调用复用的 `fused_gdn_gating` 生成 FP32 g；随后通过 `tokenspeed_kernel.ops.attention.gdn.gdn_chunk_prefill` 按 Ascend capability、dtype、traits 选择 `ascend_gdn_chunk_prefill`。公共适配器返回真实 `GdnChunkPrefillResult`，算术落在 `N/ops/gdn.py::launch_gdn_chunk_prefill`，**不是**用更低精度 state 调用 `npu_recurrent_gated_delta_rule`。
5. 每 token 的 FP32 K-last 状态按下式更新；prefill 初始 state 不被修改，最终 state 由 backend `store_state_rows(...)` 写入 LCM live 目标槽。调用方请求的 interval checkpoint 存实际中间 state，interval 为 64 的正倍数；本次 E2E 配置没有启用 in-forward checkpoint。

```text
q_hat,k_hat = x / sqrt(sum(x*x) + 1e-6)
D           = exp(g_t) * S_prev                  # S: [HV,V,K]
r           = beta_t * (v_t - sum_K(D*k_hat))
S_next      = D + r[...,V,None]*k_hat[...,None,K]
y_t         = sum_K(S_next*q_hat) / sqrt(K)
```

Prefill 会将规范化后的 Q/K 舍入到输入 dtype 再做 FP32 recurrence，以遵守 shared chunk contract；decode 保留 FP32 normalized Q/K。因此低精度 chunk/step 应按误差界比较，不应要求逐位相同。此 ABI 也不同于早期独立原型的 clamp-L2 规范，不把原型的 67 CPU/14 NPU 测试替代集成测试。

### 3.2 Decode、状态隔离与输出

`state/mamba.py` 先用 `causal_conv1d_update` 更新 conv window，再将 Q/K/V view 成 `[B,1,16,128]`。`_decode_scan` 经公共 `gdn_decode_step` 选择 `ascend_gdn_decode_step` → `N/ops/gdn.py::launch_gdn_decode_step`。FP32 state 使 `gdn_decode_solution` 不强制 BF16-state Triton 分支；厂商选择仍通过 registry，而不是模型直接导入 vendor 实现。

共享 pool 的 layout 始终是 **`[P,HV,V,K]`，K 在最后**；在此 Ascend 路径中不必为数学上的 K-major 形式转置 state。decode core 输出 `[B,1,HV,V]`，backend 仅将输出转为 `[1,B,HV,V]`，不是转置 recurrent state。模型随后展平 `[T,2048]`、做 16 组 gated RMSNorm、投影回 1024。

新 decode helper 在任何写入前快照所有 source state，允许共享只读初始页与交叉 source/destination；只写 active destination，重复 active destination 报错。负 source 为 padding：GDN 输出零且不写 pool；负 destination 只抑制写回。conv update 的负 source 保留该输入行而非强制置零，这是不同 API，不能混用。请求 ownership 仍归 scheduler/LCM；无 backend 私有全局状态。

### 3.3 设备同步与明确限制

激活、权重、卷积和 recurrent state 的算术留在 NPU，无自动 CPU fallback。**只有边界、路由索引、checkpoint starts、program maps 等整数/标志元数据用 `.tolist()` 读到 host；这仍会触发 NPU→host 同步**，而且 conv/GDN 的每层调用均有此成本。Python 循环逐 token/请求提交多个 eager Torch op，并产生 FP32 临时张量。D256 MHA extend 同样读 host 元数据并逐请求发射 BSH native op。

这些事实支持“同步与细粒度提交可能值得优化”的假设，不支持未经 profile 的耗时占比或“最大瓶颈已证实”。本实现不声称 GDN/conv graph capture、MTP/tree/replay、autograd 或优化性能；GDN `output_h=True` 不支持，checkpoint 为 caller-owned standard ABI。模型实验显式启用 eager、关闭 prefill graph/PDL/autotune，使用 greedy、无 speculative config。独立 suite 中 decode graph 回归通过，不代表全模型可 capture。

## 4. 已记录的分层测试与误差界

以下均为服务器上**实际 NPU-selected 执行**的成功摘要，而不是本次文档任务重新运行的测试；deselected 不是失败，也不是额外 NPU 通过项。warning 数保留在原日志，不能用 pytest 耗时推断模型性能。

| 标记 | 测试源（相对子模块） | 成功原始日志 | 记录结果 |
| --- | --- | --- | --- |
| G | `tokenspeed-kernel-npu/test/test_gdn.py` | [integrated-gdn-tests-2.txt](../results/integrated-gdn-tests-2.txt) | **82 passed**, 81 deselected |
| C | `tokenspeed-kernel-npu/test/test_causal_conv.py` | [integrated-conv-tests.txt](../results/integrated-conv-tests.txt) | **227 passed**, 220 deselected |
| N | `tokenspeed-kernel-npu/test/test_gated_rmsnorm.py` | [runtime-import.txt](../results/runtime-import.txt) | **144 passed**, 93 deselected；同日志另有 `ENGINE_IMPORT_OK`，不将 import 当模型验证 |
| R | `tokenspeed-kernel-npu/test/test_gemma_layer.py` | [integrated-gemma-tests.txt](../results/integrated-gemma-tests.txt) | **59 passed** |
| A | `tokenspeed-kernel-npu/test/test_mha_d256.py`（含 D64/decode 回归；另有原有 `test_mha.py` 4 项测试，不计入此 151） | [integrated-mha-tests-3.txt](../results/integrated-mha-tests-3.txt) | **151 passed** |
| E | `scripts/model_reference.py` + `scripts/engine_validate.py`（相对 homework-1） | [model-reference.json](../results/model-reference.json)、[model-validation.json](../results/model-validation.json)、[model-validation-run-3.txt](../results/model-validation-run-3.txt) | 5/32/256 输入各 16 步 IDs exact；所选 logprob 最大绝对差分别 0.1255554 / 0.0408112 / 0.0022228；A-after-B 与 batch2 两请求均 pass |

测试以相同、已舍入的输入建立独立 CPU FP64 oracle；oracle/比对拷贝到 CPU 不等于被测 runtime 在 CPU 上算。数值界来自各测试文件 `_assert_close`，按 `abs(actual-ref) <= atol + rtol*abs(ref)` 判定：

| 模块 | BF16 `(atol,rtol)` | FP16 `(atol,rtol)` | FP32 `(atol,rtol)` |
| --- | --- | --- | --- |
| GDN 输出 | `(4e-3,8e-3)` | `(5e-4,1e-3)` | `(2e-6,2e-5)` |
| Conv 输出 | `(5e-4,8e-3)` | `(5e-5,1e-3)` | `(2e-7,2e-5)` |
| Gated / Gemma RMSNorm 输出 | `(2e-3,8e-3)` | `(3e-4,1e-3)` | `(2e-6,3e-6)` |
| 新 D256 MHA 输出 | `(3e-3,8e-3)` | `(4e-4,1e-3)` | 此 suite 未声明 FP32 输出界 |

GDN state 是 FP32，采用上表 FP32 界；conv 原始 history、非目标槽保持、输入非修改等断言可用 exact。宽松的低精度输出界用于最终舍入与运算顺序差异，并非允许丢算或低精度 recurrent state；这些是有限用例的界，不是任意长上下文保证。

**保留失败历史：** [首轮 GDN 日志](../results/integrated-gdn-tests.txt) 为 81 passed/1 failed，public registry 编译暴露 Triton-Ascend 3.2.0 与本 CANN header 的兼容问题；私有环境安装 3.2.1 后后续日志为 82 passed。[模型启动](../results/engine-smoke-5.txt) 暴露 TND D256 拒绝；[MHA 首轮](../results/integrated-mha-tests.txt) 为单 query 的 compressed-mask 拒绝，[第二轮](../results/integrated-mha-tests-2.txt) 为多 query GQA 的 64-token page 拒绝，不能当成功证据；最终 BSH 适配及扩展 suite 的证据是 `integrated-mha-tests-3.txt`。

E2E 的 256-token 案例虽调用 flush，结果仍记录 **47 cached tokens**；该结果证明的是该缓存路径下生成判据通过，不是 256-token 全量 uncached prefill。后续关闭 prefix cache/kvstore 的 benchmark 对 32/256 token 都先做 16-token exact-reference warmup，测量请求也检查 `cached_tokens==0`；见 [benchmark 脚本](../scripts/benchmark.py) 和 [第二轮原始日志](../results/benchmark-run-2.txt)。这补充了 uncached 生成 ID 证据，但未补充 uncached selected-logprob 差或全词表 logits 对照。

## 5. 审计结论

已记录证据支持：本配置纯文本单 chip 的关键新增算子在真实 NPU suite 中通过，并且 TokenSpeed 完成多输入长度连续 greedy 推理、有限请求隔离检查与四组合客户端性能 baseline。直接复用的 embedding、dense GEMM、SwiGLU、QK norm、partial RoPE、sigmoid gate、cache/scheduler 仅按表中范围归因，不夸大为逐算子全面验证。

技术 baseline 已建立；[benchmark 逐样本](../results/benchmark.json)、进程级 `npu-smi` HBM 和 [CPU-only profile](../results/profile-summary.json) 已归档，详见 [报告](../report.md)。联合执行 1,057 项测试通过（663 NPU/394 CPU），另有原有 MHA suite 4 项通过；报告开头列有姓名、学号与已记录的协调贡献。现有共享服务器日志不构成 NPU 预约或隔离性能证据，不宣称完整 logits 等价、全模型族覆盖或优化后的性能。
