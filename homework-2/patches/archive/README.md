# Historical HW2 patches

**Do not use this archive as the current HW2 patch.** Apply frozen HW1 first, then the current [hccl-torch29.patch](../hccl-torch29.patch), following the [current manifest](../manifest.json) and [application instructions](../../README.md#incremental-patch-application).

| Artifact | Meaning |
| --- | --- |
| [hccl-only.patch](hccl-only.patch) | Original **three-file HCCL-only** incremental patch; historical bytes retained |
| [hccl-only-manifest.json](hccl-only-manifest.json) | Original manifest retained unchanged; SHA256 `8e3c8b22c527795e0f962b427f945d4c14b75247adbe15798b0a18fce2147364` identifies the archived patch bytes |
| [hccl-npu-profile.patch](hccl-npu-profile.patch) | Exact **six-file HCCL + NPU profiling** incremental patch, 42,916 bytes; SHA256 `ea88a840112da0c173ed3fa491efa39853fab33708f583115cd5cb5681c77fcd` |
| [hccl-npu-profile-manifest.json](hccl-npu-profile-manifest.json) | Matching six-file manifest retained byte-for-byte; manifest SHA256 `ae4547e5f70714b9994c4370bfb7c034f0f460f58dcab3bf3a0935d6615027b2` |
| [hccl-profile-cache-precision.patch](hccl-profile-cache-precision.patch) | Exact **11-file HCCL + NPU profiling + cache-precision** incremental patch, 62,069 bytes; SHA256 `85eb82b1510fbbcd2b9347cc6b116e95aa2bc43b41065957b0b8a02cd23463bd` |
| [hccl-profile-cache-precision-manifest.json](hccl-profile-cache-precision-manifest.json) | Matching 11-file manifest retained byte-for-byte; manifest SHA256 `5bcc31a742f16f8eb02d8ab54268ec9ded9024407b9ad46e326d38761d4e82ae` |

All three historical manifests still say `"patch": "hccl-torch29.patch"` and retain their original application paths. That was the filename when each version was packaged; it **does not identify today's parent-directory patch**. Match each archived manifest to the patch in the same table row pair above. Do not apply historical patches together or apply a historical patch before the current HW2 patch: each is incremental directly after frozen HW1.

The current patch adds an explicit optional FP32 final head and environment publication to HCCL + NPU profiling + Qwen3.5 cache-precision controls: **15 files, 89,738 bytes**, SHA256 `76071724b56e1f7922dd3fd9a4c2c5b359f3422252ac10c74daa5e5d0f2350f6`. [Verification](../patch-verification.json) reconstructs pinned base + frozen HW1 + current HW2 with **2,972 files byte/mode-exact**, including `100755` for `python/tokenspeed/runtime/layers/logits_processor.py`, `python/tokenspeed/runtime/utils/env.py`, and `python/tokenspeed/runtime/utils/server_args.py`, and `100644` for the other 12 incremental files. Frozen HW1 SHA256 remains `e68f2f26e49d30180280adbab0bb0ebaf433246e27457788e60277621613b8c6`; no HW1 experimental evidence or patch is replaced. This is reconstruction verification, not a new hardware correctness claim.
