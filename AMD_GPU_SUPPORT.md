# AMD GPU Support for Transformers4Rec

This document describes the changes made in this fork to support AMD Instinct
GPUs (ROCm) alongside the existing NVIDIA CUDA path.

**Upstream:** [NVIDIA-Merlin/Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec)

---

## 1. Motivation

The upstream Transformers4Rec library depends on NVIDIA's Merlin ecosystem for
GPU-accelerated data loading (`merlin-dataloader`, `cudf`, `nvtabular`). These
packages are CUDA-only and cannot be installed on AMD ROCm systems. Several
places in the codebase also hardcode `"cuda"` device strings, causing failures
on ROCm where PyTorch uses HIP behind the `torch.cuda` API.

This fork removes those barriers so that the full training and inference
pipeline works on AMD Instinct GPUs (MI300X, MI355X, etc.) without any code
changes from the user's perspective.

---

## 2. What Changed

The changes span two groups: device-agnostic fixes to the existing codebase
(on `main`), and the new dataloader plus utilities (on `feat/extend-dataloader`).
All changes are backward-compatible — the original NVIDIA Merlin code path is
preserved and activates automatically when Merlin is installed.

### Device-agnostic fixes (`main`)

| File | Change | Purpose |
|------|--------|---------|
| `transformers4rec/utils/dependencies.py` | Added `is_rocm_available()`, `is_gpu_dataloader_available()`, `is_hipdf_available()` | Single detection point for ROCm vs CUDA branching |
| `transformers4rec/config/trainer.py` | `data_loader_engine` auto-detects: ROCm → `"rocm"`, else → `"merlin"` | Works out of the box on both platforms |
| `transformers4rec/torch/block/base.py` | Replaced `out.to("cuda")` with device inference from module parameters | Device-agnostic tensor placement |
| `transformers4rec/torch/block/transformer.py` | Same device-agnostic fix | Consistent with above |
| `transformers4rec/torch/trainer.py` | Guarded CUDA RNG state restore | Safe checkpoint restore across GPU types |
| `transformers4rec/torch/utils/examples_utils.py` | Guarded `torch.cuda.empty_cache()` | No crash on CPU-only environments |
| `transformers4rec/torch/model/base.py` | Guarded NVIDIA-specific model utilities | Clean import on ROCm |
| `transformers4rec/utils/data_utils.py` | Added `session_aggregator_pandas()` | Preprocessing without RAPIDS/NVTabular |
| `transformers4rec/utils/serialization.py` | Platform-agnostic schema serialization | Schema I/O without TensorFlow/protobuf NVIDIA deps |

### New dataloader (`feat/extend-dataloader`)

| File | Change | Purpose |
|------|--------|---------|
| `transformers4rec/torch/utils/data_utils.py` | Guarded Merlin imports; added `ROCmDataLoader` and its backend iterators | Platform-agnostic GPU-native dataloader (see Section 3) |

---

## 3. Dataloader Architecture

### 3.1 Overview

The new `ROCmDataLoader` is registered in the dataloader registry as `"rocm"`
and `"rocm_dataloader"`. It reads Parquet files via PyArrow and serves batches
as `(inputs_dict, targets_dict)` tuples — the same format that `MerlinDataLoader`
produces — so the `Trainer` and model code require no changes.

The dataloader selects from four backends automatically, falling through to the
next if the preferred one is unavailable:

| Tier | Backend | When used |
|------|---------|-----------|
| 1 | `_HipDFBatchIterator` | `cudf` (hipDF) + `cupy` are installed (future ROCm releases) |
| 2a | `_StreamingGPUBatchIterator` | Data exceeds ~40% of GPU memory |
| 2b | `_GPUBatchIterator` | Data fits in GPU memory (most common) |
| 3 | `_ROCmParquetDataset` + PyTorch `DataLoader` | No GPU available; CPU fallback |

**Tier 2b (`_GPUBatchIterator`)** is the workhorse for typical use. It:

1. Reads all Parquet files into a pandas DataFrame at init (via PyArrow)
2. Converts every column to a GPU-resident PyTorch tensor once
3. Serves batches by pure tensor indexing — zero host-to-device copies during training

**Tier 2a (`_StreamingGPUBatchIterator`)** activates for large datasets. It:

1. Scans Parquet row-group metadata (no data loaded)
2. Groups row-groups into GPU-memory-sized chunks (~4 GB default)
3. Iterates chunk by chunk, with a background thread prefetching the next chunk's Parquet read

### 3.2 Data Flow

```
Parquet files
    │
    ▼
PyArrow read ──► pandas DataFrame ──► numpy arrays ──► GPU PyTorch tensors
                                                            │
                                                   batch = tensors[idx]
```

Scalar columns become 1-D tensors. List/sequence columns are padded or trimmed
to `max_sequence_length` and stored as 2-D tensors `(num_rows, max_seq_len)`.
Padding uses zeros and happens once at init, not per-batch.

### 3.3 Features

- **DDP:** Supports `global_size` / `global_rank` at every tier.
  Tier 2a assigns contiguous blocks of row-groups per rank; Tier 2b/3 shard
  rows by stride.
- **Shuffle:** Per-epoch deterministic GPU-side shuffle (seeded for DDP
  reproducibility).
- **Schema-driven:** `from_schema()` extracts categorical, continuous, target,
  and list features from schema tags, identically to `MerlinDataLoader`.
- **Auto-streaming:** Automatically switches to streaming mode when data
  exceeds GPU memory.

---

## 4. Scope and Compatibility

### What is fully supported

The standard Transformers4Rec workflow — Parquet data, schema-driven column
selection, padded sequence batches, HuggingFace `Trainer`-managed training —
works identically on both AMD and NVIDIA hardware. This covers the primary
use case of the library.

| Capability | MerlinDataLoader | ROCmDataLoader |
|------------|------------------|----------------|
| Parquet file paths (str, list, glob, directory) | Yes | Yes |
| Schema-driven column discovery (`Tags`) | Yes | Yes |
| Sequence padding/trimming to `max_sequence_length` | Yes | Yes |
| Batch format: `(inputs_dict, targets_dict)` | Yes | Yes |
| DDP / multi-GPU training | Yes | Yes |
| Shuffle with deterministic seeding | Yes | Yes |
| Large-dataset streaming | Via `buffer_size` / `parts_per_chunk` | Via auto-sized chunks + prefetch |
| `drop_last` | Yes | Yes |

### Known limitations

These are Merlin-specific features that the `ROCmDataLoader` does not
replicate. They are narrow in scope and do not affect the standard training
workflow.

| Feature | Notes |
|---------|-------|
| **`merlin.io.Dataset` as input** | `ROCmDataLoader` accepts file paths only, not in-memory `merlin.io.Dataset` objects. In practice, all upstream examples construct the Dataset from a file path anyway. |
| **`transforms` (Merlin DAG operators)** | `MerlinDataLoader` can apply per-batch transforms such as `EmbeddingOperator` for pretrained embeddings. `ROCmDataLoader` does not support this parameter. Pretrained embeddings can still be used via the model's `PretrainedEmbeddingsFeatures` module (a schema-level configuration). |
| **CSV / non-Parquet engines** | Merlin's `validate_dataset` supports `engine="csv"`. `ROCmDataLoader` reads Parquet only. No example or tutorial in the upstream repo uses CSV for training. |
| **`output_schema` property** | Not exposed. Not referenced by the Trainer or training loop. |

---

## 5. Auto-Detection

The `data_loader_engine` training argument is auto-detected at startup:

```
ROCm detected (torch.version.hip is not None)  →  "rocm"
Otherwise                                       →  "merlin"
```

Users can override this explicitly:

```shell
# Force the new dataloader on NVIDIA hardware
python train.py --data_loader_engine rocm ...

# Force Merlin on a system where both are available
python train.py --data_loader_engine merlin ...
```

---

## 6. Preprocessing Without NVTabular

The upstream library uses NVTabular for session-level feature engineering
(grouping interactions into sessions, computing sequence features). This fork
adds `session_aggregator_pandas()` in `transformers4rec/utils/data_utils.py`
as a pure-pandas alternative that produces identical output without requiring
RAPIDS.

---

## 7. Known Upstream Issue: XLNet Positional Encoding

During testing we discovered a performance issue in HuggingFace Transformers'
XLNet implementation (confirmed in v4.30.2). The
`XLNetModel.relative_positional_encoding` method calls `torch.arange()`
without a `device=` argument, which forces positional embedding computation
onto the CPU every forward pass and requires a host-to-device copy of the
result.

**Impact:** Up to 314 ms of overhead per training step (hardware-dependent),
which can be the dominant cost for shorter sequences.

**Fix:** Add `device=` to the three `torch.arange` calls inside
`relative_positional_encoding` so that all downstream sin/cos/einsum
operations run on GPU. The fix is a one-line change per call site and applies
to any GPU platform (AMD or NVIDIA).

This is an issue in the HuggingFace `transformers` library, not in
Transformers4Rec itself. The appropriate resolution is an upstream PR to
HuggingFace. The repository includes a `tools/xlnet_pos_encoding_fix.py`
utility that applies the fix as a runtime monkey-patch until it is merged
upstream.

---

## 8. Included Utilities

The repository includes two utility scripts at the top level. These are not
part of the `transformers4rec` package and are not required for normal usage.

| File | Purpose |
|------|---------|
| `tools/xlnet_pos_encoding_fix.py` | Runtime monkey-patch for the XLNet positional encoding issue described in Section 7. Import before training to apply the fix. |
| `tools/generate_synthetic_data.py` | Generates synthetic Parquet datasets matching the Transformers4Rec schema format. Useful for testing and benchmarking on new hardware. |
