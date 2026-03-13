#!/usr/bin/env python3
"""
Transformers4Rec Production Benchmark (v2)
==========================================

Extends bench_t4rec.py to support:
  - External Parquet datasets (from generate_synthetic_data.py)
  - Streaming GPU dataloader for large-scale data
  - Production-scale model configs (d=1024/1536, L=12/24)
  - BF16 mixed precision
  - Forward/Backward/Optimizer breakdown (--breakdown)
  - Ranking-metric evaluation via T4Rec built-in metrics (--eval-recall)
  - Inference latency/throughput benchmark (--mode inference)
  - Backward-compatible: falls back to built-in test data if no --data-dir

Benchmarks:
  A. Dataloader throughput  (rows/sec, data iteration only)
  B. Training step latency  (fwd + bwd + optimizer, per step)
  B+. Fwd/Bwd/Opt breakdown  (--breakdown)
  C. Full epoch throughput  (end-to-end including data loading)
  D. Inference sweep        (--mode inference or --mode both)

Usage:
    # With generated data (recommended for production benchmarking):
    python bench_t4rec_v2.py --data-dir /nvme_local/t4rec_data/medium_1m \
        --model-size xlarge --max-seq-len 200 --bf16

    # 8-GPU DDP with breakdown + eval:
    torchrun --nproc_per_node=8 bench_t4rec_v2.py \
        --data-dir /nvme_local/t4rec_data/medium_1m \
        --model-size xlarge --max-seq-len 200 --bf16 --ddp \
        --breakdown --eval-recall --eval-interval 500

    # Inference benchmark (single-GPU):
    python bench_t4rec_v2.py --data-dir /nvme_local/t4rec_data/medium_1m \
        --model-size xlarge --max-seq-len 200 --bf16 \
        --mode inference --infer-batch-sizes 1,8,32,128

    # Full benchmark (train + inference):
    torchrun --nproc_per_node=8 bench_t4rec_v2.py \
        --data-dir /nvme_local/t4rec_data/prod_50m \
        --model-size xlarge --max-seq-len 200 --bf16 --ddp \
        --streaming --breakdown --eval-recall --mode both \
        --output-json results_mi300x_xlarge_bf16_8gpu_full.json

    # Compare results:
    python bench_t4rec_v2.py --compare results_a.json results_b.json

Model size presets:
    small    d=64    L=2   H=4    (~100K params)
    medium   d=256   L=4   H=8    (~5M params)
    large    d=512   L=6   H=8    (~20M params)
    xlarge   d=1024  L=12  H=16   (~100M+ params, production)
    xxlarge  d=1536  L=24  H=24   (~500M+ params, stress test)
"""

import argparse
import gc
import glob as glob_mod
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np
import torch

# ── PyTorch AMP compatibility ────────────────────────────────────────────
if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    _GradScaler = lambda **kw: torch.amp.GradScaler("cuda", **kw)
else:
    _GradScaler = torch.cuda.amp.GradScaler

# ── merlin-core compatibility shim ───────────────────────────────────────
try:
    from merlin.schema import Tags

    if not hasattr(Tags, "EMBEDDING"):
        _placeholder = Tags.CATEGORICAL
        Tags._member_map_["EMBEDDING"] = _placeholder
        Tags._value2member_map_["embedding"] = _placeholder
        type.__setattr__(Tags, "EMBEDDING", _placeholder)
except Exception:
    Tags = None

# ── Model size presets ───────────────────────────────────────────────────
MODEL_PRESETS = {
    "small": {"d_model": 64, "n_layer": 2, "n_head": 4},
    "medium": {"d_model": 256, "n_layer": 4, "n_head": 8},
    "large": {"d_model": 512, "n_layer": 6, "n_head": 8},
    "xlarge": {"d_model": 1024, "n_layer": 12, "n_head": 16},
    "xxlarge": {"d_model": 1536, "n_layer": 24, "n_head": 24},
}

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="T4Rec production benchmark (v2)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
# Data
parser.add_argument(
    "--data-dir", type=str, default=None,
    help="Path to generated data dir (with schema.json + train/valid splits). "
         "If omitted, uses built-in test data with --num-rows synthetic rows.",
)
parser.add_argument("--split", default="train", choices=["train", "valid"])
parser.add_argument("--num-rows", type=int, default=50_000,
                    help="Rows to generate if --data-dir is not set")
# Model
parser.add_argument(
    "--model-size", choices=MODEL_PRESETS.keys(), default=None,
    help="Model preset (overrides --d-model/--n-layer/--n-head)",
)
parser.add_argument("--d-model", type=int, default=64)
parser.add_argument("--n-layer", type=int, default=2)
parser.add_argument("--n-head", type=int, default=4)
# Training
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=20)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--bench-steps", type=int, default=100)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                    help="Cap steps per epoch in Benchmark C (default: full epoch). "
                         "Useful for fixed step-budget runs on large datasets.")
# DDP / Precision
parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
parser.add_argument("--fp16", action="store_true", help="Mixed precision FP16")
parser.add_argument("--bf16", action="store_true", help="Mixed precision BF16")
parser.add_argument("--compile", action="store_true", help="torch.compile (PyTorch 2.x)")
# Dataloader
parser.add_argument("--streaming", action="store_true",
                    help="Force streaming dataloader (auto-detected by default)")
parser.add_argument("--rows-per-chunk", type=int, default=None,
                    help="Rows per streaming chunk (auto-sized if omitted)")
# Softmax
parser.add_argument("--sampled-softmax", action="store_true",
                    help="Use sampled softmax for NextItemPrediction (required for "
                         "large item vocabs, auto-enabled when vocab > 500K)")
parser.add_argument("--softmax-samples", type=int, default=10000,
                    help="Number of negative samples for sampled softmax (default: 10000)")
# Breakdown
parser.add_argument("--breakdown", action="store_true",
                    help="Measure fwd/bwd/optimizer breakdown per step")
parser.add_argument("--breakdown-steps", type=int, default=20,
                    help="Number of steps to measure for breakdown (default: 20)")
# Evaluation (uses T4Rec built-in ranking metrics: Recall@K, NDCG@K, etc.)
parser.add_argument("--eval-recall", action="store_true",
                    help="Run T4Rec ranking metrics on validation set at step intervals")
parser.add_argument("--eval-interval", type=int, default=500,
                    help="Steps between evaluation passes (default: 500)")
parser.add_argument("--eval-batches", type=int, default=50,
                    help="Max batches per evaluation pass (default: 50)")
# Inference benchmark
parser.add_argument("--mode", choices=["train", "inference", "both"], default="train",
                    help="Benchmark mode: train, inference, or both (default: train)")
parser.add_argument("--infer-batch-sizes", type=str, default="1,8,32,128",
                    help="Comma-separated batch sizes for inference benchmark")
parser.add_argument("--infer-seq-lens", type=str, default="50,100,200",
                    help="Comma-separated sequence lengths for inference benchmark")
parser.add_argument("--infer-warmup", type=int, default=20,
                    help="Warmup iterations for inference (default: 20)")
parser.add_argument("--infer-iters", type=int, default=100,
                    help="Measured iterations for inference (default: 100)")
# Output
parser.add_argument("--output-json", type=str, default=None)
parser.add_argument(
    "--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
    help="Compare two JSON result files and exit",
)
ARGS = parser.parse_args()

if ARGS.model_size:
    p = MODEL_PRESETS[ARGS.model_size]
    ARGS.d_model = p["d_model"]
    ARGS.n_layer = p["n_layer"]
    ARGS.n_head = p["n_head"]

if ARGS.bf16 and ARGS.fp16:
    print("ERROR: --bf16 and --fp16 are mutually exclusive", file=sys.stderr)
    sys.exit(1)

# ── DDP setup ────────────────────────────────────────────────────────────
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DDP = ARGS.ddp and WORLD_SIZE > 1
IS_MAIN = LOCAL_RANK == 0

if IS_DDP:
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)

DEVICE = torch.device(f"cuda:{LOCAL_RANK}")


def log(msg):
    if IS_MAIN:
        print(msg, flush=True)


# ═════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════


def detect_platform():
    info = {
        "hostname": platform.node(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(LOCAL_RANK)
            if torch.cuda.is_available() else "N/A"
        ),
        "gpu_count": torch.cuda.device_count(),
        "world_size": WORLD_SIZE,
    }
    hip = getattr(torch.version, "hip", None)
    if hip:
        info["platform"] = "AMD ROCm"
        info["hip_version"] = hip
    else:
        info["platform"] = "NVIDIA CUDA"
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    from transformers4rec.utils.dependencies import is_rocm_available
    info["dataloader_engine"] = "rocm" if is_rocm_available() else "merlin"
    return info


def load_schema(data_dir):
    """Load a merlin_standard_lib Schema from a data directory."""
    schema_path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"No schema.json in {data_dir}")
    from merlin_standard_lib import Schema
    return Schema().from_json(schema_path)


def load_metadata(data_dir):
    meta_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


def get_data_paths(data_dir, split):
    """Resolve Parquet file paths from a data directory."""
    split_dir = os.path.join(data_dir, split)
    if os.path.isdir(split_dir):
        paths = sorted(glob_mod.glob(os.path.join(split_dir, "*.parquet")))
        if paths:
            return split_dir, paths
    paths = sorted(glob_mod.glob(os.path.join(data_dir, "*.parquet")))
    return data_dir, paths


def _get_cardinality(schema, col_name):
    try:
        feats = schema.select_by_name(col_name).feature
        if feats and hasattr(feats[0], "int_domain"):
            return int(feats[0].int_domain.max)
    except Exception:
        pass
    return None


def generate_fallback_data(path, n_rows, max_seq_len, schema, seed=42):
    """Generate small synthetic Parquet (fallback when --data-dir is not set)."""
    import pandas as pd

    rng = np.random.RandomState(seed)
    list_names = set(schema.select_by_tag(Tags.LIST).column_names)
    cat_names = set(schema.select_by_tag(Tags.CATEGORICAL).column_names)
    cont_names = set(schema.select_by_tag(Tags.CONTINUOUS).column_names)
    all_cols = list(dict.fromkeys(list(cat_names) + list(cont_names)))
    seq_lens = rng.randint(1, max_seq_len + 1, size=n_rows)

    data = {}
    for col in all_cols:
        is_list = col in list_names
        is_cat = col in cat_names
        if is_list and is_cat:
            card = _get_cardinality(schema, col) or 500
            total = int(seq_lens.sum())
            vals = rng.randint(1, card + 1, size=total)
            offsets = np.concatenate([[0], np.cumsum(seq_lens)])
            data[col] = [vals[offsets[i]:offsets[i + 1]].tolist()
                         for i in range(n_rows)]
        elif is_list:
            total = int(seq_lens.sum())
            vals = rng.uniform(-1.0, 1.0, size=total)
            offsets = np.concatenate([[0], np.cumsum(seq_lens)])
            data[col] = [vals[offsets[i]:offsets[i + 1]].tolist()
                         for i in range(n_rows)]
        elif is_cat:
            card = _get_cardinality(schema, col) or 100
            data[col] = rng.randint(1, card + 1, size=n_rows).tolist()
        else:
            data[col] = rng.uniform(0.0, 1.0, size=n_rows).tolist()

    pd.DataFrame(data).to_parquet(path, index=False)


def build_model(schema, d_model, n_head, n_layer, max_seq_len,
                sampled_softmax=False, max_n_samples=10000):
    import transformers4rec.torch as tr
    from transformers4rec.config import transformer as tconf

    input_module = tr.TabularSequenceFeatures.from_schema(
        schema, max_sequence_length=max_seq_len, d_output=d_model,
        masking="causal",
    )
    xlnet_config = tconf.XLNetConfig.build(
        d_model=d_model, n_head=n_head, n_layer=n_layer,
        total_seq_length=max_seq_len,
    )
    prediction_task = tr.NextItemPredictionTask(
        weight_tying=True,
        sampled_softmax=sampled_softmax,
        max_n_samples=max_n_samples,
    )
    return xlnet_config.to_torch_model(input_module, prediction_task)


def build_dataloader(data_path, schema, batch_size, max_seq_len, shuffle=True,
                     device_id=None, global_size=None, global_rank=None,
                     streaming=None, rows_per_chunk=None):
    from transformers4rec.torch.utils.data_utils import T4RecDataLoader
    from transformers4rec.utils.dependencies import is_rocm_available

    if is_rocm_available():
        engine = "rocm"
    else:
        for candidate in ("merlin", "merlin_dataloader", "rocm"):
            try:
                T4RecDataLoader.parse(candidate)
                engine = candidate
                break
            except (KeyError, Exception):
                continue
        else:
            engine = "rocm"

    loader_cls = T4RecDataLoader.parse(engine)
    log(f"  Dataloader engine: {engine} ({loader_cls.__name__})")
    loader = loader_cls.from_schema(
        schema=schema,
        paths_or_dataset=data_path,
        batch_size=batch_size,
        max_sequence_length=max_seq_len,
        shuffle=shuffle,
        device=device_id,
        global_size=global_size,
        global_rank=global_rank,
        streaming=streaming,
        rows_per_chunk=rows_per_chunk,
    )
    return loader


def _move(d, device):
    if d is None:
        return {}
    return {k: v.to(device, non_blocking=True) for k, v in d.items()}


def _get_target(inputs, targets, key="item_id/list"):
    if targets and key in targets:
        return targets[key]
    if key in inputs:
        return inputs[key]
    src = targets if targets else inputs
    return next(iter(src.values()))


def _train_step(model, loader_batch, optimizer, scaler, autocast_ctx,
                 breakdown=False):
    """One training step.  Returns (loss, breakdown_dict | None).

    When *breakdown* is True, inserts ``torch.cuda.synchronize()`` barriers
    to measure forward, backward, and optimizer phases separately.  In DDP
    mode the backward phase includes the overlapped all-reduce gradient sync.
    """
    b_in, b_tgt = loader_batch
    b_in = _move(b_in, DEVICE)
    b_tgt = _move(b_tgt, DEVICE)
    target = _get_target(b_in, b_tgt).to(DEVICE)

    timings = None
    optimizer.zero_grad(set_to_none=True)

    if not breakdown:
        # Fast path – no extra syncs
        if autocast_ctx is not None:
            with autocast_ctx():
                out = model(b_in, targets=target, training=True)
                loss = out["loss"]
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            out = model(b_in, targets=target, training=True)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
        return loss, None

    # --- Breakdown path: measure each phase ---------------------------------
    timings = {}
    torch.cuda.synchronize(DEVICE)

    # Forward
    t0 = time.perf_counter()
    if autocast_ctx is not None:
        with autocast_ctx():
            out = model(b_in, targets=target, training=True)
            loss = out["loss"]
    else:
        out = model(b_in, targets=target, training=True)
        loss = out["loss"]
    torch.cuda.synchronize(DEVICE)
    timings["fwd_ms"] = (time.perf_counter() - t0) * 1000

    # Backward (includes all-reduce in DDP)
    t0 = time.perf_counter()
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    torch.cuda.synchronize(DEVICE)
    timings["bwd_ms"] = (time.perf_counter() - t0) * 1000

    # Optimizer step
    t0 = time.perf_counter()
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    torch.cuda.synchronize(DEVICE)
    timings["opt_ms"] = (time.perf_counter() - t0) * 1000

    return loss, timings


def evaluate_ranking_metrics(model, loader, autocast_ctx, max_batches=50,
                             top_ks=(10, 20)):
    """Evaluate Recall@K and NDCG@K using held-out-last-item protocol.

    For each validation sequence, the last non-padded item is masked and used
    as the ground-truth target.  The model predicts at the new last position
    (training=False) producing full-vocabulary logits.  ``torch.topk`` is used
    to check whether the target appears in the top-K predictions -- no one-hot
    encoding is created, so this works with arbitrarily large vocabularies.
    """
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    n_batches = 0

    all_hits = {k: [] for k in top_ks}
    all_ndcg = {k: [] for k in top_ks}
    max_k = max(top_ks)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            b_in, b_tgt = batch
            b_in = _move(b_in, DEVICE)
            b_tgt = _move(b_tgt, DEVICE)

            item_key = "item_id/list"
            item_seq = b_in.get(item_key)
            if item_seq is None:
                continue

            non_pad = item_seq != 0
            seq_lens = non_pad.sum(dim=1)

            valid_mask = seq_lens >= 2
            if not valid_mask.any():
                continue

            rows = torch.arange(item_seq.size(0), device=item_seq.device)
            last_pos = seq_lens - 1
            targets = item_seq[rows, last_pos]

            item_seq_modified = item_seq.clone()
            item_seq_modified[rows, last_pos] = 0
            b_in[item_key] = item_seq_modified

            target_dummy = _get_target(b_in, b_tgt).to(DEVICE)
            if autocast_ctx is not None:
                with autocast_ctx():
                    out = raw_model(b_in, targets=target_dummy, training=False)
            else:
                out = raw_model(b_in, targets=target_dummy, training=False)

            if isinstance(out, dict):
                preds = out.get("predictions", out.get("logits"))
            else:
                preds = out

            b_in[item_key] = item_seq

            if preds is None or preds.ndim < 2:
                continue

            preds_valid = preds[valid_mask]
            targets_valid = targets[valid_mask]

            _, topk_ids = torch.topk(preds_valid, k=min(max_k, preds_valid.size(-1)),
                                     dim=-1)
            for k in top_ks:
                topk_k = topk_ids[:, :k]
                hits = (topk_k == targets_valid.unsqueeze(1)).any(dim=1).float()
                all_hits[k].append(hits)
                match_pos = (topk_k == targets_valid.unsqueeze(1))
                positions = torch.arange(1, k + 1, device=preds.device).float()
                dcg = (match_pos.float() / torch.log2(positions + 1)).sum(dim=1)
                idcg = 1.0 / torch.log2(torch.tensor(2.0, device=preds.device))
                all_ndcg[k].append(dcg / idcg)

            n_batches += 1

    metrics = {}
    for k in top_ks:
        if all_hits[k]:
            metrics[f"recall_at_{k}"] = torch.cat(all_hits[k]).mean().item()
            metrics[f"ndcg_at_{k}"] = torch.cat(all_ndcg[k]).mean().item()

    raw_model.train()
    return metrics, n_batches


def run_inference_benchmark(model, schema, autocast_ctx, item_vocab, args):
    """Run inference latency/throughput sweep over batch sizes and seq lens."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    batch_sizes = [int(x) for x in args.infer_batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.infer_seq_lens.split(",")]
    results = []

    log("\n" + "=" * 72)
    log("INFERENCE BENCHMARK")
    log("=" * 72)
    hdr = (f"  {'BS':>5} {'SeqLen':>6} {'P50 ms':>8} {'P95 ms':>8} "
           f"{'P99 ms':>8} {'Avg ms':>8} {'Samp/s':>10}")
    log(hdr)
    log("  " + "-" * 63)

    for seq_len in seq_lens:
        for bs in batch_sizes:
            latencies = _run_infer_sweep(
                raw_model, schema, autocast_ctx, bs, seq_len,
                item_vocab, args.infer_warmup, args.infer_iters,
            )
            if latencies is None:
                log(f"  {bs:>5} {seq_len:>6}   SKIPPED (OOM or error)")
                continue

            ms = np.array(latencies) * 1000
            p50 = np.percentile(ms, 50)
            p95 = np.percentile(ms, 95)
            p99 = np.percentile(ms, 99)
            avg = np.mean(ms)
            throughput = bs / (avg / 1000)

            log(f"  {bs:>5} {seq_len:>6} {p50:>8.2f} {p95:>8.2f} "
                f"{p99:>8.2f} {avg:>8.2f} {throughput:>10,.0f}")
            results.append({
                "batch_size": bs, "seq_len": seq_len,
                "p50_ms": round(p50, 2), "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2), "avg_ms": round(avg, 2),
                "throughput_samples_sec": round(throughput, 1),
            })
    log("  " + "-" * 63)
    raw_model.train()
    return results


def _run_infer_sweep(model, schema, autocast_ctx, bs, seq_len,
                     item_vocab, warmup, iters):
    """Run inference for one (batch_size, seq_len) config.

    Generates synthetic input tensors matching the schema and measures
    forward-pass latency.
    """
    try:
        from merlin.schema import Tags as _Tags
    except ImportError:
        _Tags = None

    try:
        if _Tags is not None and schema is not None:
            cat_names = schema.select_by_tag(_Tags.CATEGORICAL).column_names
            cont_names = schema.select_by_tag(_Tags.CONTINUOUS).column_names
            list_names = set(schema.select_by_tag(_Tags.LIST).column_names)
        else:
            cat_names, cont_names, list_names = [], [], set()

        inputs = {}
        for col in cat_names:
            if col in list_names:
                card = _get_cardinality(schema, col) or 500
                inputs[col] = torch.randint(1, card + 1, (bs, seq_len),
                                            device=DEVICE)
            else:
                card = _get_cardinality(schema, col) or 100
                inputs[col] = torch.randint(1, card + 1, (bs,),
                                            device=DEVICE)
        for col in cont_names:
            if col in list_names:
                inputs[col] = torch.randn(bs, seq_len, device=DEVICE)
            else:
                inputs[col] = torch.randn(bs, device=DEVICE)

        target = inputs.get("item_id/list",
                            torch.randint(1, max(item_vocab, 2),
                                          (bs, seq_len), device=DEVICE))

        latencies = []
        with torch.no_grad():
            for i in range(warmup + iters):
                torch.cuda.synchronize(DEVICE)
                t0 = time.perf_counter()
                if autocast_ctx is not None:
                    with autocast_ctx():
                        model(inputs, targets=target, training=False)
                else:
                    model(inputs, targets=target, training=False)
                torch.cuda.synchronize(DEVICE)
                if i >= warmup:
                    latencies.append(time.perf_counter() - t0)
        return latencies
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        torch.cuda.empty_cache()
        return None


def estimate_peak_memory_gb(n_params, d_model, n_layer, n_head, batch_size,
                            max_seq_len, bf16=True):
    """Estimate peak GPU memory (GB) for XLNet-based T4Rec training.

    Memory components:
      - Model parameters (BF16) + FP32 master + AdamW states: params * 14 bytes
      - Gradients: params * 2 bytes
      - Per-sample activation memory (summed over L layers):
          attention activations: L * H * S^2 * 8 * bpe  (scores, probs, grads)
          FFN activations: L * d * 4 * S * 4 * bpe
      - 1.2x overhead for fragmentation, DDP buffers, data chunks
    """
    bpe = 2 if bf16 else 4
    model_and_optim_gb = n_params * 16 / 1e9
    S = max_seq_len
    attn_per_sample = n_layer * n_head * (S ** 2) * 8 * bpe
    ffn_per_sample = n_layer * d_model * 4 * S * 4 * bpe
    act_per_sample_gb = (attn_per_sample + ffn_per_sample) / 1e9
    total_gb = (model_and_optim_gb + batch_size * act_per_sample_gb) * 1.2
    return total_gb, act_per_sample_gb


def auto_adjust_batch_size(args, n_params, gpu_gb, safety=0.85):
    """Reduce batch_size if estimated memory exceeds GPU capacity.

    Returns the (possibly reduced) batch_size and logs warnings.
    """
    est_gb, act_per_sample_gb = estimate_peak_memory_gb(
        n_params, args.d_model, args.n_layer, args.n_head,
        args.batch_size, args.max_seq_len, bf16=args.bf16,
    )
    usable_gb = gpu_gb * safety
    if est_gb <= usable_gb:
        return args.batch_size

    model_gb = n_params * 16 / 1e9 * 1.2
    avail_for_act = usable_gb - model_gb
    if avail_for_act <= 0:
        log(f"  ** ERROR: model alone (~{model_gb:.1f} GB) exceeds GPU "
            f"capacity ({gpu_gb:.1f} GB). Cannot fit even batch_size=1.")
        sys.exit(1)

    max_bs = int(avail_for_act / (act_per_sample_gb * 1.2))
    safe_bs = max(1, max_bs & ~1)
    log(f"\n  ** MEMORY WARNING: estimated peak = {est_gb:.1f} GB "
        f"(GPU has {gpu_gb:.1f} GB, usable = {usable_gb:.1f} GB)")
    log(f"     Reducing batch_size: {args.batch_size} -> {safe_bs}")
    log(f"     (model={model_gb:.1f} GB, activations/sample="
        f"{act_per_sample_gb*1e3:.1f} MB, overhead=1.2x)")
    return safe_bs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def gpu_mem_mb():
    return torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024


# ═════════════════════════════════════════════════════════════════════════
# COMPARE MODE
# ═════════════════════════════════════════════════════════════════════════


def compare_results(file_a, file_b):
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    ca, cb = a["config"], b["config"]
    name_a = f"{ca.get('gpu_name', '?')} x{ca.get('world_size', 1)}"
    name_b = f"{cb.get('gpu_name', '?')} x{cb.get('world_size', 1)}"

    W = 72
    print("=" * W)
    print("TRANSFORMERS4REC BENCHMARK COMPARISON (v2)")
    print("=" * W)
    print(f"\n  {'Platform A':<16}: {ca.get('platform', '?')}  ({name_a})")
    print(f"  {'Platform B':<16}: {cb.get('platform', '?')}  ({name_b})")
    print(f"  {'PyTorch A':<16}: {ca.get('pytorch', '?')}")
    print(f"  {'PyTorch B':<16}: {cb.get('pytorch', '?')}")

    mismatches = []
    for key in ("d_model", "n_layer", "n_head", "batch_size",
                "max_seq_len", "world_size"):
        if ca.get(key) != cb.get(key):
            mismatches.append(f"    {key}: {ca.get(key)} vs {cb.get(key)}")
    if mismatches:
        print(f"\n  ** CONFIG MISMATCH:")
        for m in mismatches:
            print(m)
    else:
        d = ca.get("d_model", "?")
        L = ca.get("n_layer", "?")
        H = ca.get("n_head", "?")
        bs = ca.get("batch_size", "?")
        ws = ca.get("world_size", "?")
        print(f"\n  Model config    : d={d} L={L} H={H}")
        print(f"  Batch size      : {bs} x {ws} GPUs")

    hdr = f"  {'Metric':<34} {'A':>12} {'B':>12} {'A/B':>10}"
    sep = "  " + "-" * (W - 4)
    print(f"\n{sep}\n{hdr}\n{sep}")

    def _row(label, va, vb, lower_better=None):
        sa = f"{va:>12,.1f}" if va is not None else f"{'N/A':>12}"
        sb = f"{vb:>12,.1f}" if vb is not None else f"{'N/A':>12}"
        if va is not None and vb is not None and vb != 0:
            r = va / vb
            if lower_better is True:
                mark = " <" if r < 1 else ""
            elif lower_better is False:
                mark = " <" if r > 1 else ""
            else:
                mark = ""
            sr = f"{r:>8.2f}x{mark}"
        else:
            sr = f"{'N/A':>10}"
        print(f"  {label:<34}{sa}{sb}{sr}")

    ta = a.get("training_step", {})
    tb = b.get("training_step", {})
    ea = a.get("full_epoch", {})
    eb = b.get("full_epoch", {})
    da = a.get("dataloader", {})
    db = b.get("dataloader", {})

    _row("Step latency (ms)", ta.get("avg_step_ms"), tb.get("avg_step_ms"), lower_better=True)
    _row("Step P50 (ms)", ta.get("p50_ms"), tb.get("p50_ms"), lower_better=True)
    _row("Step P95 (ms)", ta.get("p95_ms"), tb.get("p95_ms"), lower_better=True)
    _row("Step throughput (samp/s)", ta.get("throughput_samples_sec"), tb.get("throughput_samples_sec"), lower_better=False)
    _row("Epoch throughput (samp/s)", ea.get("throughput_samples_sec"), eb.get("throughput_samples_sec"), lower_better=False)
    _row("Avg epoch time (s)", ea.get("avg_epoch_sec"), eb.get("avg_epoch_sec"), lower_better=True)
    _row("DL throughput (rows/s)", da.get("throughput_rows_sec"), db.get("throughput_rows_sec"), lower_better=False)
    _row("Peak GPU memory (MB)", ta.get("peak_gpu_mem_mb"), tb.get("peak_gpu_mem_mb"))
    _row("Final loss", ta.get("loss_last"), tb.get("loss_last"))

    print(sep)
    print(f"  < = better side for that metric\n")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════


def main():
    if ARGS.compare:
        compare_results(*ARGS.compare)
        return

    pinfo = detect_platform()
    precision_tag = "BF16" if ARGS.bf16 else ("FP16" if ARGS.fp16 else "FP32")

    log("=" * 72)
    log("Transformers4Rec Production Benchmark (v2)")
    log("=" * 72)
    log(f"  Platform        : {pinfo['platform']}")
    log(f"  PyTorch         : {pinfo['pytorch']}")
    log(f"  GPU             : {pinfo['gpu_name']}")
    log(f"  GPU count       : {pinfo['gpu_count']}")
    log(f"  World size      : {WORLD_SIZE}")
    log(f"  DDP             : {IS_DDP}")
    log(f"  Precision       : {precision_tag}")
    log(f"  torch.compile   : {ARGS.compile}")
    log(f"  Dataloader      : {pinfo['dataloader_engine']}")
    log(f"  Streaming       : {ARGS.streaming}")
    preset_tag = ARGS.model_size or "custom"
    log(f"  Model           : {preset_tag} (d={ARGS.d_model} L={ARGS.n_layer} H={ARGS.n_head})")
    log(f"  batch_size      : {ARGS.batch_size}")
    log(f"  max_seq_len     : {ARGS.max_seq_len}")
    log(f"  warmup_steps    : {ARGS.warmup_steps}")
    log(f"  bench_steps     : {ARGS.bench_steps}")
    log(f"  epochs          : {ARGS.epochs}")
    if ARGS.max_steps_per_epoch is not None:
        log(f"  max_steps/epoch : {ARGS.max_steps_per_epoch}")

    results = {"config": {**vars(ARGS), **pinfo, "precision": precision_tag}}

    # ── 1. Data & Schema ─────────────────────────────────────────────────
    if ARGS.data_dir:
        log(f"\n  Data source     : {ARGS.data_dir} ({ARGS.split})")
        schema = load_schema(ARGS.data_dir)
        meta = load_metadata(ARGS.data_dir)
        data_src, data_files = get_data_paths(ARGS.data_dir, ARGS.split)
        if not data_files:
            log(f"  ERROR: no Parquet files in {ARGS.data_dir}/{ARGS.split}")
            sys.exit(1)

        import pyarrow.parquet as pq
        total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in data_files)
        log(f"  Parquet files   : {len(data_files)}")
        log(f"  Total rows      : {total_rows:,}")
        if meta:
            log(f"  Avg seq length  : {meta.get('max_seq_len', '?')}")
            log(f"  Item vocab      : {meta.get('item_vocab_size', '?'):,}")
        results["config"]["total_rows"] = total_rows
        results["config"]["data_dir"] = ARGS.data_dir
    else:
        log(f"\n  Data source     : built-in test data ({ARGS.num_rows} synthetic rows)")
        from transformers4rec.data import tabular_sequence_testing_data as td
        schema = td.schema

        tmpdir = os.path.join(tempfile.gettempdir(), "t4rec_bench_v2_data")
        os.makedirs(tmpdir, exist_ok=True)
        data_path = os.path.join(tmpdir, "data.parquet")
        if IS_MAIN:
            t_gen = time.perf_counter()
            generate_fallback_data(
                data_path, ARGS.num_rows, ARGS.max_seq_len, schema,
            )
            log(f"  Generated in {time.perf_counter() - t_gen:.1f}s")
        if IS_DDP:
            torch.distributed.barrier()
        data_src = tmpdir
        results["config"]["total_rows"] = ARGS.num_rows

    # ── 2. Model ─────────────────────────────────────────────────────────
    item_vocab = _get_cardinality(schema, "item_id/list") or 0
    use_sampled = ARGS.sampled_softmax
    if not use_sampled and item_vocab > 0:
        bytes_per_elem = 2 if (ARGS.bf16 or ARGS.fp16) else 4
        logits_gb = (ARGS.batch_size * ARGS.max_seq_len * item_vocab
                     * bytes_per_elem) / (1024 ** 3)
        gpu_gb = (torch.cuda.get_device_properties(LOCAL_RANK).total_memory
                  / (1024 ** 3))
        if logits_gb > gpu_gb * 0.5:
            log(f"\n  ** WARNING: full softmax over {item_vocab:,} items would "
                f"allocate ~{logits_gb:.1f} GB for the logits tensor "
                f"(GPU has {gpu_gb:.1f} GB).")
            log(f"     Auto-enabling sampled softmax with "
                f"{ARGS.softmax_samples:,} negative samples.")
            use_sampled = True

    if use_sampled:
        log(f"  Sampled softmax : {ARGS.softmax_samples:,} negative samples "
            f"(item vocab = {item_vocab:,})")

    # ── 2a. Memory estimation & auto batch-size adjustment ────────────
    gpu_gb = (torch.cuda.get_device_properties(LOCAL_RANK).total_memory
              / (1024 ** 3))
    emb_dim = 64
    n_params_est = (
        item_vocab * emb_dim
        + ARGS.d_model * emb_dim
        + ARGS.n_layer * (4 * ARGS.d_model ** 2 + 4 * ARGS.d_model * 4 * ARGS.d_model)
        + (ARGS.softmax_samples if use_sampled else item_vocab) * emb_dim
    )
    adjusted_bs = auto_adjust_batch_size(ARGS, n_params_est, gpu_gb)
    if adjusted_bs != ARGS.batch_size:
        ARGS.batch_size = adjusted_bs
        results["config"]["batch_size"] = adjusted_bs
        results["config"]["batch_size_auto_reduced"] = True

    log(f"\n  Building model ...")
    model = build_model(
        schema, ARGS.d_model, ARGS.n_head, ARGS.n_layer, ARGS.max_seq_len,
        sampled_softmax=use_sampled, max_n_samples=ARGS.softmax_samples,
    )
    n_params = count_parameters(model)
    model = model.to(DEVICE)
    log(f"  Model params    : {n_params:,}")
    results["config"]["sampled_softmax"] = use_sampled
    results["config"]["item_vocab"] = item_vocab

    if ARGS.compile:
        if hasattr(torch, "compile"):
            log(f"  Compiling model with torch.compile ...")
            model = torch.compile(model)
        else:
            log(f"  torch.compile unavailable, skipping")

    if IS_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
            find_unused_parameters=True,
        )

    results["config"]["n_params"] = n_params

    # ── 3. Precision setup ───────────────────────────────────────────────
    if ARGS.bf16:
        autocast_ctx = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
        scaler = None
    elif ARGS.fp16:
        autocast_ctx = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
        scaler = _GradScaler()
    else:
        autocast_ctx = None
        scaler = None

    # ── 4. Dataloader ────────────────────────────────────────────────────
    streaming_flag = True if ARGS.streaming else None
    dl_paths = data_files if ARGS.data_dir else data_src
    loader = build_dataloader(
        dl_paths, schema, ARGS.batch_size, ARGS.max_seq_len,
        shuffle=True, device_id=LOCAL_RANK,
        global_size=WORLD_SIZE if IS_DDP else None,
        global_rank=LOCAL_RANK if IS_DDP else None,
        streaming=streaming_flag,
        rows_per_chunk=ARGS.rows_per_chunk,
    )
    n_batches = len(loader)
    tier = getattr(loader, "_tier", "unknown")
    log(f"  Dataloader tier : {tier}")
    log(f"  Batches/epoch   : {n_batches}")

    # ── 5. BENCHMARK A — Dataloader throughput ───────────────────────────
    if ARGS.mode == "inference":
        log("\n  [Skipping Benchmark A (dataloader) in inference-only mode]")
    else:
        log("\n" + "=" * 72)
        log("BENCHMARK A: Dataloader throughput (data iteration only)")
        log("=" * 72)

        if hasattr(loader, "set_epoch"):
            loader.set_epoch(0)
        for _ in loader:
            pass
        torch.cuda.synchronize(DEVICE)
        gc.collect()

        dl_times = []
        dl_rows = 0
        for ep in range(ARGS.epochs):
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(ep + 100)
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            n_rows_ep = 0
            for b_in, _ in loader:
                n_rows_ep += next(iter(b_in.values())).shape[0]
            torch.cuda.synchronize(DEVICE)
            dl_times.append(time.perf_counter() - t0)
            dl_rows = n_rows_ep

        dl_avg = np.mean(dl_times)
        dl_std = np.std(dl_times)
        dl_rows_total = dl_rows * WORLD_SIZE if IS_DDP else dl_rows
        dl_throughput = dl_rows_total / dl_avg if dl_avg > 0 else 0

        log(f"  Rows/epoch      : {dl_rows_total:,}")
        log(f"  Avg epoch time  : {dl_avg:.4f}s (+/- {dl_std:.4f}s)")
        log(f"  Throughput      : {dl_throughput:,.0f} rows/sec")
        log(f"  Per-batch       : {dl_avg / max(n_batches, 1) * 1000:.2f} ms")

        results["dataloader"] = {
            "tier": tier,
            "rows_per_epoch": dl_rows_total,
            "avg_epoch_sec": round(dl_avg, 4),
            "std_epoch_sec": round(dl_std, 4),
            "throughput_rows_sec": round(dl_throughput, 1),
            "per_batch_ms": round(dl_avg / max(n_batches, 1) * 1000, 2),
        }

    # ── 6. BENCHMARK B — Training step latency ──────────────────────────
    if ARGS.mode in ("train", "both"):
        log("\n" + "=" * 72)
        log("BENCHMARK B: Training step (fwd + bwd + optimizer)")
        log("=" * 72)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        step_times = []
        losses = []
        step_count = 0
        total_needed = ARGS.warmup_steps + ARGS.bench_steps

        torch.cuda.reset_peak_memory_stats(DEVICE)

        epoch_idx = 0
        while step_count < total_needed:
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch_idx + 200)
            epoch_idx += 1
            for batch in loader:
                if step_count >= total_needed:
                    break

                torch.cuda.synchronize(DEVICE)
                t0 = time.perf_counter()

                loss, _ = _train_step(model, batch, optimizer, scaler,
                                      autocast_ctx)

                torch.cuda.synchronize(DEVICE)
                elapsed = time.perf_counter() - t0

                if step_count >= ARGS.warmup_steps:
                    step_times.append(elapsed)
                    losses.append(loss.item())
                step_count += 1

        step_ms = np.array(step_times) * 1000
        avg_step = np.mean(step_ms)
        std_step = np.std(step_ms)
        p50 = np.percentile(step_ms, 50)
        p95 = np.percentile(step_ms, 95)
        p99 = np.percentile(step_ms, 99)
        eff_bs = ARGS.batch_size * WORLD_SIZE if IS_DDP else ARGS.batch_size
        samples_per_sec = eff_bs / (avg_step / 1000)
        peak_mem = gpu_mem_mb()

        log(f"  Measured steps  : {len(step_times)}")
        log(f"  Avg step time   : {avg_step:.2f} ms (+/- {std_step:.2f})")
        log(f"  P50 / P95 / P99 : {p50:.2f} / {p95:.2f} / {p99:.2f} ms")
        log(f"  Throughput      : {samples_per_sec:,.0f} samples/sec")
        log(f"  Loss (first)    : {losses[0]:.4f}")
        log(f"  Loss (last)     : {losses[-1]:.4f}")
        log(f"  Peak GPU mem    : {peak_mem:.0f} MB")

        results["training_step"] = {
            "measured_steps": len(step_times),
            "avg_step_ms": round(avg_step, 2),
            "std_step_ms": round(std_step, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "throughput_samples_sec": round(samples_per_sec, 1),
            "loss_first": round(losses[0], 4),
            "loss_last": round(losses[-1], 4),
            "peak_gpu_mem_mb": round(peak_mem, 0),
        }

        # ── 6b. BENCHMARK B+ — Forward/Backward/AllReduce Breakdown ──────
        if ARGS.breakdown:
            log("\n" + "-" * 72)
            log("BENCHMARK B+: Forward / Backward / Optimizer Breakdown")
            log("-" * 72)

            bd_fwd, bd_bwd, bd_opt = [], [], []
            bd_count = 0
            ep_bd = 0
            while bd_count < ARGS.warmup_steps + ARGS.breakdown_steps:
                if hasattr(loader, "set_epoch"):
                    loader.set_epoch(ep_bd + 900)
                ep_bd += 1
                for batch in loader:
                    if bd_count >= ARGS.warmup_steps + ARGS.breakdown_steps:
                        break
                    _, timings = _train_step(model, batch, optimizer, scaler,
                                             autocast_ctx, breakdown=True)
                    if bd_count >= ARGS.warmup_steps and timings:
                        bd_fwd.append(timings["fwd_ms"])
                        bd_bwd.append(timings["bwd_ms"])
                        bd_opt.append(timings["opt_ms"])
                    bd_count += 1

            fwd_avg = np.mean(bd_fwd)
            bwd_avg = np.mean(bd_bwd)
            opt_avg = np.mean(bd_opt)
            total_avg = fwd_avg + bwd_avg + opt_avg

            log(f"  Steps measured  : {len(bd_fwd)}")
            log(f"  Forward         : {fwd_avg:.2f} ms  "
                f"({fwd_avg / total_avg * 100:.1f}%)")
            log(f"  Backward (incl. DDP sync) : {bwd_avg:.2f} ms  "
                f"({bwd_avg / total_avg * 100:.1f}%)")
            log(f"  Optimizer       : {opt_avg:.2f} ms  "
                f"({opt_avg / total_avg * 100:.1f}%)")
            log(f"  Total (sum)     : {total_avg:.2f} ms")
            if IS_DDP:
                log(f"  Note: DDP all-reduce is overlapped with backward. "
                    f"Use scaling tests (1 vs N GPU) or profiler for "
                    f"communication cost.")

            results["breakdown"] = {
                "steps": len(bd_fwd),
                "fwd_avg_ms": round(fwd_avg, 2),
                "bwd_avg_ms": round(bwd_avg, 2),
                "opt_avg_ms": round(opt_avg, 2),
                "total_avg_ms": round(total_avg, 2),
                "fwd_pct": round(fwd_avg / total_avg * 100, 1),
                "bwd_pct": round(bwd_avg / total_avg * 100, 1),
                "opt_pct": round(opt_avg / total_avg * 100, 1),
            }

    else:
        eff_bs = ARGS.batch_size * WORLD_SIZE if IS_DDP else ARGS.batch_size
        avg_step = 0

    # ── 7. BENCHMARK C — Full epoch ─────────────────────────────────────
    if ARGS.mode in ("train", "both"):
        log("\n" + "=" * 72)
        log("BENCHMARK C: Full training epochs (end-to-end)")
        log("=" * 72)

        if IS_DDP:
            torch.distributed.barrier()

        if IS_DDP:
            local_n = torch.tensor([n_batches], device=DEVICE, dtype=torch.long)
            torch.distributed.all_reduce(local_n, op=torch.distributed.ReduceOp.MIN)
            max_steps_per_epoch = int(local_n.item())
            log(f"  DDP sync: {max_steps_per_epoch} batches/rank "
                f"(min across {WORLD_SIZE} ranks)")
        else:
            max_steps_per_epoch = n_batches

        if ARGS.max_steps_per_epoch is not None:
            max_steps_per_epoch = min(max_steps_per_epoch,
                                     ARGS.max_steps_per_epoch)
            log(f"  Capped to {max_steps_per_epoch} steps/epoch "
                f"(--max-steps-per-epoch)")

        est_sec = max_steps_per_epoch * (avg_step / 1000)
        log(f"  Est. time/epoch : ~{est_sec:.0f}s ({max_steps_per_epoch} steps "
            f"x {avg_step:.0f}ms)")
        log(f"  Total est.      : ~{est_sec * (1 + ARGS.epochs):.0f}s "
            f"(1 warmup + {ARGS.epochs} measured)")

        progress_interval = max(max_steps_per_epoch // 10, 1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # -- Build validation loader for ranking-metric evaluation ------
        valid_loader = None
        if ARGS.eval_recall and ARGS.data_dir:
            valid_src, valid_files = get_data_paths(ARGS.data_dir, "valid")
            if valid_files:
                valid_loader = build_dataloader(
                    valid_files, schema, ARGS.batch_size, ARGS.max_seq_len,
                    shuffle=False, device_id=LOCAL_RANK,
                    global_size=None, global_rank=None,
                    streaming=None, rows_per_chunk=ARGS.rows_per_chunk,
                )
                log(f"  Eval metrics    : every {ARGS.eval_interval} steps, "
                    f"{ARGS.eval_batches} batches, "
                    f"{len(valid_files)} valid files")
            else:
                log(f"  WARNING: no valid/ files found, skipping eval")

        recall_history = []

        # Warmup (limited number of steps to warm up GPU / JIT)
        wu_steps = min(ARGS.warmup_steps, max_steps_per_epoch)
        log(f"  Warmup ({wu_steps} steps) ...")
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(300)
        wu_t0 = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= wu_steps:
                break
            _train_step(model, batch, optimizer, scaler, autocast_ctx)
        torch.cuda.synchronize(DEVICE)
        log(f"  Warmup done in {time.perf_counter() - wu_t0:.1f}s")
        gc.collect()

        epoch_times = []
        epoch_steps = []
        loss_history = []
        global_step = 0
        loss_log_interval = progress_interval
        for ep in range(ARGS.epochs):
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(ep + 400)
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            n_steps = 0
            recent_losses = []
            for i, batch in enumerate(loader):
                if i >= max_steps_per_epoch:
                    break
                loss_val, _ = _train_step(model, batch, optimizer, scaler,
                                          autocast_ctx)
                n_steps += 1
                global_step += 1
                recent_losses.append(loss_val.item())

                if IS_MAIN and (i + 1) % progress_interval == 0:
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / (i + 1) * (max_steps_per_epoch - i - 1)
                    avg_loss = np.mean(recent_losses[-progress_interval:])
                    print(f"    epoch {ep + 1}: {i + 1}/{max_steps_per_epoch} "
                          f"steps ({elapsed:.1f}s, ETA {eta:.0f}s, "
                          f"loss={avg_loss:.4f})",
                          flush=True)

                if global_step % loss_log_interval == 0:
                    avg_loss = np.mean(recent_losses[-loss_log_interval:])
                    loss_history.append({
                        "step": global_step, "epoch": ep + 1,
                        "loss": round(float(avg_loss), 4),
                    })

                # Ranking-metric evaluation at fixed step intervals
                if (valid_loader is not None
                        and global_step % ARGS.eval_interval == 0):
                    if hasattr(valid_loader, "set_epoch"):
                        valid_loader.set_epoch(global_step)
                    eval_metrics, n_eval = evaluate_ranking_metrics(
                        model, valid_loader, autocast_ctx, ARGS.eval_batches,
                    )
                    entry = {"step": global_step, "epoch": ep + 1,
                             "eval_batches": n_eval, **eval_metrics}
                    recall_history.append(entry)
                    metric_strs = [f"{k}={v:.4f}" for k, v in
                                   eval_metrics.items()]
                    log(f"    [step {global_step}] "
                        + ", ".join(metric_strs[:4]))

            torch.cuda.synchronize(DEVICE)
            epoch_times.append(time.perf_counter() - t0)
            epoch_steps.append(n_steps)
            log(f"  Epoch {ep + 1}/{ARGS.epochs}: {epoch_times[-1]:.3f}s  "
                f"({n_steps} steps)")

        ep_avg = np.mean(epoch_times)
        ep_std = np.std(epoch_times)
        avg_steps = int(np.mean(epoch_steps))
        rows_per_epoch = avg_steps * ARGS.batch_size
        if IS_DDP:
            rows_per_epoch *= WORLD_SIZE
        e2e_throughput = rows_per_epoch / ep_avg if ep_avg > 0 else 0

        log(f"\n  Avg epoch time  : {ep_avg:.3f}s (+/- {ep_std:.3f}s)")
        log(f"  Steps/epoch     : {avg_steps}")
        log(f"  Throughput      : {e2e_throughput:,.0f} samples/sec (end-to-end)")

        results["full_epoch"] = {
            "avg_epoch_sec": round(ep_avg, 3),
            "std_epoch_sec": round(ep_std, 3),
            "steps_per_epoch": avg_steps,
            "samples_per_epoch": rows_per_epoch,
            "throughput_samples_sec": round(e2e_throughput, 1),
        }

        if loss_history:
            results["loss_curve"] = loss_history
            log(f"  Loss (first)    : {loss_history[0]['loss']:.4f}")
            log(f"  Loss (last)     : {loss_history[-1]['loss']:.4f}")
        if recall_history:
            results["eval_metrics"] = recall_history

    # ── 8. BENCHMARK D — Inference ────────────────────────────────────
    if ARGS.mode in ("inference", "both"):
        infer_results = run_inference_benchmark(
            model, schema, autocast_ctx, item_vocab, ARGS,
        )
        results["inference"] = infer_results

    # ── Summary ──────────────────────────────────────────────────────────
    log("\n" + "=" * 72)
    log("SUMMARY")
    log("=" * 72)
    log(f"  Platform             : {pinfo['platform']}")
    log(f"  GPU                  : {pinfo['gpu_name']} x{WORLD_SIZE}")
    m = ARGS.model_size or "custom"
    log(f"  Model                : XLNet {m} (d={ARGS.d_model} L={ARGS.n_layer} "
        f"H={ARGS.n_head}, {n_params:,} params)")
    log(f"  Batch size           : {ARGS.batch_size} x{WORLD_SIZE} = {eff_bs} effective")
    log(f"  Precision            : {precision_tag}")
    log(f"  torch.compile        : {ARGS.compile}")
    log(f"  Dataloader tier      : {tier}")
    log(f"  Mode                 : {ARGS.mode}")
    if "dataloader" in results:
        log(f"  DL throughput        : "
            f"{results['dataloader']['throughput_rows_sec']:,.0f} rows/sec")
    if "training_step" in results:
        log(f"  Step throughput      : "
            f"{results['training_step']['throughput_samples_sec']:,.0f} samples/sec")
        log(f"  Avg step latency     : "
            f"{results['training_step']['avg_step_ms']:.2f} ms")
        log(f"  Peak GPU memory      : "
            f"{results['training_step']['peak_gpu_mem_mb']:.0f} MB")
    if "full_epoch" in results:
        log(f"  Epoch throughput     : "
            f"{results['full_epoch']['throughput_samples_sec']:,.0f} samples/sec (e2e)")
    if "breakdown" in results:
        bd = results["breakdown"]
        log(f"  Fwd / Bwd / Opt      : "
            f"{bd['fwd_avg_ms']:.1f} / {bd['bwd_avg_ms']:.1f} / "
            f"{bd['opt_avg_ms']:.1f} ms")
        if IS_DDP:
            log(f"  Note               : Bwd includes overlapped DDP sync")
    if "loss_curve" in results and results["loss_curve"]:
        lc = results["loss_curve"]
        log(f"  Loss curve           : {lc[0]['loss']:.4f} → "
            f"{lc[-1]['loss']:.4f} ({len(lc)} checkpoints)")
    if "eval_metrics" in results and results["eval_metrics"]:
        last_eval = results["eval_metrics"][-1]
        metric_strs = [f"{k}={v:.4f}" for k, v in last_eval.items()
                       if k not in ("step", "epoch", "eval_batches")
                       and isinstance(v, (int, float))]
        log(f"  Eval (step {last_eval['step']})  : "
            + ", ".join(metric_strs[:4]))
    if "inference" in results:
        log(f"  Inference configs    : {len(results['inference'])} "
            f"(BS x SeqLen)")

    # ── Write JSON ───────────────────────────────────────────────────────
    if IS_MAIN and ARGS.output_json:
        with open(ARGS.output_json, "w") as f:
            json.dump(results, f, indent=2)
        log(f"\n  Results written to: {ARGS.output_json}")

    if IS_DDP:
        torch.distributed.destroy_process_group()

    log("\nDone.")


if __name__ == "__main__":
    main()
