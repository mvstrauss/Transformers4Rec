#!/usr/bin/env python3
"""
Parallel optimized version of generate_synthetic_data.py.

Key optimizations over the original:
  1. pa.ListArray.from_arrays() instead of Python list-of-lists conversion
  2. multiprocessing.Pool to generate files across all available CPU cores
  3. Shared Zipf CDF via fork (no pickle overhead)

Output is identical to the sequential version given the same seed,
because each file uses seed = base_seed + file_index.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ITEM_VOCAB = 5_000_000
DEFAULT_NUM_CATEGORIES = 1_000
DEFAULT_NUM_BRANDS = 5_000
DEFAULT_NUM_PRICE_BUCKETS = 20
DEFAULT_NUM_COUNTRIES = 200
DEFAULT_ZIPF_EXPONENT = 1.2
DEFAULT_MIN_SEQ_LEN = 20
DEFAULT_MAX_SEQ_LEN = 400
DEFAULT_SESSIONS_PER_FILE = 200_000
DEFAULT_ROW_GROUP_SIZE = 50_000
DEFAULT_VALID_FRACTION = 0.05

_ITEM_CDF = None  # module-level for fork-based sharing


def build_zipf_cdf(vocab_size, exponent):
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, exponent)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return cdf.astype(np.float64)


def sample_zipfian(n, cdf, rng):
    u = rng.random(n).astype(np.float64)
    return np.searchsorted(cdf, u).astype(np.int32) + 1


def generate_file_fast(args_tuple):
    """Generate one Parquet file using PyArrow native list construction."""
    (path, n_sessions, min_seq_len, max_seq_len,
     num_categories, num_brands, num_price_buckets,
     num_countries, row_group_size, seed) = args_tuple

    item_cdf = _ITEM_CDF
    rng = np.random.RandomState(seed)

    seq_lens = rng.randint(min_seq_len, max_seq_len + 1, size=n_sessions)
    total_items = int(seq_lens.sum())

    flat_items = sample_zipfian(total_items, item_cdf, rng)

    flat_category = (flat_items % num_categories).astype(np.int32) + 1
    flat_brand = ((flat_items * 7 + 13) % num_brands).astype(np.int32) + 1
    flat_price = ((flat_items * 3 + 7) % num_price_buckets).astype(np.int32) + 1

    flat_hour_sin = rng.uniform(-1.0, 1.0, total_items).astype(np.float32)
    flat_hour_cos = rng.uniform(-1.0, 1.0, total_items).astype(np.float32)
    flat_weekday_sin = rng.uniform(-1.0, 1.0, total_items).astype(np.float32)
    flat_weekday_cos = rng.uniform(-1.0, 1.0, total_items).astype(np.float32)
    flat_recency = rng.uniform(-3.0, 3.0, total_items).astype(np.float32)

    offsets = np.empty(n_sessions + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(seq_lens, out=offsets[1:])
    pa_offsets = pa.array(offsets, type=pa.int64())

    def _make_list_col(flat_np, pa_type):
        return pa.ListArray.from_arrays(pa_offsets, pa.array(flat_np, type=pa_type))

    table = pa.table({
        "item_id/list": _make_list_col(flat_items, pa.int32()),
        "category/list": _make_list_col(flat_category, pa.int32()),
        "brand/list": _make_list_col(flat_brand, pa.int32()),
        "price_bucket/list": _make_list_col(flat_price, pa.int32()),
        "event_hour_sin/list": _make_list_col(flat_hour_sin, pa.float32()),
        "event_hour_cos/list": _make_list_col(flat_hour_cos, pa.float32()),
        "event_weekday_sin/list": _make_list_col(flat_weekday_sin, pa.float32()),
        "event_weekday_cos/list": _make_list_col(flat_weekday_cos, pa.float32()),
        "event_recency/list": _make_list_col(flat_recency, pa.float32()),
        "user_country": pa.array(
            rng.randint(1, num_countries + 1, size=n_sessions).astype(np.int32),
            type=pa.int32(),
        ),
    })

    pq.write_table(table, path, row_group_size=row_group_size)
    file_size = os.path.getsize(path)
    return total_items, file_size


def write_schema(output_dir, item_vocab, num_categories, num_brands,
                 num_price_buckets, num_countries, min_seq_len, max_seq_len):
    def _list_int_feature(name, max_val, tags):
        return {
            "name": name,
            "valueCount": {"min": str(min_seq_len), "max": str(max_seq_len)},
            "type": "INT",
            "intDomain": {
                "name": name, "min": "1", "max": str(max_val),
                "isCategorical": True,
            },
            "annotation": {"tag": tags},
        }

    def _list_float_feature(name, lo, hi, tags):
        return {
            "name": name,
            "valueCount": {"min": str(min_seq_len), "max": str(max_seq_len)},
            "type": "FLOAT",
            "floatDomain": {"name": name, "min": lo, "max": hi},
            "annotation": {"tag": tags},
        }

    schema = {"feature": [
        _list_int_feature("item_id/list", item_vocab,
                          ["item_id", "list", "categorical", "item"]),
        _list_int_feature("category/list", num_categories,
                          ["list", "categorical", "item"]),
        _list_int_feature("brand/list", num_brands,
                          ["list", "categorical", "item"]),
        _list_int_feature("price_bucket/list", num_price_buckets,
                          ["list", "categorical", "item"]),
        _list_float_feature("event_hour_sin/list", -1.0, 1.0,
                            ["continuous", "time", "list"]),
        _list_float_feature("event_hour_cos/list", -1.0, 1.0,
                            ["continuous", "time", "list"]),
        _list_float_feature("event_weekday_sin/list", -1.0, 1.0,
                            ["continuous", "time", "list"]),
        _list_float_feature("event_weekday_cos/list", -1.0, 1.0,
                            ["continuous", "time", "list"]),
        _list_float_feature("event_recency/list", -3.0, 3.0,
                            ["continuous", "list"]),
        {
            "name": "user_country",
            "type": "INT",
            "intDomain": {
                "name": "user_country", "min": "1",
                "max": str(num_countries), "isCategorical": True,
            },
            "annotation": {"tag": ["categorical"]},
        },
    ]}

    with open(os.path.join(output_dir, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)


def main():
    global _ITEM_CDF

    parser = argparse.ArgumentParser(
        description="Parallel synthetic data generator for T4Rec benchmarks",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-sessions", type=int, default=1_000_000)
    parser.add_argument("--item-vocab", type=int, default=DEFAULT_ITEM_VOCAB)
    parser.add_argument("--num-categories", type=int, default=DEFAULT_NUM_CATEGORIES)
    parser.add_argument("--num-brands", type=int, default=DEFAULT_NUM_BRANDS)
    parser.add_argument("--num-price-buckets", type=int, default=DEFAULT_NUM_PRICE_BUCKETS)
    parser.add_argument("--num-countries", type=int, default=DEFAULT_NUM_COUNTRIES)
    parser.add_argument("--zipf-exponent", type=float, default=DEFAULT_ZIPF_EXPONENT)
    parser.add_argument("--min-seq-len", type=int, default=DEFAULT_MIN_SEQ_LEN)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--sessions-per-file", type=int, default=DEFAULT_SESSIONS_PER_FILE)
    parser.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    parser.add_argument("--valid-fraction", type=float, default=DEFAULT_VALID_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = auto = nproc/2)")
    args = parser.parse_args()

    workers = args.workers or max(1, cpu_count() // 2)
    workers = min(workers, 128)

    print("=" * 72)
    print("Transformers4Rec Synthetic Data Generator (PARALLEL)")
    print("=" * 72)
    for k, v in vars(args).items():
        print(f"  {k:<22}: {v}")
    print(f"  {'workers':<22}: {workers}")

    train_dir = os.path.join(args.output, "train")
    valid_dir = os.path.join(args.output, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    n_valid = max(1, int(args.num_sessions * args.valid_fraction))
    n_train = args.num_sessions - n_valid
    spf = args.sessions_per_file
    n_train_files = (n_train + spf - 1) // spf
    n_valid_files = max(1, (n_valid + spf - 1) // spf)

    print(f"\n  Train sessions  : {n_train:,}  ({n_train_files} files)")
    print(f"  Valid sessions  : {n_valid:,}  ({n_valid_files} files)")

    print(f"\n  Building Zipf CDF (vocab={args.item_vocab:,}, "
          f"exponent={args.zipf_exponent}) ...")
    t0 = time.perf_counter()
    _ITEM_CDF = build_zipf_cdf(args.item_vocab, args.zipf_exponent)
    print(f"  CDF built in {time.perf_counter() - t0:.1f}s "
          f"({_ITEM_CDF.nbytes / 1e6:.0f} MB)")

    # Build task lists
    train_tasks = []
    for fi in range(n_train_files):
        ns = min(spf, n_train - fi * spf)
        path = os.path.join(train_dir, f"part_{fi:04d}.parquet")
        train_tasks.append((
            path, ns, args.min_seq_len, args.max_seq_len,
            args.num_categories, args.num_brands, args.num_price_buckets,
            args.num_countries, args.row_group_size, args.seed + fi,
        ))

    valid_tasks = []
    for fi in range(n_valid_files):
        ns = min(spf, n_valid - fi * spf)
        path = os.path.join(valid_dir, f"part_{fi:04d}.parquet")
        valid_tasks.append((
            path, ns, args.min_seq_len, args.max_seq_len,
            args.num_categories, args.num_brands, args.num_price_buckets,
            args.num_countries, args.row_group_size,
            args.seed + n_train_files + fi,
        ))

    # Generate training data in parallel
    print(f"\n  Generating training data ({n_train_files} files, {workers} workers) ...")
    t0 = time.perf_counter()
    total_interactions = 0
    total_size = 0
    done = 0

    with Pool(workers) as pool:
        for ni, fsize in pool.imap_unordered(generate_file_fast, train_tasks):
            total_interactions += ni
            total_size += fsize
            done += 1
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            if done % 10 == 0 or done == n_train_files:
                print(f"    [{done}/{n_train_files}] "
                      f"{rate:.1f} files/s, "
                      f"{total_size / 1e9:.1f} GB written, "
                      f"elapsed {elapsed:.0f}s", flush=True)

    train_time = time.perf_counter() - t0
    print(f"  Train done: {n_train:,} sessions, "
          f"{total_interactions:,} interactions, "
          f"{total_size / 1e9:.1f} GB in {train_time:.1f}s")

    # Generate validation data
    print(f"\n  Generating validation data ({n_valid_files} files) ...")
    t0v = time.perf_counter()
    valid_interactions = 0

    with Pool(min(workers, n_valid_files)) as pool:
        for ni, fsize in pool.imap_unordered(generate_file_fast, valid_tasks):
            valid_interactions += ni
            total_size += fsize

    valid_time = time.perf_counter() - t0v
    print(f"  Valid done: {n_valid:,} sessions, "
          f"{valid_interactions:,} interactions in {valid_time:.1f}s")

    write_schema(
        args.output, args.item_vocab, args.num_categories,
        args.num_brands, args.num_price_buckets, args.num_countries,
        args.min_seq_len, args.max_seq_len,
    )

    metadata = {
        "num_train_sessions": n_train,
        "num_valid_sessions": n_valid,
        "total_train_interactions": total_interactions,
        "total_valid_interactions": valid_interactions,
        "item_vocab_size": args.item_vocab,
        "num_features": 10,
        "list_features": [
            "item_id/list", "category/list", "brand/list",
            "price_bucket/list", "event_hour_sin/list",
            "event_hour_cos/list", "event_weekday_sin/list",
            "event_weekday_cos/list", "event_recency/list",
        ],
        "scalar_features": ["user_country"],
        "categorical_features": [
            "item_id/list", "category/list", "brand/list",
            "price_bucket/list", "user_country",
        ],
        "continuous_features": [
            "event_hour_sin/list", "event_hour_cos/list",
            "event_weekday_sin/list", "event_weekday_cos/list",
            "event_recency/list",
        ],
        "min_seq_len": args.min_seq_len,
        "max_seq_len": args.max_seq_len,
        "zipf_exponent": args.zipf_exponent,
        "sessions_per_file": args.sessions_per_file,
        "row_group_size": args.row_group_size,
        "seed": args.seed,
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = train_time + valid_time
    print(f"\n{'=' * 72}")
    print(f"  Output dir       : {args.output}")
    print(f"  Total sessions   : {args.num_sessions:,}")
    print(f"  Total interactions: {total_interactions + valid_interactions:,}")
    print(f"  Avg seq length   : {(total_interactions + valid_interactions) / args.num_sessions:.0f}")
    print(f"  Disk size        : {total_size / 1e9:.2f} GB")
    print(f"  Generation time  : {total_time:.1f}s")
    print(f"  Schema           : {os.path.join(args.output, 'schema.json')}")
    print(f"  Metadata         : {os.path.join(args.output, 'metadata.json')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
