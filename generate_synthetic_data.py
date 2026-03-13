#!/usr/bin/env python3
"""
Large-Scale Synthetic Data Generator for Transformers4Rec Benchmarks
====================================================================

Generates session-level Parquet datasets with Zipfian item distribution,
matching the schema expected by Transformers4Rec's dataloader and model
pipeline.  Designed to produce datasets at the scale requested by
production benchmarking (up to 50M+ sessions, 5M item vocab).

Output structure::

    <output_dir>/
        schema.json          # Merlin-compatible schema
        metadata.json        # Dataset statistics
        train/
            part_000.parquet
            part_001.parquet
            ...
        valid/
            part_000.parquet
            ...

Usage::

    # Quick test (100K sessions, ~30 sec)
    python generate_synthetic_data.py --output /nvme_local/t4rec_data/test_100k \
        --num-sessions 100000 --max-seq-len 200

    # Medium scale (1M sessions)
    python generate_synthetic_data.py --output /nvme_local/t4rec_data/medium_1m \
        --num-sessions 1000000 --max-seq-len 400

    # Production scale (5M sessions, ~1.25B interactions)
    python generate_synthetic_data.py --output /nvme_local/t4rec_data/prod_5m \
        --num-sessions 5000000 --item-vocab 5000000 --max-seq-len 400
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ── Defaults matching customer requirements ─────────────────────────────
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


def build_zipf_cdf(vocab_size, exponent):
    """Build a normalised CDF for inverse-CDF sampling from Zipf's law."""
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, exponent)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return cdf.astype(np.float64)


def sample_zipfian(n, cdf, rng):
    """Draw *n* samples from a Zipfian distribution via inverse-CDF."""
    u = rng.random(n).astype(np.float64)
    return np.searchsorted(cdf, u).astype(np.int32) + 1  # 1-indexed


def generate_file(
    path,
    n_sessions,
    item_cdf,
    min_seq_len,
    max_seq_len,
    num_categories,
    num_brands,
    num_price_buckets,
    num_countries,
    row_group_size,
    seed,
    session_id_offset,
):
    """Generate one Parquet file of *n_sessions* sessions."""
    rng = np.random.RandomState(seed)

    # Session lengths: uniform in [min_seq_len, max_seq_len]
    seq_lens = rng.randint(min_seq_len, max_seq_len + 1, size=n_sessions)
    total_items = int(seq_lens.sum())

    # ── Item IDs (Zipfian) ──────────────────────────────────────────
    flat_items = sample_zipfian(total_items, item_cdf, rng)

    # ── Derived categorical features (deterministic from item_id) ───
    flat_category = (flat_items % num_categories).astype(np.int32) + 1
    flat_brand = ((flat_items * 7 + 13) % num_brands).astype(np.int32) + 1
    flat_price = ((flat_items * 3 + 7) % num_price_buckets).astype(np.int32) + 1

    # ── Continuous time features ────────────────────────────────────
    flat_hour_sin = rng.uniform(-1.0, 1.0, total_items).astype(np.float32)
    flat_hour_cos = rng.uniform(-1.0, 1.0, total_items).astype(np.float32)
    flat_weekday_sin = rng.uniform(-1.0, 1.0, total_items).astype(np.float32)
    flat_weekday_cos = rng.uniform(-1.0, 1.0, total_items).astype(np.float32)
    flat_recency = rng.uniform(-3.0, 3.0, total_items).astype(np.float32)

    # ── Split flat arrays into per-session lists ────────────────────
    offsets = np.empty(n_sessions + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(seq_lens, out=offsets[1:])

    def _to_lists(flat):
        return [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(n_sessions)]

    col_item = _to_lists(flat_items)
    col_category = _to_lists(flat_category)
    col_brand = _to_lists(flat_brand)
    col_price = _to_lists(flat_price)
    col_hour_sin = _to_lists(flat_hour_sin)
    col_hour_cos = _to_lists(flat_hour_cos)
    col_weekday_sin = _to_lists(flat_weekday_sin)
    col_weekday_cos = _to_lists(flat_weekday_cos)
    col_recency = _to_lists(flat_recency)

    # ── Scalar features ─────────────────────────────────────────────
    col_country = rng.randint(1, num_countries + 1, size=n_sessions).tolist()

    # ── Build Arrow table ───────────────────────────────────────────
    table = pa.table({
        "item_id/list": pa.array(col_item, type=pa.list_(pa.int32())),
        "category/list": pa.array(col_category, type=pa.list_(pa.int32())),
        "brand/list": pa.array(col_brand, type=pa.list_(pa.int32())),
        "price_bucket/list": pa.array(col_price, type=pa.list_(pa.int32())),
        "event_hour_sin/list": pa.array(col_hour_sin, type=pa.list_(pa.float32())),
        "event_hour_cos/list": pa.array(col_hour_cos, type=pa.list_(pa.float32())),
        "event_weekday_sin/list": pa.array(col_weekday_sin, type=pa.list_(pa.float32())),
        "event_weekday_cos/list": pa.array(col_weekday_cos, type=pa.list_(pa.float32())),
        "event_recency/list": pa.array(col_recency, type=pa.list_(pa.float32())),
        "user_country": pa.array(col_country, type=pa.int32()),
    })

    pq.write_table(table, path, row_group_size=row_group_size)
    return total_items


def write_schema(output_dir, item_vocab, num_categories, num_brands,
                 num_price_buckets, num_countries, min_seq_len, max_seq_len):
    """Write a Merlin-compatible schema.json."""
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
        _list_int_feature(
            "item_id/list", item_vocab,
            ["item_id", "list", "categorical", "item"],
        ),
        _list_int_feature(
            "category/list", num_categories,
            ["list", "categorical", "item"],
        ),
        _list_int_feature(
            "brand/list", num_brands,
            ["list", "categorical", "item"],
        ),
        _list_int_feature(
            "price_bucket/list", num_price_buckets,
            ["list", "categorical", "item"],
        ),
        _list_float_feature(
            "event_hour_sin/list", -1.0, 1.0,
            ["continuous", "time", "list"],
        ),
        _list_float_feature(
            "event_hour_cos/list", -1.0, 1.0,
            ["continuous", "time", "list"],
        ),
        _list_float_feature(
            "event_weekday_sin/list", -1.0, 1.0,
            ["continuous", "time", "list"],
        ),
        _list_float_feature(
            "event_weekday_cos/list", -1.0, 1.0,
            ["continuous", "time", "list"],
        ),
        _list_float_feature(
            "event_recency/list", -3.0, 3.0,
            ["continuous", "list"],
        ),
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
    parser = argparse.ArgumentParser(
        description="Generate large-scale synthetic T4Rec benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory (e.g. /nvme_local/t4rec_data/prod_5m)",
    )
    parser.add_argument("--num-sessions", type=int, default=1_000_000)
    parser.add_argument("--item-vocab", type=int, default=DEFAULT_ITEM_VOCAB)
    parser.add_argument("--num-categories", type=int, default=DEFAULT_NUM_CATEGORIES)
    parser.add_argument("--num-brands", type=int, default=DEFAULT_NUM_BRANDS)
    parser.add_argument("--num-price-buckets", type=int, default=DEFAULT_NUM_PRICE_BUCKETS)
    parser.add_argument("--num-countries", type=int, default=DEFAULT_NUM_COUNTRIES)
    parser.add_argument("--zipf-exponent", type=float, default=DEFAULT_ZIPF_EXPONENT)
    parser.add_argument("--min-seq-len", type=int, default=DEFAULT_MIN_SEQ_LEN)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--sessions-per-file", type=int, default=DEFAULT_SESSIONS_PER_FILE,
    )
    parser.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    parser.add_argument("--valid-fraction", type=float, default=DEFAULT_VALID_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 72)
    print("Transformers4Rec Synthetic Data Generator")
    print("=" * 72)
    for k, v in vars(args).items():
        print(f"  {k:<22}: {v}")

    # ── Setup ───────────────────────────────────────────────────────
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

    # ── Build Zipfian CDF ───────────────────────────────────────────
    print(f"\n  Building Zipf CDF (vocab={args.item_vocab:,}, "
          f"exponent={args.zipf_exponent}) ...")
    t0 = time.perf_counter()
    item_cdf = build_zipf_cdf(args.item_vocab, args.zipf_exponent)
    print(f"  CDF built in {time.perf_counter() - t0:.1f}s "
          f"({item_cdf.nbytes / 1e6:.0f} MB)")

    # ── Generate training data ──────────────────────────────────────
    print(f"\n  Generating training data ...")
    t0 = time.perf_counter()
    total_interactions = 0
    session_offset = 0
    for fi in range(n_train_files):
        ns = min(spf, n_train - fi * spf)
        path = os.path.join(train_dir, f"part_{fi:04d}.parquet")
        ni = generate_file(
            path, ns, item_cdf,
            args.min_seq_len, args.max_seq_len,
            args.num_categories, args.num_brands,
            args.num_price_buckets, args.num_countries,
            args.row_group_size,
            seed=args.seed + fi,
            session_id_offset=session_offset,
        )
        total_interactions += ni
        session_offset += ns
        elapsed = time.perf_counter() - t0
        rate = session_offset / elapsed if elapsed > 0 else 0
        print(f"    [{fi + 1}/{n_train_files}] {ns:,} sessions, "
              f"{ni:,} interactions  "
              f"({rate:,.0f} sess/s)", flush=True)

    train_time = time.perf_counter() - t0
    print(f"  Train done: {n_train:,} sessions, "
          f"{total_interactions:,} interactions in {train_time:.1f}s")

    # ── Generate validation data ────────────────────────────────────
    print(f"\n  Generating validation data ...")
    t0v = time.perf_counter()
    valid_interactions = 0
    for fi in range(n_valid_files):
        ns = min(spf, n_valid - fi * spf)
        path = os.path.join(valid_dir, f"part_{fi:04d}.parquet")
        ni = generate_file(
            path, ns, item_cdf,
            args.min_seq_len, args.max_seq_len,
            args.num_categories, args.num_brands,
            args.num_price_buckets, args.num_countries,
            args.row_group_size,
            seed=args.seed + n_train_files + fi,
            session_id_offset=session_offset,
        )
        valid_interactions += ni
        session_offset += ns
    valid_time = time.perf_counter() - t0v
    print(f"  Valid done: {n_valid:,} sessions, "
          f"{valid_interactions:,} interactions in {valid_time:.1f}s")

    # ── Write schema ────────────────────────────────────────────────
    write_schema(
        args.output, args.item_vocab, args.num_categories,
        args.num_brands, args.num_price_buckets, args.num_countries,
        args.min_seq_len, args.max_seq_len,
    )

    # ── Write metadata ──────────────────────────────────────────────
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

    # ── Summary ─────────────────────────────────────────────────────
    total_time = train_time + valid_time
    total_size = 0
    for root, _, files in os.walk(args.output):
        for fn in files:
            total_size += os.path.getsize(os.path.join(root, fn))

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
