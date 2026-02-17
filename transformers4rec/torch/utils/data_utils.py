#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import glob as glob_module
import logging
import warnings
from abc import ABC

import numpy as np
import torch
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch.utils.data import Dataset, IterableDataset

from ...utils import dependencies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Merlin schema & registry -- pure-Python packages that should install
# without CUDA.  Guarded so the module can still load if merlin is absent.
# ---------------------------------------------------------------------------
try:
    from merlin.models.utils.registry import Registry
    from merlin.schema import Tags
    from merlin.schema.io.tensorflow_metadata import TensorflowMetadata
    from merlin_standard_lib import Schema

    dataloader_registry: Registry = Registry("torch.dataloader_loader")
except ImportError:
    Registry = None  # type: ignore[assignment,misc]
    Tags = None  # type: ignore[assignment,misc]
    TensorflowMetadata = None  # type: ignore[assignment,misc]
    Schema = None  # type: ignore[assignment,misc]

    class _FallbackRegistry:
        """Minimal registry so ROCmDataLoader can register without merlin."""

        def __init__(self):
            self._registry = {}

        def register_with_multiple_names(self, *names):
            def decorator(cls):
                for name in names:
                    self._registry[name] = cls
                return cls
            return decorator

        def parse(self, name):
            return self._registry[name]

    dataloader_registry = _FallbackRegistry()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# NVIDIA Merlin dataloader -- only available on CUDA systems with the
# merlin-dataloader package installed.  Guarded so ROCm systems can skip it.
# ---------------------------------------------------------------------------
_merlin_loader_available = False
try:
    from merlin.dataloader.torch import Loader
    from merlin.models.utils.misc_utils import validate_dataset

    _merlin_loader_available = True
except ImportError:
    Loader = None  # type: ignore[assignment,misc]
    validate_dataset = None  # type: ignore[assignment,misc]

from transformers4rec.torch.utils.padding import pad_batch


class T4RecDataLoader(ABC):
    """
    Base Helper class to build dataloader from the schema with properties
    required by T4Rec Trainer class.
    """

    @classmethod
    def from_schema(
        self, schema: Schema, paths_or_dataset, batch_size, max_sequence_length, **kwargs
    ):
        # Build the data-loader from the schema
        raise NotImplementedError

    def set_dataset(self, paths_or_dataset):
        # set the dataset from paths
        # or from provided dataset
        raise NotImplementedError

    @classmethod
    def parse(cls, class_or_str):
        return dataloader_registry.parse(class_or_str)


if dependencies.is_pyarrow_available():
    import pyarrow.parquet as pq

    @dataloader_registry.register_with_multiple_names("pyarrow_builder", "pyarrow")
    class PyarrowDataLoader(T4RecDataLoader, PyTorchDataLoader):
        def __init__(
            self,
            paths_or_dataset,
            batch_size,
            max_sequence_length,
            cols_to_read=None,
            target_names=None,
            shuffle=False,
            shuffle_buffer_size=0,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            **kwargs,
        ):
            T4RecDataLoader.__init__(self)
            warnings.warn(
                "The `pyarrow` data loader is deprecated and should be replaced "
                "by `merlin_dataloader`",
                DeprecationWarning,
            )
            self.paths_or_dataset = paths_or_dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.shuffle_buffer_size = shuffle_buffer_size
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.max_sequence_length = max_sequence_length
            self.drop_last = drop_last

            self.set_dataset(cols_to_read=cols_to_read, target_names=target_names)

            PyTorchDataLoader.__init__(
                self,
                self.dataset,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            # set _batch_size attribute needed by HF trainer
            self._batch_size = self.batch_size

        def set_dataset(self, cols_to_read, target_names):
            """
            set the Parquet dataset

            Parameters
            ----------
            cols_to_read: str
                The list of features names to load
            """

            if isinstance(self.paths_or_dataset, ParquetDataset):
                dataset = self.paths_or_dataset
            dataset = ParquetDataset(
                self.paths_or_dataset,
                cols_to_read,
                seq_features_len_pad_trim=self.max_sequence_length,
                target_names=target_names,
            )
            if self.shuffle and self.shuffle_buffer_size > 0:
                dataset = ShuffleDataset(dataset, buffer_size=self.shuffle_buffer_size)

            self.dataset = dataset

        @classmethod
        def from_schema(
            cls,
            schema,
            paths_or_dataset,
            batch_size,
            max_sequence_length,
            continuous_features=None,
            categorical_features=None,
            targets=None,
            shuffle=False,
            shuffle_buffer_size=0,
            num_workers=1,
            pin_memory=True,
            **kwargs,
        ):
            """
            Instantiates ``PyarrowDataLoader`` from a ``DatasetSchema``.

            Parameters
            ----------
            schema: DatasetSchema
                Dataset schema
            paths_or_dataset: Union[str, Dataset]
                Path to paquet data of Dataset object.
            batch_size: int
                batch size of Dataloader.
            max_sequence_length: int
                The maximum length of list features.
            """

            categorical_features = (
                categorical_features or schema.select_by_tag(Tags.CATEGORICAL).column_names
            )
            continuous_features = (
                continuous_features or schema.select_by_tag(Tags.CONTINUOUS).column_names
            )
            targets = targets or schema.select_by_tag(Tags.TARGET).column_names

            cols_to_read = categorical_features + continuous_features + targets

            return cls(
                paths_or_dataset,
                batch_size,
                max_sequence_length,
                cols_to_read=cols_to_read,
                target_names=targets,
                shuffle=shuffle,
                shuffle_buffer_size=shuffle_buffer_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **kwargs,
            )


class DLDataLoader(PyTorchDataLoader):
    """
    This class is an extension of the torch dataloader.
    It is required to support the FastAI framework.

    Setting the batch size directly to DLDataLoader makes it 3x slower.
    So we set as an alternative attribute and use it within
    T4Rec Trainer during evaluation
    # TODO : run experiments with new merlin-dataloader
    """

    def __init__(self, *args, **kwargs) -> None:
        if "batch_size" in kwargs:
            self._batch_size = kwargs.pop("batch_size")
            super().__init__(*args, **kwargs)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.dataset)


# ---------------------------------------------------------------------------
# ROCm-compatible DataLoader  (no NVIDIA Merlin dependency)
# Uses PyArrow (already a base dependency) for Parquet reading and standard
# PyTorch DataLoader for batching / shuffling / DDP distribution.
# ---------------------------------------------------------------------------
import pyarrow.parquet as pq
from torch.utils.data.distributed import DistributedSampler


class _ROCmParquetDataset(Dataset):
    """Torch map-style Dataset backed by Parquet files, read via PyArrow."""

    def __init__(self, paths, cols_to_read, target_names, list_cols, max_seq_len):
        import pandas as pd

        self.target_names = set(target_names)
        self.list_cols = set(list_cols)
        self.max_seq_len = max_seq_len
        self.cols_to_read = cols_to_read

        # Read all parquet files into a single pandas DataFrame
        frames = []
        for p in paths:
            table = pq.read_table(p, columns=cols_to_read if cols_to_read else None)
            frames.append(table.to_pandas())
        self.data = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_cols = [c for c in self.cols_to_read if c not in self.target_names]
        inputs = {col: self._to_tensor(row[col], col) for col in input_cols}
        targets = {col: self._to_tensor(row[col], col) for col in self.target_names}
        return inputs, targets

    def _to_tensor(self, value, col_name):
        """Convert a single cell value to a torch Tensor, padding sequences."""
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value)
            # Trim to max_seq_len
            arr = arr[: self.max_seq_len]
            # Pad if shorter
            if len(arr) < self.max_seq_len:
                pad = np.zeros(self.max_seq_len, dtype=arr.dtype)
                pad[: len(arr)] = arr
                arr = pad
            if np.issubdtype(arr.dtype, np.floating):
                return torch.tensor(arr, dtype=torch.float32)
            return torch.tensor(arr, dtype=torch.long)
        elif isinstance(value, (int, np.integer)):
            return torch.tensor(value, dtype=torch.long)
        elif isinstance(value, (float, np.floating)):
            return torch.tensor(value, dtype=torch.float32)
        else:
            return torch.tensor(value)


class _HipDFParquetDataset(Dataset):
    """
    GPU-accelerated Dataset using hipDF (cuDF on ROCm) for Parquet reading.

    When hipDF and CuPy-ROCm are installed, this reads data directly into
    GPU memory and converts to PyTorch tensors via DLPack (zero-copy when
    possible).  Falls back gracefully if hipDF is not available -- callers
    should check ``dependencies.is_gpu_dataloader_available()`` first.
    """

    def __init__(self, paths, cols_to_read, target_names, list_cols, max_seq_len, device=0):
        import cudf  # hipDF provides this on ROCm

        self.target_names = set(target_names)
        self.list_cols = set(list_cols)
        self.max_seq_len = max_seq_len
        self.cols_to_read = cols_to_read
        self.device = device

        frames = [cudf.read_parquet(p, columns=cols_to_read if cols_to_read else None)
                  for p in paths]
        self.data = cudf.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Access each column individually; cudf.DataFrame.iloc[row] fails on
        # ListDtype / StructDtype columns because it cannot find a common type
        # when concatenating heterogeneous series into a single row.
        input_cols = [c for c in self.cols_to_read if c not in self.target_names]
        inputs = {col: self._to_tensor(self.data[col].iloc[index], col)
                  for col in input_cols}
        targets = {col: self._to_tensor(self.data[col].iloc[index], col)
                   for col in self.target_names}
        return inputs, targets

    def _to_tensor(self, value, col_name):
        """Convert a cuDF cell to a torch Tensor, with optional zero-copy via DLPack."""
        import cupy as cp

        if col_name in self.list_cols:
            # List / sequence column -- convert through cupy for GPU-resident data
            if hasattr(value, "to_arrow"):
                arr = cp.array(value.to_arrow().as_py())
            elif hasattr(value, "values_host"):
                arr = cp.array(value.values_host)
            else:
                arr = cp.array(value if isinstance(value, (list, np.ndarray)) else [value])

            arr = arr[: self.max_seq_len]
            if len(arr) < self.max_seq_len:
                padded = cp.zeros(self.max_seq_len, dtype=arr.dtype)
                padded[: len(arr)] = arr
                arr = padded

            # DLPack zero-copy: cupy -> torch
            return torch.from_dlpack(arr)
        else:
            # Scalar value -- small, just create tensor directly
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, np.integer)):
                return torch.tensor(value, dtype=torch.long)
            elif isinstance(value, (float, np.floating)):
                return torch.tensor(value, dtype=torch.float32)
            return torch.tensor(value)


class _GPUBatchIterator:
    """GPU-batch-level iterator that pre-converts a pandas DataFrame to
    GPU-resident PyTorch tensors at init time, then yields
    ``(inputs_dict, targets_dict)`` batches using pure tensor indexing.

    This implementation requires **only** PyArrow, pandas, NumPy, and
    PyTorch -- no ``cudf`` or ``cupy`` dependency.  Data flows through:

        Parquet  --(PyArrow)--> pandas  --(numpy)--> torch GPU tensors

    Scalar columns are transferred as contiguous 1-D tensors.
    List columns are flattened, padded / trimmed to ``max_seq_len``, and
    stored as 2-D tensors ``(num_rows, max_seq_len)`` on GPU.
    """

    def __init__(
        self, data, cols_to_read, target_names, list_cols, max_seq_len,
        batch_size, shuffle=False, drop_last=False, device=0,
        global_size=None, global_rank=None,
    ):
        self.target_names = set(target_names)
        self.list_cols = set(list_cols)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
        self.max_seq_len = max_seq_len
        self.cols_to_read = cols_to_read

        torch_dev = torch.device(f"cuda:{device}")

        # -- one-time bulk conversion: pandas columns -> GPU tensors -----------
        self._tensors = {}
        for col in cols_to_read:
            series = data[col]
            if col in self.list_cols:
                self._tensors[col] = self._prep_list(series, torch_dev)
            else:
                self._tensors[col] = self._prep_scalar(series, torch_dev)

        self._num_rows = len(data)

        # -- optional DDP sharding (slice tensors to this rank) ----------------
        if global_size is not None and global_size > 1:
            rank = global_rank or 0
            idx = torch.arange(rank, self._num_rows, global_size, device=torch_dev)
            for col in list(self._tensors):
                self._tensors[col] = self._tensors[col][idx]
            self._num_rows = len(idx)

        # Dummy dataset attribute so Trainer.evaluation_loop doesn't crash
        self.dataset = self

    # ------------------------------------------------------------------
    # Column pre-processing (called once at init)
    # ------------------------------------------------------------------
    @staticmethod
    def _prep_scalar(series, torch_dev):
        """Scalar column -> 1-D GPU tensor (copies from CPU to GPU)."""
        np_arr = np.asarray(series)
        if np_arr.dtype.kind in ("O", "U", "S"):
            # Object / string columns -- try numeric conversion
            try:
                np_arr = np_arr.astype(np.float64)
            except (ValueError, TypeError):
                np_arr = np.arange(len(np_arr), dtype=np.int64)
        if np_arr.dtype.kind == "f":
            return torch.tensor(
                np_arr.astype(np.float32, copy=False),
                dtype=torch.float32,
                device=torch_dev,
            )
        return torch.tensor(
            np_arr.astype(np.int64, copy=False),
            dtype=torch.long,
            device=torch_dev,
        )

    def _prep_list(self, series, torch_dev):
        """List/sequence column -> 2-D GPU tensor ``(num_rows, max_seq_len)``.

        Flattens all per-row lists into a single numpy array, computes
        offsets, transfers to GPU, then scatters into a zero-padded matrix.
        """
        n = len(series)

        # Flatten lists + compute offsets on CPU (cheap metadata work)
        flat_parts = []
        offsets = [0]
        for lst in series:
            if isinstance(lst, (list, np.ndarray)):
                arr = np.asarray(lst)
            elif lst is None or (isinstance(lst, float) and np.isnan(lst)):
                arr = np.array([], dtype=np.int64)
            else:
                arr = np.array([lst])
            flat_parts.append(arr)
            offsets.append(offsets[-1] + len(arr))

        if flat_parts and sum(len(p) for p in flat_parts) > 0:
            all_vals = np.concatenate(flat_parts)
        else:
            all_vals = np.array([], dtype=np.int64)

        # Determine output dtype
        if all_vals.dtype.kind == "f":
            torch_dtype = torch.float32
            all_vals = all_vals.astype(np.float32, copy=False)
        else:
            torch_dtype = torch.long
            all_vals = all_vals.astype(np.int64, copy=False)

        # Allocate zero-padded output on GPU
        out = torch.zeros((n, self.max_seq_len), dtype=torch_dtype, device=torch_dev)

        # Transfer flat values to GPU
        if len(all_vals) > 0:
            vals_gpu = torch.tensor(all_vals, dtype=torch_dtype, device=torch_dev)

            # Build scatter indices on CPU (cheap -- only offset metadata)
            row_idx, col_idx, src_idx = [], [], []
            for i in range(n):
                s, e = offsets[i], offsets[i + 1]
                length = min(e - s, self.max_seq_len)
                if length > 0:
                    row_idx.extend([i] * length)
                    col_idx.extend(range(length))
                    src_idx.extend(range(s, s + length))

            if row_idx:
                row_t = torch.tensor(row_idx, dtype=torch.long, device=torch_dev)
                col_t = torch.tensor(col_idx, dtype=torch.long, device=torch_dev)
                src_t = torch.tensor(src_idx, dtype=torch.long, device=torch_dev)
                out[row_t, col_t] = vals_gpu[src_t]

        return out

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __len__(self):
        if self.drop_last:
            return self._num_rows // self.batch_size
        return (self._num_rows + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        dev = f"cuda:{self.device}"
        if self.shuffle:
            perm = torch.randperm(self._num_rows, device=dev)
        else:
            perm = torch.arange(self._num_rows, device=dev)

        for start in range(0, self._num_rows, self.batch_size):
            end = min(start + self.batch_size, self._num_rows)
            if self.drop_last and (end - start) < self.batch_size:
                break
            idx = perm[start:end]
            inputs, targets = {}, {}
            for col in self.cols_to_read:
                t = self._tensors[col][idx]
                if col in self.target_names:
                    targets[col] = t
                else:
                    inputs[col] = t
            yield inputs, targets


# Keep the old name as an alias for backwards compatibility
_HipDFBatchIterator = _GPUBatchIterator


@dataloader_registry.register_with_multiple_names("rocm_dataloader", "rocm")
class ROCmDataLoader(T4RecDataLoader):
    """
    ROCm-compatible data loader using PyArrow for Parquet reading and
    standard PyTorch DataLoader for batching.  Drop-in replacement for
    MerlinDataLoader that does **not** depend on NVIDIA Merlin dataloader.

    It produces ``(inputs_dict, targets_dict)`` batches identical in format
    to MerlinDataLoader, so the T4Rec Trainer works without modification.

    When hipDF (cudf on ROCm) and CuPy-ROCm are installed, activates a
    GPU-accelerated path that pre-converts all data to GPU tensors once,
    then yields batches via pure tensor indexing (no per-row cudf overhead).

    Parameters
    ----------
    paths_or_dataset : Union[str, list]
        Path(s) to Parquet data files or glob patterns.
    batch_size : int
        Batch size.
    max_sequence_length : int, optional
        Maximum length for padding / trimming list (sequence) features.
    cats, conts, labels, lists : list of str, optional
        Column names by type (categorical, continuous, target, list/sequence).
    shuffle : bool
        Whether to shuffle data each epoch.
    device : int, optional
        GPU device id (unused for data placement; tensors are moved by the
        HF Trainer's ``_prepare_inputs``).
    global_size, global_rank : int, optional
        DDP world size and rank -- a ``DistributedSampler`` is used when set.
    num_workers : int
        Number of PyTorch DataLoader workers.  Default 2.
    pin_memory : bool
        Pin CPU memory for faster GPU transfer.  Default True.
    """

    def __init__(
        self,
        paths_or_dataset,
        batch_size,
        max_sequence_length=None,
        conts=None,
        cats=None,
        labels=None,
        lists=None,
        collate_fn=None,
        shuffle=False,
        device=None,
        global_size=None,
        global_rank=None,
        drop_last=False,
        schema=None,
        num_workers=2,
        pin_memory=True,
        **kwargs,
    ):
        T4RecDataLoader.__init__(self)
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.drop_last = drop_last
        self.schema = schema
        self._device_id = device

        # --- resolve file paths ------------------------------------------------
        paths = self._resolve_paths(paths_or_dataset)

        # --- column lists ------------------------------------------------------
        cats = cats or []
        conts = conts or []
        labels_list = [labels] if isinstance(labels, str) else (labels or [])
        lists = lists or []
        cols_to_read = list(dict.fromkeys(cats + conts + labels_list + lists))  # unique, ordered

        # --- build backend -----------------------------------------------------
        # Prefer GPU batch iterator when a CUDA/ROCm device is available.
        # Priority: cudf (GPU-resident read) > PyArrow+pandas (CPU read, GPU tensors)
        if device is not None and torch.cuda.is_available():
            try:
                import pandas as pd

                # Try cudf first (hipDF on ROCm) for GPU-resident parquet read
                try:
                    import cudf
                    frames = [cudf.read_parquet(p, columns=cols_to_read or None)
                              for p in paths]
                    data = cudf.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
                    # cudf DataFrames expose .to_pandas(); convert for _GPUBatchIterator
                    data = data.to_pandas()
                    logger.info("ROCmDataLoader: read parquet via cudf (GPU)")
                except ImportError:
                    # cudf not available -- fall back to PyArrow (CPU read)
                    frames = [pq.read_table(p, columns=cols_to_read or None).to_pandas()
                              for p in paths]
                    data = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
                    logger.info("ROCmDataLoader: read parquet via PyArrow (CPU)")

                self._loader = _GPUBatchIterator(
                    data,
                    cols_to_read,
                    labels_list,
                    lists,
                    max_sequence_length or 20,
                    batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    device=device if isinstance(device, int) else 0,
                    global_size=global_size,
                    global_rank=global_rank,
                )
                del data  # free DataFrame; tensors live on GPU now
                self._batch_size = batch_size
                self._sampler = None
                logger.info("ROCmDataLoader: using GPU batch iterator (device=%s)", device)
                return  # done -- skip CPU path below
            except Exception as exc:
                logger.warning(
                    "ROCmDataLoader: GPU batch iterator failed (%s), "
                    "falling back to PyArrow CPU DataLoader",
                    exc,
                )

        # --- CPU / PyArrow fallback path ---------------------------------------
        dataset = _ROCmParquetDataset(
            paths,
            cols_to_read,
            labels_list,
            lists,
            max_sequence_length or 20,
        )

        # --- distributed sampler for DDP ---------------------------------------
        sampler = None
        if global_size is not None and global_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=global_size,
                rank=global_rank or 0,
                shuffle=shuffle,
            )
            shuffle = False  # sampler handles shuffling

        # --- collate: stack dicts of tensors -----------------------------------
        def _dict_collate(batch):
            inputs_batch = {}
            targets_batch = {}
            for key in batch[0][0]:
                inputs_batch[key] = torch.stack([b[0][key] for b in batch])
            for key in batch[0][1]:
                targets_batch[key] = torch.stack([b[1][key] for b in batch])
            return inputs_batch, targets_batch

        self._loader = PyTorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=_dict_collate,
        )
        # _batch_size is read directly by the Trainer (num_examples)
        self._batch_size = batch_size
        self._sampler = sampler

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_paths(paths_or_dataset):
        """Turn a path string (possibly with globs) into a list of Parquet files."""
        if isinstance(paths_or_dataset, (list, tuple)):
            result = []
            for p in paths_or_dataset:
                result.extend(sorted(glob_module.glob(str(p))))
            return result if result else [str(p) for p in paths_or_dataset]
        path_str = str(paths_or_dataset)
        files = sorted(glob_module.glob(path_str))
        if not files:
            files = sorted(glob_module.glob(path_str + "/*.parquet"))
        if not files:
            files = [path_str]
        return files

    # ------------------------------------------------------------------
    # from_schema  (called by Trainer.get_train_dataloader et al.)
    # ------------------------------------------------------------------
    @classmethod
    def from_schema(
        cls,
        schema,
        paths_or_dataset,
        batch_size,
        max_sequence_length=None,
        continuous_features=None,
        categorical_features=None,
        list_features=None,
        targets=None,
        collate_fn=None,
        shuffle=True,
        **kwargs,
    ):
        """Instantiate ``ROCmDataLoader`` from a ``DatasetSchema``."""
        if Tags is not None and schema is not None:
            cat_feats = (
                categorical_features or schema.select_by_tag(Tags.CATEGORICAL).column_names
            )
            cont_feats = (
                continuous_features or schema.select_by_tag(Tags.CONTINUOUS).column_names
            )
            tgt_feats = targets or schema.select_by_tag(Tags.TARGET).column_names
            lst_feats = list_features or schema.select_by_tag(Tags.LIST).column_names
        else:
            cat_feats = categorical_features or []
            cont_feats = continuous_features or []
            tgt_feats = targets or []
            lst_feats = list_features or []

        return cls(
            paths_or_dataset,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            cats=cat_feats,
            conts=cont_feats,
            labels=tgt_feats,
            lists=lst_feats,
            shuffle=shuffle,
            schema=schema,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Iterator / len / device  (Trainer interface)
    # ------------------------------------------------------------------
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def dataset(self):
        """Expose the underlying torch Dataset (needed by Trainer.evaluation_loop)."""
        return self._loader.dataset

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)


# ---------------------------------------------------------------------------
# Original NVIDIA MerlinDataLoader -- only defined when merlin-dataloader
# is installed (i.e. CUDA / NVIDIA systems).
# ---------------------------------------------------------------------------
if _merlin_loader_available:

    @dataloader_registry.register_with_multiple_names(
        "merlin_dataloader", "merlin", "nvtabular_dataloader", "nvtabular"
    )
    class MerlinDataLoader(T4RecDataLoader, DLDataLoader):
        """
        This class extends the [Merlin data loader]
        (https://github.com/NVIDIA-Merlin/dataloader/blob/stable/merlin/dataloader/torch.py).
        The data input requires a merlin.io.Dataset or a path to the data files.
        It also sets the dataset's schema with the necessary properties to prepare the input
        list features as dense tensors (i.e. padded to the specified `max_sequence_length`).
        The dense representation is required by the Transformers4Rec input modules.

        Parameters
        ----------
        paths_or_dataset: Union[str, merlin.io.Dataset]
            The dataset to load.
        batch_size: int
            The size of each batch to supply to the model.
        max_sequence_length: int
            The maximum sequence length to use for padding list columns.
            By default, `0` is used as the padding index.
        cats : List[str], optional
            The list of categorical columns in the dataset.
            By default None.
        conts: List[str], optional
            The list of continuous columns in the dataset.
            By default None.
        labels : List[str], optional
            The list of label columns in the dataset.
            By default None.
        lists : List[str], optional
            The list of sequential columns in the dataset.
            By default None.
        shuffle : bool, optional
            Enable/disable shuffling of dataset.
            By default False.
        parts_per_chunk : int, optional
            The number of partitions from the iterator, an Merlin Dataset,
            to concatenate into a "chunk". By default 1.
        device : int, optional
            The device id of the selected GPU
            By default None.
        drop_last: bool, optional
            Whether or not to drop the last batch in an epoch. This is useful when you need to
            guarantee that each batch contains exactly `batch_size` rows - since the last batch
            will usually contain fewer rows.
        seed_fn: callable
            Function used to initialize random state
        parts_per_chunk: int
            Number of dataset partitions with size dictated by `buffer_size`
            to load and concatenate asynchronously. More partitions leads to
            better epoch-level randomness but can negatively impact throughput
        global_size: int, optional
            When doing distributed training, this indicates the number of total processes that are
            training the model.
        global_rank: int, optional
            When doing distributed training, this indicates the local rank for the current process.
        schema: Schema, optional
             The `Schema` with the input features.
        reader_kwargs:
            Extra arguments to pass to the merlin.io.Dataset object, when the path to data files
            is provided in `paths_or_dataset` argument.
        row_groups_per_part: bool, optional
            If true, preserve the group partitions when loading the dataset from parquet files.
        collate_fn: Callable, optional
            A processing function to collect and prepare the list samples
            (tuple of (input, target) Tensor(s)) returned by the Merlin DataLoader.
        transforms: List[merlin.dag.BaseOperator]
            A list of operators that the Merlin dataloader applies on top of the loaded
            batch, which is a tuple of input and target tensors.
        """

        def __init__(
            self,
            paths_or_dataset,
            batch_size,
            max_sequence_length=None,
            conts=None,
            cats=None,
            labels=None,
            lists=None,
            collate_fn=lambda x: x[0],
            engine=None,
            buffer_size=0.1,
            reader_kwargs=None,
            shuffle=False,
            seed_fn=None,
            parts_per_chunk=1,
            device=None,
            global_size=None,
            global_rank=None,
            drop_last=False,
            schema=None,
            row_groups_per_part=True,
            transforms=None,
            **kwargs,
        ):
            T4RecDataLoader.__init__(self)

            self.paths_or_dataset = paths_or_dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.max_sequence_length = max_sequence_length
            self.drop_last = drop_last

            reader_kwargs = reader_kwargs or {}
            reader_kwargs["row_groups_per_part"] = row_groups_per_part
            self.set_dataset(buffer_size, engine, reader_kwargs, schema=schema)

            if (global_rank is not None) and (self.dataset.npartitions < global_size):
                logger.warning(
                    "UserWarning: User is advised to repartition the parquet file before training "
                    "so npartitions>=global_size. Cudf or pandas can be used for repartitioning "
                    "eg. pdf.to_parquet('file.parquet',row_group_size=N_ROWS/NPARTITIONS) "
                    "for pandas or "
                    "gdf.to_parquet('file.parquet',row_group_size_rows=N_ROWS/NPARTITIONS) "
                    "for cudf so that npartitions=nr_rows/row_group_size. Also ensure "
                    "npartitions is divisible by number of GPUs to be used "
                    "(eg. 2 or 4 partitions, if 2 GPUs will be used)."
                )
                self.dataset = self.dataset.repartition(npartitions=global_size)

            if (global_rank is not None) and (self.dataset.npartitions % global_size != 0):
                logger.warning(
                    f"UserWarning: User is advised to set the number of partitions"
                    f" ({self.dataset.npartitions}) divisible by the number of available"
                    f" GPUs ({global_size}). This will divide the work equally among GPUs"
                    " for DDP training and ensure optimal performance."
                )

            self.dataset.schema = self._augment_schema(
                self.dataset.schema,
                cats=cats,
                conts=conts,
                labels=labels,
                lists=lists,
            )

            loader = Loader(
                self.dataset,
                self.batch_size,
                shuffle,
                seed_fn=seed_fn,
                parts_per_chunk=parts_per_chunk,
                device=device,
                global_size=global_size,
                global_rank=global_rank,
                drop_last=drop_last,
                transforms=transforms,
            )
            if max_sequence_length:
                # Apply padding
                output_schema = loader.output_schema
                sparse_feats = [col.name for col in output_schema if Tags.LIST in col.tags]
                sparse_max = {name: max_sequence_length for name in sparse_feats}
                loader = loader.map(self._get_pad_fn(sparse_max))

            DLDataLoader.__init__(
                self,
                loader,
                collate_fn=collate_fn,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
            self.schema = schema
            self.max_sequence_length = max_sequence_length

        @staticmethod
        def _get_pad_fn(padding_lengths):
            def pad_fn(x, y):
                new_x = pad_batch(x, padding_lengths)
                if y is not None and isinstance(y, dict):
                    new_y = pad_batch(y, padding_lengths)
                else:
                    new_y = y
                return new_x, new_y

            return pad_fn

        @staticmethod
        def _augment_schema(
            schema,
            cats=None,
            conts=None,
            labels=None,
            lists=None,
        ):
            cats = cats or []
            conts = conts or []
            labels = labels or []

            schema = schema.select_by_name(conts + cats + labels + lists)

            labels = [labels] if isinstance(labels, str) else labels
            for label in labels:
                schema[label] = schema[label].with_tags(Tags.TARGET)
            for label in cats:
                schema[label] = schema[label].with_tags(Tags.CATEGORICAL)
            for label in conts:
                schema[label] = schema[label].with_tags(Tags.CONTINUOUS)
            for col in lists:
                schema[col] = schema[col].with_tags(Tags.LIST)

            return schema

        def set_dataset(self, buffer_size, engine, reader_kwargs, schema=None):
            dataset = validate_dataset(
                self.paths_or_dataset,
                self.batch_size,
                buffer_size,
                engine,
                reader_kwargs,
            )
            if schema:
                if isinstance(schema, Schema):
                    schema = to_core_schema(schema)
                dataset.schema = schema
            self.dataset = dataset

        @classmethod
        def from_schema(
            cls,
            schema,
            paths_or_dataset,
            batch_size,
            max_sequence_length=None,
            continuous_features=None,
            categorical_features=None,
            list_features=None,
            targets=None,
            collate_fn=lambda x: x[0],
            shuffle=True,
            buffer_size=0.06,
            parts_per_chunk=1,
            transforms=None,
            **kwargs,
        ):
            """Instantiate ``MerlinDataLoader`` from a ``DatasetSchema``."""
            categorical_features = (
                categorical_features or schema.select_by_tag(Tags.CATEGORICAL).column_names
            )
            continuous_features = (
                continuous_features or schema.select_by_tag(Tags.CONTINUOUS).column_names
            )
            targets = targets or schema.select_by_tag(Tags.TARGET).column_names
            list_features = list_features or schema.select_by_tag(Tags.LIST).column_names
            schema = schema.select_by_name(
                categorical_features + continuous_features + targets + list_features
            )
            loader = cls(
                paths_or_dataset,
                batch_size=batch_size,
                labels=targets,
                max_sequence_length=max_sequence_length,
                cats=categorical_features,
                conts=continuous_features,
                lists=list_features,
                collate_fn=collate_fn,
                engine="parquet",
                shuffle=shuffle,
                buffer_size=buffer_size,
                parts_per_chunk=parts_per_chunk,
                schema=schema,
                transforms=transforms,
                **kwargs,
            )

            return loader

        @property
        def output_schema(self):
            return self.dataset.output_schema


class ParquetDataset(Dataset):
    def __init__(self, parquet_file, cols_to_read, target_names, seq_features_len_pad_trim):
        self.cols_to_read = cols_to_read
        self.target_names = target_names
        self.data = pq.ParquetDataset(parquet_file).read(columns=self.cols_to_read).to_pandas()
        self.seq_features_len_pad_trim = seq_features_len_pad_trim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        df = self.data.loc[index]
        input_features = list(set(self.cols_to_read).difference(self.target_names))
        inputs = {col: self.pad_seq_column_if_needed(df[col]) for col in input_features}
        targets = {col: self.pad_seq_column_if_needed(df[col]) for col in self.target_names}
        return inputs, targets

    def pad_seq_column_if_needed(self, values):
        if type(values) is np.ndarray:
            values = values[: self.seq_features_len_pad_trim]
            if len(values) < self.seq_features_len_pad_trim:
                placeholder = np.zeros(self.seq_features_len_pad_trim, dtype=values.dtype)
                placeholder[: len(values)] = values
                values = placeholder
            if isinstance(values[0], np.floating) and values.dtype is not np.float32:
                values = values.astype(np.float32)
            if isinstance(values[0], np.integer) and values.dtype is not np.int64:
                values = values.astype(np.int64)
        return values


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        logger.info("[SHUFFLE] INITIALIZING BUFFER_SIZE: {}".format(self.buffer_size))

        raise StopIteration()
        # TODO define The shuffle method for pyarrow dataloader

    def __len__(self):
        return len(self.dataset)


def to_core_schema(t4rec_schema):
    if TensorflowMetadata is None:
        raise ImportError(
            "merlin-core is required for schema conversion. "
            "Install it with: pip install merlin-core"
        )
    return TensorflowMetadata.from_json(t4rec_schema.to_json()).to_merlin_schema()
