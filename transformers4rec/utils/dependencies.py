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
def is_gpu_dataloader_available() -> bool:
    """Check if the GPU-accelerated dataloader path is available.

    The GPU batch iterator only needs PyTorch with CUDA/ROCm support.
    It reads parquet via PyArrow into pandas, then transfers data to
    GPU tensors using plain PyTorch -- no cudf or cupy required.

    If cudf/cupy *are* installed (hipDF on ROCm, or RAPIDS on CUDA),
    they will be used opportunistically for GPU-resident parquet reads,
    but they are not required.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def is_hipdf_available() -> bool:
    """Check if hipDF (cudf) and cupy are both importable.

    When both are present the dataloader can read Parquet directly into
    GPU memory and convert columns to PyTorch tensors via DLPack
    (zero-copy), bypassing the CPU entirely.

    hipDF exposes itself as ``cudf`` at import time, so we test for
    ``import cudf`` plus ``import cupy``.

    Set the environment variable ``T4REC_DISABLE_HIPDF=1`` to force-skip
    the hipDF path even when the packages are installed (useful for A/B
    testing against the PyArrow+PyTorch GPU batch iterator).
    """
    import os

    if os.environ.get("T4REC_DISABLE_HIPDF", "0") == "1":
        return False
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def is_pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def is_merlin_dataloader_available() -> bool:
    try:
        import merlin.dataloader  # noqa: F401
    except ImportError:
        return False
    return True


def is_rocm_available() -> bool:
    """Check if ROCm is available via PyTorch's HIP backend."""
    try:
        import torch

        return torch.cuda.is_available() and torch.version.hip is not None
    except (ImportError, AttributeError):
        return False
