"""Preload Windows NVIDIA runtime libraries without importing PyTorch."""

import ctypes
import os
import site
import sys
from pathlib import Path

# Keep directory/library handles alive while CUDA loads engines lazily.
_CUDA_DLL_DIRECTORIES = []
_CUDA_DLL_HANDLES = []


def _preload_windows_cuda():
    """Load NVIDIA pip-wheel CUDA 12 / cuDNN 9 DLLs without importing torch."""
    if os.name != "nt" or _CUDA_DLL_HANDLES:
        return
    loaded_torch = sys.modules.get("torch")
    torch_cuda = getattr(getattr(loaded_torch, "version", None), "cuda", None)
    if torch_cuda and not torch_cuda.startswith("12."):
        raise RuntimeError(
            f"PyTorch CUDA {torch_cuda} is already loaded in this process, but this "
            "ONNX Runtime CUDA provider requires CUDA 12 / cuDNN 9. Run ONNX GPU "
            "inference in a separate process to avoid conflicting CUDA DLLs."
        )
    roots = set(site.getsitepackages())
    roots.add(site.getusersitepackages())
    roots.update(path for path in sys.path if path)
    directories = {}
    for root in roots:
        nvidia_root = Path(root) / "nvidia"
        if nvidia_root.is_dir():
            for directory in nvidia_root.glob("*/bin"):
                if directory.is_dir():
                    resolved = directory.resolve()
                    directories[str(resolved).lower()] = resolved
    if not directories:
        # Also permit a correctly installed system CUDA runtime on PATH.
        return
    for directory in sorted(directories.values()):
        _CUDA_DLL_DIRECTORIES.append(os.add_dll_directory(str(directory)))
    files = {}
    for directory in directories.values():
        for path in directory.glob("*.dll"):
            files[path.name.lower()] = path.resolve()
    # Dependency order follows ORT's official preload_dlls implementation;
    # CUDA runtime is loaded first, before the libraries that use it.
    names = [
        "cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_engines_precompiled64_9.dll", "cudnn_heuristic64_9.dll",
        "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_adv64_9.dll",
        "cudnn_graph64_9.dll", "cudnn64_9.dll",
    ]
    required = {name.lower() for name in names}
    missing = sorted(required - set(files))
    if missing:
        raise RuntimeError(
            "Incomplete NVIDIA CUDA 12 runtime packages; missing: " + ", ".join(missing)
            + ". Install nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, "
            "nvidia-cufft-cu12 and nvidia-cudnn-cu12."
        )
    try:
        for name in names:
            path = files.get(name.lower())
            if path is not None:
                _CUDA_DLL_HANDLES.append(ctypes.WinDLL(str(path)))
    except OSError as exc:
        _CUDA_DLL_HANDLES.clear()
        raise RuntimeError(f"Could not preload NVIDIA CUDA runtime DLL {path}: {exc}") from exc
