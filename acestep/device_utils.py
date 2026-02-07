"""
Device Utilities Module
Shared utility functions for cross-backend device management (CUDA, MPS, XPU, CPU).
"""

import torch


def get_device_type():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device_type(device=None):
    """Resolve device to a string type."""
    if device is None:
        return get_device_type()
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0]  # handle "cuda:0" etc.


def empty_cache(device=None):
    """Clear GPU cache for the appropriate backend."""
    device = _resolve_device_type(device)
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def synchronize(device=None):
    """Synchronize the appropriate backend."""
    device = _resolve_device_type(device)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def memory_allocated(device=None):
    """Get memory allocated on the appropriate backend."""
    device = _resolve_device_type(device)
    if device == "cuda":
        return torch.cuda.memory_allocated()
    elif device == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def max_memory_allocated(device=None):
    """Get max memory allocated on the appropriate backend."""
    device = _resolve_device_type(device)
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    return 0  # MPS doesn't track max


def manual_seed(seed):
    """Set seed for all backends."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def supports_bfloat16(device):
    """Check if device supports bfloat16."""
    if isinstance(device, torch.device):
        device = device.type
    return device in ("cuda", "xpu", "mps")


def get_dtype(device):
    """Get appropriate dtype for device."""
    if supports_bfloat16(device):
        return torch.bfloat16
    return torch.float32


def is_mlx_available():
    """Check if MLX is available (Apple Silicon with mlx installed)."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def mlx_clear_cache():
    """Clear the MLX Metal memory cache if available."""
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass
