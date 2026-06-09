import os
from typing import Optional, Union

import torch


DeviceLike = Optional[Union[str, torch.device]]


def get_accelerator(requested: DeviceLike = None) -> torch.device:
    requested = requested or os.environ.get("PIXAL3D_DEVICE", "auto")
    if str(requested) != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def configure_runtime(device: DeviceLike) -> torch.device:
    device = get_accelerator(device)
    os.environ["PIXAL3D_DEVICE"] = str(device)

    if device.type == "xpu":
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ["ATTN_BACKEND"])
        os.environ.setdefault("SPARSE_CONV_BACKEND", "torch")

    try:
        from pixal3d.modules.attention import config as attn_config
        attn_config.set_backend(os.environ.get("ATTN_BACKEND", attn_config.BACKEND))
    except Exception:
        pass

    try:
        from pixal3d.modules.sparse import config as sparse_config
        sparse_config.set_attn_backend(os.environ.get("SPARSE_ATTN_BACKEND", os.environ.get("ATTN_BACKEND", sparse_config.ATTN)))
        sparse_config.set_conv_backend(os.environ.get("SPARSE_CONV_BACKEND", sparse_config.CONV))
    except Exception:
        pass

    return device


def device_count(device: DeviceLike = None) -> int:
    device = get_accelerator(device)
    if device.type == "cuda":
        return torch.cuda.device_count()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.device_count()
    return 1


def empty_cache(device: DeviceLike = None) -> None:
    device = get_accelerator(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()


def synchronize(device: DeviceLike = None) -> None:
    device = get_accelerator(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)


def manual_seed_all(seed: int, device: DeviceLike = None) -> None:
    torch.manual_seed(seed)
    device = get_accelerator(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(seed)
