import math
import torch
import torch.nn as nn
from .. import SparseTensor


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and padding is None, "The torch sparse fallback only supports submanifold conv3d."

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, *self.kernel_size, in_channels)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in = in_channels
        for k in self.kernel_size:
            fan_in *= k
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)


def _encoded_coords(coords: torch.Tensor, spatial_shape: torch.Size) -> torch.Tensor:
    coords64 = coords.to(torch.int64)
    sx, sy, sz = [int(v) for v in spatial_shape]
    return (((coords64[:, 0] * sx + coords64[:, 1]) * sy + coords64[:, 2]) * sz + coords64[:, 3])


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    coords = x.coords
    feats = x.feats
    spatial_shape = x.spatial_shape
    sorted_codes, sorted_idx = torch.sort(_encoded_coords(coords, spatial_shape))

    out = feats.new_zeros((feats.shape[0], self.out_channels))
    if self.bias is not None:
        out += self.bias

    centers = tuple(k // 2 for k in self.kernel_size)
    for kd in range(self.kernel_size[0]):
        for kh in range(self.kernel_size[1]):
            for kw in range(self.kernel_size[2]):
                offset = torch.tensor(
                    [
                        (kd - centers[0]) * self.dilation[0],
                        (kh - centers[1]) * self.dilation[1],
                        (kw - centers[2]) * self.dilation[2],
                    ],
                    dtype=coords.dtype,
                    device=coords.device,
                )
                neighbor_coords = coords.clone()
                neighbor_coords[:, 1:] += offset
                valid = ((neighbor_coords[:, 1:] >= 0) & (neighbor_coords[:, 1:] < torch.tensor(spatial_shape, dtype=coords.dtype, device=coords.device))).all(dim=1)
                if not valid.any():
                    continue

                neighbor_codes = _encoded_coords(neighbor_coords[valid], spatial_shape)
                pos = torch.searchsorted(sorted_codes, neighbor_codes)
                pos_valid = pos < sorted_codes.numel()
                if not pos_valid.any():
                    continue
                matched = sorted_codes[pos[pos_valid]] == neighbor_codes[pos_valid]
                if not matched.any():
                    continue

                dst = valid.nonzero(as_tuple=False).squeeze(1)[pos_valid][matched]
                src = sorted_idx[pos[pos_valid][matched]]
                weight = self.weight[:, kd, kh, kw, :]
                out[dst] += feats[src] @ weight.t()

    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError("SparseInverseConv3d is not implemented for the torch sparse fallback.")


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError("SparseInverseConv3d is not implemented for the torch sparse fallback.")
