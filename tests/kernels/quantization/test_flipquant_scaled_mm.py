# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlipQuant (MIXFP4) GEMM: NVFP4 operands where each 256x64 weight tile is
decoded as E2M1 or E0M3 according to a 1-bit map (arXiv 2611.18769)."""

import pytest
import torch
from nvfp4_utils import break_fp4_bytes

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

capability = current_platform.get_device_capability()
if (
    not current_platform.is_cuda()
    or capability is None
    or not ops.cutlass_scaled_mm_supports_flipquant(capability.to_int())
):
    pytest.skip(
        reason="FlipQuant GEMM requires SM100.",
        allow_module_level=True,
    )

MAP_N, MAP_K = 256, 64
DTYPES = [torch.float16, torch.bfloat16]
# m covers the M16, M256 and large-M configs; n = 160 and 384 end in a
# partial map row; k = 2080 spans two map words and ends in half a K64 block.
SHAPES = [
    (1, 256, 256),
    (16, 384, 512),
    (17, 160, 2080),
    (256, 512, 1024),
    (257, 384, 2080),
    (1024, 1024, 4096),
]
DEVICE = "cuda:0"


def pack_tile_map(e0m3: torch.Tensor) -> torch.Tensor:
    """Bool `[ceil(N/256), ceil(K/64)]` -> int32 `[ceil(N/256), ceil(K/2048)]`."""
    rows, blocks = e0m3.shape
    words = (blocks + 31) // 32
    bits = torch.zeros(rows, words * 32, dtype=torch.int64, device=e0m3.device)
    bits[:, :blocks] = e0m3.long()
    weights = 1 << torch.arange(32, dtype=torch.int64, device=e0m3.device)
    packed = (bits.view(rows, words, 32) * weights).sum(-1)
    return torch.where(packed >= 1 << 31, packed - (1 << 32), packed).to(torch.int32)


def decode_e0m3(codes: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude nibbles with magnitudes 0..7, low nibble first."""
    nibbles = torch.stack((codes & 0xF, codes >> 4), dim=-1).flatten(-2)
    mag = (nibbles & 0x7).float()
    return torch.where((nibbles & 0x8).bool(), -mag, mag)


def random_operand(rows: int, k: int):
    codes = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, device=DEVICE)
    levels = torch.tensor([0.125, 0.25, 0.5, 1.0], device=DEVICE)
    scales = levels[torch.randint(0, 4, (rows, k // 16), device=DEVICE)]
    return codes, scales


def reference(a, sfa, b, sfb, e0m3_tiles, alpha):
    n, k = b.shape[0], b.shape[1] * 2
    a_val = break_fp4_bytes(a, torch.float32) * sfa.repeat_interleave(16, -1)
    e0m3 = e0m3_tiles.repeat_interleave(MAP_N, 0)[:n]
    e0m3 = e0m3.repeat_interleave(MAP_K, 1)[:, :k]
    b_val = torch.where(e0m3, decode_e0m3(b), break_fp4_bytes(b, torch.float32))
    b_val = b_val * sfb.repeat_interleave(16, -1)
    return alpha * (a_val @ b_val.T)


def run(a, sfa, b, sfb, tile_map, alpha, dtype):
    return ops.cutlass_scaled_flipquant_mm(
        a,
        b,
        swizzle_blockscale(sfa.to(torch.float8_e4m3fn)),
        swizzle_blockscale(sfb.to(torch.float8_e4m3fn)),
        tile_map,
        alpha,
        dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fill", ["mixed", "e0m3"])
@torch.inference_mode()
def test_flipquant_gemm_decodes_per_tile_format(
    dtype: torch.dtype, shape: tuple[int, int, int], fill: str
) -> None:
    """Each 256x64 weight tile must decode in the format its map bit names."""
    set_random_seed(0)
    m, n, k = shape
    a, sfa = random_operand(m, k)
    b, sfb = random_operand(n, k)
    tiles = ((n + MAP_N - 1) // MAP_N, (k + MAP_K - 1) // MAP_K)
    if fill == "mixed":
        e0m3_tiles = torch.rand(tiles, device=DEVICE) < 0.5
    else:
        e0m3_tiles = torch.ones(tiles, dtype=torch.bool, device=DEVICE)
    alpha = torch.tensor([1.0 / 16], dtype=torch.float32, device=DEVICE)

    out = run(a, sfa, b, sfb, pack_tile_map(e0m3_tiles), alpha, dtype)

    expected = reference(a, sfa, b, sfb, e0m3_tiles, alpha)
    torch.testing.assert_close(
        out.float(), expected, rtol=1e-2, atol=1e-2 * expected.abs().max().item()
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_flipquant_gemm_all_e2m1_matches_nvfp4(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    """An all-E2M1 map must reproduce the NVFP4 GEMM bit for bit."""
    set_random_seed(0)
    m, n, k = shape
    a, sfa = random_operand(m, k)
    b, sfb = random_operand(n, k)
    sfa_swz = swizzle_blockscale(sfa.to(torch.float8_e4m3fn))
    sfb_swz = swizzle_blockscale(sfb.to(torch.float8_e4m3fn))
    tile_map = torch.zeros(
        (n + MAP_N - 1) // MAP_N,
        ((k + MAP_K - 1) // MAP_K + 31) // 32,
        dtype=torch.int32,
        device=DEVICE,
    )
    alpha = torch.tensor([1.0 / 16], dtype=torch.float32, device=DEVICE)

    out = ops.cutlass_scaled_flipquant_mm(
        a, b, sfa_swz, sfb_swz, tile_map, alpha, dtype
    )

    expected = ops.cutlass_scaled_fp4_mm(a, b, sfa_swz, sfb_swz, alpha, dtype)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_flipquant_gemm_op_schema_and_map_shape_check() -> None:
    m, n, k = 128, 256, 256
    a, sfa = random_operand(m, k)
    b, sfb = random_operand(n, k)
    sfa_swz = swizzle_blockscale(sfa.to(torch.float8_e4m3fn))
    sfb_swz = swizzle_blockscale(sfb.to(torch.float8_e4m3fn))
    alpha = torch.ones(1, dtype=torch.float32, device=DEVICE)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=DEVICE)
    tile_map = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)

    opcheck(
        torch.ops._C.cutlass_scaled_flipquant_mm,
        (out, a, b, sfa_swz, sfb_swz, tile_map, alpha),
    )

    with pytest.raises(RuntimeError, match="tile_map must have shape"):
        torch.ops._C.cutlass_scaled_flipquant_mm(
            out, a, b, sfa_swz, sfb_swz, tile_map[:, :0], alpha
        )
