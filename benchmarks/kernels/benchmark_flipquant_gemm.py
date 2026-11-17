# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlipQuant (MIXFP4) GEMM vs the NVFP4 GEMM it extends, on the same operands.

Both GEMMs take pre-quantized activations, so the difference is the cost of
the per-tile weight format.
"""

import argparse
import copy
import itertools
import os

import torch
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton

if not ops.cutlass_scaled_mm_supports_flipquant(
    current_platform.get_device_capability().to_int()
):
    raise RuntimeError("FlipQuant GEMM requires compute capability 10.0")

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
PROVIDERS = ["torch-bf16", "nvfp4", "flipquant"]


def random_tile_map(n: int, k: int, e0m3_fraction: float) -> torch.Tensor:
    rows, words = (n + 255) // 256, ((k + 63) // 64 + 31) // 32
    bits = torch.rand(rows, words, 32, device="cuda") < e0m3_fraction
    weights = 1 << torch.arange(32, dtype=torch.int64, device="cuda")
    packed = (bits.long() * weights).sum(-1)
    return torch.where(packed >= 1 << 31, packed - (1 << 32), packed).to(torch.int32)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        x_log=False,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDERS,
        ylabel="TFLOP/s (larger is better)",
        plot_name="BF16 vs NVFP4 vs FlipQuant GEMMs",
        args={},
    )
)
def benchmark(batch_size, provider, N, K, e0m3_fraction):
    M = batch_size
    dtype = torch.bfloat16
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((N, K), device="cuda", dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-bf16":
        fn = lambda: torch.nn.functional.linear(a, b)
    else:
        a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a.abs().max().float()
        b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b.abs().max().float()
        alpha = 1.0 / (a_gs * b_gs)
        a_fp4, a_sf = ops.scaled_fp4_quant(a, a_gs)
        b_fp4, b_sf = ops.scaled_fp4_quant(b, b_gs)
        if provider == "nvfp4":
            fn = lambda: ops.cutlass_scaled_fp4_mm(
                a_fp4, b_fp4, a_sf, b_sf, alpha, dtype
            )
        else:
            tile_map = random_tile_map(N, K, e0m3_fraction)
            fn = lambda: ops.cutlass_scaled_flipquant_mm(
                a_fp4, b_fp4, a_sf, b_sf, tile_map, alpha, dtype
            )

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


def prepare_shapes(args):
    out = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            KN.append(model)
            out.append(KN)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.1-8B-Instruct"],
        choices=list(WEIGHT_SHAPES.keys()),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--e0m3-fraction",
        type=float,
        default=0.5,
        help="Fraction of 256x64 weight tiles decoded as E0M3.",
    )
    args = parser.parse_args()

    for K, N, model in prepare_shapes(args):
        print(f"{model}, N={N} K={K}, BF16 vs NVFP4 vs FlipQuant GEMMs TFLOP/s:")
        save_dir = f"bench_flipquant_res_n{N}_k{K}"
        os.makedirs(save_dir, exist_ok=True)
        benchmark.run(
            print_data=True,
            show_plots=False,
            save_path=save_dir,
            N=N,
            K=K,
            e0m3_fraction=args.e0m3_fraction,
        )
