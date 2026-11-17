/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// FlipQuant (MIXFP4) mainloop for SM100 block-scaled NVFP4 GEMMs.
//
// Wraps CUTLASS's `MainloopSm100TmaUmmaWarpSpecializedBlockScaled` collective
// and replaces only `mma()`, whose body is CUTLASS v4.7.1's with one change:
// before each K64 MMA the B operand's format field of the instruction
// descriptor is set from a 1-bit-per-tile map. tcgen05.mma reads that
// descriptor from a register, so the choice costs a register write.
//
// Map layout: int32 [ceil(N / 256), ceil(K / 2048)]; bit j of word w in row r
// covers weight rows [256 r, 256 r + 256) and K64 block 32 w + j.
// Bit 1 = E0M3 (uniform INT4 grid, format enum 0), bit 0 = E2M1 (enum 1).

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

namespace vllm::flipquant {

using namespace cute;

static constexpr int kMapTileN = 256;
static constexpr int kMapTileK = 64;

template <class Base>
struct FlipQuantCollectiveMma : Base {
  using ClusterShape = typename Base::DispatchPolicy::ClusterShape;
  using Base::IsCtaN192;
  using Base::IsCtaN64;
  using Base::IsOverlappingAccum;
  using typename Base::CtaShape_MNK;
  using typename Base::MainloopPipeline;
  using typename Base::MainloopPipelineState;

  static constexpr int CtaN = size<1>(CtaShape_MNK{});
  static_assert(kMapTileN % CtaN == 0,
                "a CTA tile must lie within one 256-wide map row");
  static_assert(size<2>(typename Base::TileShape{}) == 4 * kMapTileK,
                "expects four K64 MMA blocks per K tile");

  struct Arguments : Base::Arguments {
    uint32_t const* tile_map{nullptr};
    int tile_map_stride{0};
  };

  struct Params : Base::Params {
    uint32_t const* tile_map;
    int tile_map_stride;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape, Arguments const& args, void* workspace,
      cutlass::KernelHardwareInfo const& hw_info =
          cutlass::KernelHardwareInfo{}) {
    return {
        Base::to_underlying_arguments(problem_shape, args, workspace, hw_info),
        args.tile_map, args.tile_map_stride};
  }

  CUTLASS_DEVICE
  FlipQuantCollectiveMma(Params const& params, ClusterShape cluster_shape,
                         uint32_t block_rank_in_cluster)
      : Base(params, cluster_shape, block_rank_in_cluster),
        tile_map_(params.tile_map),
        tile_map_stride_(params.tile_map_stride) {}

  template <class AccumulatorPipeline, class FrgEngine, class FrgLayout,
            class MmaParams, class CtaTileCoord>
  CUTLASS_DEVICE auto mma(
      cute::tuple<MainloopPipeline, AccumulatorPipeline> pipelines,
      cute::tuple<MainloopPipelineState,
                  typename AccumulatorPipeline::PipelineState>
          pipeline_states,
      cute::tuple<cute::Tensor<FrgEngine, FrgLayout>> const& accumulators_pair,
      MmaParams const& mma_inputs, CtaTileCoord cta_tile_coord,
      int k_tile_count) {
    static_assert(is_tmem<FrgEngine>::value,
                  "Accumulator must be tmem resident.");
    static_assert(rank(FrgLayout{}) == 3,
                  "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

    auto accumulators = get<0>(accumulators_pair);
    auto [tiled_mma, tCrA, tCrB, tCtSFA, tCtSFB, tiled_copy_s2t_SFA,
          thr_tCsSFA_s2t, thr_tCtSFA_s2t, tiled_copy_s2t_SFB, thr_tCsSFB_s2t,
          thr_tCtSFB_s2t] = mma_inputs;

    auto [mainloop_pipeline, accumulator_pipeline] = pipelines;
    auto [mainloop_pipe_consumer_state, accumulator_pipe_producer_state] =
        pipeline_states;

    auto tCtSFB_mma = [tCtSFB = tCtSFB, cta_tile_coord]() {
      if constexpr (IsCtaN192) {
        auto tCtSFB_tmp = tCtSFB;
        if (size<1>(cta_tile_coord) % 2 == 1) {
          tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + 2;
        }
        return tCtSFB_tmp;
      } else if constexpr (IsCtaN64) {
        auto tCtSFB_tmp = tCtSFB;
        tCtSFB_tmp.data() =
            tCtSFB_tmp.data().get() + (size<1>(cta_tile_coord) % 2) * 2;
        return tCtSFB_tmp;
      } else {
        return tCtSFB;
      }
    }();

    // Four map bits per K tile; the next K tile's bits load while the current
    // one issues. Bit set = E0M3 = format enum 0, so b_format_ = ~bit & 1.
    uint32_t const* map_row =
        tile_map_ + size_t(int(get<1>(cta_tile_coord)) * CtaN / kMapTileN) *
                        size_t(tile_map_stride_);
    int const map_k_tiles = tile_map_stride_ * 8;
    auto load_formats = [map_row, map_k_tiles](int kt) -> uint32_t {
      uint32_t const e0m3 =
          kt < map_k_tiles ? (__ldg(map_row + kt / 8) >> (4 * (kt % 8))) & 0xFu
                           : 0u;
      return ~e0m3;
    };
    int map_k_tile = 0;
    uint32_t prefetched_formats = load_formats(0);
    auto next_formats = [&]() -> uint32_t {
      uint32_t const formats = prefetched_formats;
      prefetched_formats = load_formats(++map_k_tile);
      return formats;
    };

    uint32_t skip_wait = k_tile_count <= 0;
    auto barrier_token = mainloop_pipeline.consumer_try_wait(
        mainloop_pipe_consumer_state, skip_wait);

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    if constexpr (IsOverlappingAccum) {
      if (k_tile_count > 0) {
        mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state,
                                        barrier_token);

        int read_stage = mainloop_pipe_consumer_state.index();
        auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

        ++mainloop_pipe_consumer_state;
        --k_tile_count;
        skip_wait = k_tile_count <= 0;
        barrier_token = mainloop_pipeline.consumer_try_wait(
            mainloop_pipe_consumer_state, skip_wait);

        if (cute::elect_one_sync()) {
          copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_, _, _, _, read_stage),
               thr_tCtSFA_s2t);
          copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_, _, _, _, read_stage),
               thr_tCtSFB_s2t);
        }

        accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
        uint32_t const formats = next_formats();

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          tiled_mma.idesc_.b_format_ = (formats >> k_block) & 1u;
          cute::gemm(
              tiled_mma.with(tiled_mma.accumulate_, tCtSFA(_, _, k_block),
                             tCtSFB_mma(_, _, k_block)),
              tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage),
              accumulators);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }

        mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
      }
    } else {
      accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
    }

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state,
                                      barrier_token);

      int read_stage = mainloop_pipe_consumer_state.index();
      auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

      ++mainloop_pipe_consumer_state;
      --k_tile_count;
      skip_wait = k_tile_count <= 0;
      barrier_token = mainloop_pipeline.consumer_try_wait(
          mainloop_pipe_consumer_state, skip_wait);

      if (cute::elect_one_sync()) {
        copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_, _, _, _, read_stage),
             thr_tCtSFA_s2t);
        copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_, _, _, _, read_stage),
             thr_tCtSFB_s2t);
      }
      uint32_t const formats = next_formats();

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        tiled_mma.idesc_.b_format_ = (formats >> k_block) & 1u;
        cute::gemm(tiled_mma.with(tiled_mma.accumulate_, tCtSFA(_, _, k_block),
                                  tCtSFB_mma(_, _, k_block)),
                   tCrA(_, _, k_block, read_stage),
                   tCrB(_, _, k_block, read_stage), accumulators);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
    }

    return mainloop_pipe_consumer_state;
  }

 protected:
  uint32_t const* tile_map_{nullptr};
  int tile_map_stride_{0};
};

}  // namespace vllm::flipquant
