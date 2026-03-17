// SPDX-License-Identifier: Apache-2.0

/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "cuda_compat.h"
namespace lmc {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding_fused(
    scalar_t* __restrict__ arr, const scalar_t* __restrict__ old_cos_ptr,
    const scalar_t* __restrict__ old_sin_ptr,
    const scalar_t* __restrict__ new_cos_ptr,
    const scalar_t* __restrict__ new_sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t old_cos, old_sin;
  scalar_t new_cos, new_sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    old_cos = LMCACHE_LDG(old_cos_ptr + x_index);
    old_sin = LMCACHE_LDG(old_sin_ptr + x_index);

    new_cos = LMCACHE_LDG(new_cos_ptr + x_index);
    new_sin = LMCACHE_LDG(new_sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    old_cos = LMCACHE_LDG(old_cos_ptr + x_index / 2);
    old_sin = LMCACHE_LDG(old_sin_ptr + x_index / 2);

    new_cos = LMCACHE_LDG(new_cos_ptr + x_index / 2);
    new_sin = LMCACHE_LDG(new_sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];

  const scalar_t x_reverse = x * old_cos + y * old_sin;
  const scalar_t y_reverse = y * old_cos - x * old_sin;

  arr[x_index] = x_reverse * new_cos - y_reverse * new_sin;
  arr[y_index] = y_reverse * new_cos + x_reverse * new_sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding_fused(
    scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* old_cache_ptr, const scalar_t* new_cache_ptr,
    const int head_size, const int num_kv_heads, const int rot_dim,
    const int token_idx, const int64_t key_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* old_cos_ptr = old_cache_ptr;
  const scalar_t* old_sin_ptr = old_cache_ptr + embed_dim;

  const scalar_t* new_cos_ptr = new_cache_ptr;
  const scalar_t* new_sin_ptr = new_cache_ptr + embed_dim;

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding_fused<scalar_t, IS_NEOX>(
        key + token_head, old_cos_ptr, old_sin_ptr, new_cos_ptr, new_sin_ptr,
        rot_offset, embed_dim);
  }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel_fused(
    const int64_t* __restrict__ old_positions,  // [batch_size, seq_len] or
                                                // [num_tokens]

    const int64_t* __restrict__ new_positions,  // [batch_size, seq_len] or
                                                // [num_tokens]

    scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim, const int64_t key_stride, const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t old_pos = old_positions[token_idx];
  int64_t new_pos = new_positions[token_idx];

  const scalar_t* old_cache_ptr = cos_sin_cache + old_pos * rot_dim;
  const scalar_t* new_cache_ptr = cos_sin_cache + new_pos * rot_dim;

  apply_rotary_embedding_fused<scalar_t, IS_NEOX>(
      key, old_cache_ptr, new_cache_ptr, head_size, num_kv_heads, rot_dim,
      token_idx, key_stride);
}

}  // namespace lmc

void rotary_embedding_k_fused(
    const torch::Tensor&
        old_positions,  // [batch_size, seq_len] or [num_tokens]
    const torch::Tensor&
        new_positions,   // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& key,  // [batch_size, seq_len, num_kv_heads * head_size] or
                         // Jiayi: [num_tokens, num_kv_heads, head_size]
    int64_t head_size,
    const torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  int64_t num_tokens = key.numel() / (key.size(-1) * key.size(-2));
  int rot_dim = cos_sin_cache.size(1);
  int num_kv_heads = key.size(-2);
  int64_t key_stride = num_kv_heads * head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_kv_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  LMC_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "rotary_embedding_k_fused", [&] {
        if (is_neox) {
          lmc::rotary_embedding_kernel_fused<scalar_t, true>
              <<<grid, block, 0, stream>>>(
                  old_positions.data_ptr<int64_t>(),
                  new_positions.data_ptr<int64_t>(), key.data_ptr<scalar_t>(),
                  cos_sin_cache.data_ptr<scalar_t>(), rot_dim, key_stride,
                  num_kv_heads, head_size);
        } else {
          lmc::rotary_embedding_kernel_fused<scalar_t, false>
              <<<grid, block, 0, stream>>>(
                  old_positions.data_ptr<int64_t>(),
                  new_positions.data_ptr<int64_t>(), key.data_ptr<scalar_t>(),
                  cos_sin_cache.data_ptr<scalar_t>(), rot_dim, key_stride,
                  num_kv_heads, head_size);
        }
      });
}

namespace lmc {

template <typename scalar_t, bool IS_NEOX>
__global__ void multi_fragment_copy_rope_kv_kernel(
    const int64_t* __restrict__ src_ptrs,
    const int64_t* __restrict__ dst_starts,
    const int64_t* __restrict__ span_lens,
    const int64_t* __restrict__ position_offsets,
    const int64_t* __restrict__ old_positions,
    const int64_t* __restrict__ new_positions,
    scalar_t* __restrict__ dst_kv,
    const scalar_t* __restrict__ cos_sin_cache, const int rot_dim,
    const int num_kv_heads, const int head_size,
    const int64_t dst_plane_stride) {
  const int64_t span_idx = blockIdx.y;
  const int64_t local_token_idx = blockIdx.x;
  const int64_t span_len = span_lens[span_idx];
  if (local_token_idx >= span_len) {
    return;
  }

  const int64_t dst_token_idx = dst_starts[span_idx] + local_token_idx;
  const int64_t pos_offset = position_offsets[span_idx] + local_token_idx;
  const int64_t hidden_dim = static_cast<int64_t>(num_kv_heads) * head_size;

  scalar_t* dst_k = dst_kv + dst_token_idx * hidden_dim;
  scalar_t* dst_v = dst_kv + dst_plane_stride + dst_token_idx * hidden_dim;

  const scalar_t* src_base =
      reinterpret_cast<const scalar_t*>(src_ptrs[span_idx]);
  const scalar_t* src_k = src_base + local_token_idx * hidden_dim;
  const scalar_t* src_v =
      src_base + span_len * hidden_dim + local_token_idx * hidden_dim;

  const int64_t old_pos = old_positions[pos_offset];
  const int64_t new_pos = new_positions[pos_offset];
  const int embed_dim = rot_dim / 2;
  const scalar_t* old_cache_ptr = cos_sin_cache + old_pos * rot_dim;
  const scalar_t* new_cache_ptr = cos_sin_cache + new_pos * rot_dim;

  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    dst_v[i] = src_v[i];
  }

  const int nk = num_kv_heads * embed_dim;
  const scalar_t* old_cos_ptr = old_cache_ptr;
  const scalar_t* old_sin_ptr = old_cache_ptr + embed_dim;
  const scalar_t* new_cos_ptr = new_cache_ptr;
  const scalar_t* new_sin_ptr = new_cache_ptr + embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    int x_index, y_index;
    scalar_t old_cos, old_sin;
    scalar_t new_cos, new_sin;
    if (IS_NEOX) {
      x_index = rot_offset;
      y_index = embed_dim + rot_offset;
      old_cos = LMCACHE_LDG(old_cos_ptr + x_index);
      old_sin = LMCACHE_LDG(old_sin_ptr + x_index);
      new_cos = LMCACHE_LDG(new_cos_ptr + x_index);
      new_sin = LMCACHE_LDG(new_sin_ptr + x_index);
    } else {
      x_index = 2 * rot_offset;
      y_index = 2 * rot_offset + 1;
      old_cos = LMCACHE_LDG(old_cos_ptr + x_index / 2);
      old_sin = LMCACHE_LDG(old_sin_ptr + x_index / 2);
      new_cos = LMCACHE_LDG(new_cos_ptr + x_index / 2);
      new_sin = LMCACHE_LDG(new_sin_ptr + x_index / 2);
    }

    const int64_t base = static_cast<int64_t>(head_idx) * head_size;
    const scalar_t x = src_k[base + x_index];
    const scalar_t y = src_k[base + y_index];
    const scalar_t x_reverse = x * old_cos + y * old_sin;
    const scalar_t y_reverse = y * old_cos - x * old_sin;
    dst_k[base + x_index] = x_reverse * new_cos - y_reverse * new_sin;
    dst_k[base + y_index] = y_reverse * new_cos + x_reverse * new_sin;
  }
}

}  // namespace lmc

void multi_fragment_copy_rope_kv_fused(
    const torch::Tensor& src_ptrs, const torch::Tensor& dst_starts,
    const torch::Tensor& span_lens, const torch::Tensor& position_offsets,
    const torch::Tensor& old_positions, const torch::Tensor& new_positions,
    torch::Tensor& dst_kv, int64_t head_size,
    const torch::Tensor& cos_sin_cache, bool is_neox) {
  TORCH_CHECK(src_ptrs.is_cuda(), "src_ptrs must be a CUDA tensor");
  TORCH_CHECK(dst_starts.is_cuda(), "dst_starts must be a CUDA tensor");
  TORCH_CHECK(span_lens.is_cuda(), "span_lens must be a CUDA tensor");
  TORCH_CHECK(position_offsets.is_cuda(),
              "position_offsets must be a CUDA tensor");
  TORCH_CHECK(old_positions.is_cuda(), "old_positions must be a CUDA tensor");
  TORCH_CHECK(new_positions.is_cuda(), "new_positions must be a CUDA tensor");
  TORCH_CHECK(dst_kv.is_cuda(), "dst_kv must be a CUDA tensor");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be a CUDA tensor");
  TORCH_CHECK(src_ptrs.scalar_type() == torch::kInt64,
              "src_ptrs must be int64");
  TORCH_CHECK(dst_starts.scalar_type() == torch::kInt64,
              "dst_starts must be int64");
  TORCH_CHECK(span_lens.scalar_type() == torch::kInt64,
              "span_lens must be int64");
  TORCH_CHECK(position_offsets.scalar_type() == torch::kInt64,
              "position_offsets must be int64");
  TORCH_CHECK(old_positions.scalar_type() == torch::kInt64,
              "old_positions must be int64");
  TORCH_CHECK(new_positions.scalar_type() == torch::kInt64,
              "new_positions must be int64");
  TORCH_CHECK(dst_kv.dim() == 3 && dst_kv.size(0) == 2,
              "dst_kv must have shape [2, num_tokens, hidden_dim]");
  TORCH_CHECK(dst_starts.numel() == span_lens.numel() &&
                  dst_starts.numel() == position_offsets.numel() &&
                  dst_starts.numel() == src_ptrs.numel(),
              "span metadata tensors must have identical lengths");
  TORCH_CHECK(old_positions.numel() == new_positions.numel(),
              "old_positions and new_positions must have identical lengths");

  const int64_t num_spans = src_ptrs.numel();
  if (num_spans == 0) {
    return;
  }

  const int rot_dim = cos_sin_cache.size(1);
  const int hidden_dim = dst_kv.size(2);
  TORCH_CHECK(head_size > 0, "head_size must be positive");
  TORCH_CHECK(hidden_dim % head_size == 0,
              "hidden_dim must be divisible by head_size");
  const int num_kv_heads = hidden_dim / head_size;
  const int64_t dst_plane_stride = dst_kv.size(1) * hidden_dim;
  const int64_t max_span_len = span_lens.max().item<int64_t>();
  if (max_span_len == 0) {
    return;
  }

  dim3 grid(max_span_len, num_spans);
  dim3 block(std::min<int64_t>(num_kv_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(dst_kv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  LMC_DISPATCH_FLOATING_TYPES(
      dst_kv.scalar_type(), "multi_fragment_copy_rope_kv_fused", [&] {
        if (is_neox) {
          lmc::multi_fragment_copy_rope_kv_kernel<scalar_t, true>
              <<<grid, block, 0, stream>>>(
                  src_ptrs.data_ptr<int64_t>(), dst_starts.data_ptr<int64_t>(),
                  span_lens.data_ptr<int64_t>(),
                  position_offsets.data_ptr<int64_t>(),
                  old_positions.data_ptr<int64_t>(),
                  new_positions.data_ptr<int64_t>(), dst_kv.data_ptr<scalar_t>(),
                  cos_sin_cache.data_ptr<scalar_t>(), rot_dim, num_kv_heads,
                  head_size, dst_plane_stride);
        } else {
          lmc::multi_fragment_copy_rope_kv_kernel<scalar_t, false>
              <<<grid, block, 0, stream>>>(
                  src_ptrs.data_ptr<int64_t>(), dst_starts.data_ptr<int64_t>(),
                  span_lens.data_ptr<int64_t>(),
                  position_offsets.data_ptr<int64_t>(),
                  old_positions.data_ptr<int64_t>(),
                  new_positions.data_ptr<int64_t>(), dst_kv.data_ptr<scalar_t>(),
                  cos_sin_cache.data_ptr<scalar_t>(), rot_dim, num_kv_heads,
                  head_size, dst_plane_stride);
        }
      });
}
