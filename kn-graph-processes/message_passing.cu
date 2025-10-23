#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void msgpass_kernel(
    const long* __restrict__ indptr,
    const long* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t num_nodes,
    int64_t feat_dim)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;

    int64_t start = indptr[v];
    int64_t end   = indptr[v + 1];
    for (int f = 0; f < feat_dim; ++f) {
        float sum = 0.0f;
        for (int64_t eid = start; eid < end; ++eid)
            sum += x[indices[eid] * feat_dim + f];
        out[v * feat_dim + f] = sum;
    }
}

torch::Tensor message_passing_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor x)
{
    int64_t V = x.size(0);
    int64_t F = x.size(1);
    auto out = torch::zeros_like(x);

    int threads = 256;
    int blocks = (V + threads - 1) / threads;

    msgpass_kernel<<<blocks, threads>>>(
        indptr.data_ptr<long>(),
        indices.data_ptr<long>(),
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        V, F);

    return out;
}
