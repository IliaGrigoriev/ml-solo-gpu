#include <torch/extension.h>

torch::Tensor message_passing_cuda(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor x);

torch::Tensor message_passing(
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor x)
{
    return message_passing_cuda(indptr, indices, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("message_passing", &message_passing, "Graph message passing (CUDA)");
}
