#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/async_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__async(void) { return NULL; }
#endif

void synchronize() {
#ifdef WITH_CUDA
  synchronize_cuda();
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

void read_async(torch::Tensor src,
                torch::optional<torch::Tensor> optional_offset,
                torch::optional<torch::Tensor> optional_count,
                torch::Tensor index, torch::Tensor dst, torch::Tensor buffer) {
#ifdef WITH_CUDA
  read_async_cuda(src, optional_offset, optional_count, index, dst, buffer);
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

void write_async(torch::Tensor src, torch::Tensor offset, torch::Tensor count,
                 torch::Tensor dst) {
#ifdef WITH_CUDA
  write_async_cuda(src, offset, count, dst);
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_geometric_autoscale::synchronize", &synchronize)
        .op("torch_geometric_autoscale::read_async", &read_async)
        .op("torch_geometric_autoscale::write_async", &write_async);
