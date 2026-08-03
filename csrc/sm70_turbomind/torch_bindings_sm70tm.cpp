// Thin torch binding for the vendored TurboMind SM70 AWQ s884 GEMM.
// Registered under namespace "_sm70tm" so it coexists with our existing
// WMMA AWQ kernel (torch.ops._C.awq_gemm_sm70, 4-arg) as a switchable backend.

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <vector>

// Forward declarations of the public entry points defined in
// ops/awq_sm70_gemm.cu (extern-C-style wrappers around vllm::awq_sm70::*).
std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor _kernel,
                                            torch::Tensor _scaling_factors,
                                            torch::Tensor _zeros,
                                            int64_t group_size,
                                            bool interleave_gated_silu);

torch::Tensor awq_gemm_sm70(torch::Tensor _in_feats, torch::Tensor _kernel,
                            torch::Tensor _scaling_factors, int64_t group_size,
                            int64_t k_ld, int64_t q_ld);

void awq_gemm_sm70_out(torch::Tensor out, torch::Tensor _in_feats,
                       torch::Tensor _kernel, torch::Tensor _scaling_factors,
                       int64_t group_size, int64_t k_ld, int64_t q_ld,
                       bool gated_silu);

TORCH_LIBRARY_FRAGMENT(_sm70tm, m) {
  m.def("awq_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, Tensor "
        "_zeros, int group_size, bool interleave_gated_silu) -> Tensor[]");
  m.def("awq_gemm_sm70(Tensor _in_feats, Tensor _kernel, Tensor "
        "_scaling_factors, int group_size, int k_ld, int q_ld) -> Tensor");
  m.def("awq_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
        "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
        "bool gated_silu) -> ()");
}

TORCH_LIBRARY_IMPL(_sm70tm, CUDA, m) {
  m.impl("awq_sm70_prepare", &awq_sm70_prepare);
  m.impl("awq_gemm_sm70", &awq_gemm_sm70);
  m.impl("awq_gemm_sm70_out", &awq_gemm_sm70_out);
}

// Provides PyInit_<name> so the extension can be imported as a Python module.
// Op registration above runs via static initializers at load time.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
