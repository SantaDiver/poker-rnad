#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>

int add(int i, int j) {
    return i + j;
}

torch::jit::Module module_identity(const torch::jit::Module &mod) { return mod; }

torch::Tensor tensor_identity(const torch::Tensor &tnsr) {
    std::cerr << "123" << std::endl;
    return tnsr;
}

PYBIND11_MODULE(poker_rnad_py, m) {
    m.def("add", &add);
    m.def("tensor_identity", &tensor_identity);
    m.def("module_identity", &module_identity);
}
