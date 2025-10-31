#include <torch/extension.h>


torch::Tensor bilinear_interpolate_forward(
    const torch::Tensor& input,
    torch::Tensor& y,
    torch::Tensor& x
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilinear_interpolate_forward", &bilinear_interpolate_forward, "Bilinear Interpolate (CUDA)");
    py::arg("input"), py::arg("y"), py::arg("x");
}

torch::Tensor batch_matrix_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_matrix_mul_forward", &batch_matrix_mul_forward, "Batch Matrix Multiplication Forward (CUDA)");
    py::arg("A"), py::arg("B");
}

torch::Tensor batch_matrix_mul_backward(
    const torch::Tensor& grad_C,
    const torch::Tensor& A,
    const torch::Tensor& B
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_matrix_mul_backward", &batch_matrix_mul_backward, "Batch Matrix Multiplication Backward (CUDA)");
    py::arg("grad_C"), py::arg("A"), py::arg("B");
}

//#include <torch/library.h>
//#include "registration.h"
//#include "torch_binding.h"
//
//TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
//    ops.def("bilinear_interpolate(torch::Tensor input, int height, int width, torch::Tensor output) -> ()");
//    ops.impl("bilinear_interpolate", torch::kCUDA, &bilinear_interpolate);
//}