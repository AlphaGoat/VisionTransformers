#include <torch/extension.h>


torch::Tensor bilinear_interpolate(
    const torch::Tensor& input,
    torch::Tensor& y,
    torch::Tensor& x
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilinear_interpolate", &bilinear_interpolate, "Bilinear Interpolate (CUDA)");
    py::arg("input"), py::arg("y"), py::arg("x");
}



//#include <torch/library.h>
//#include "registration.h"
//#include "torch_binding.h"
//
//TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
//    ops.def("bilinear_interpolate(torch::Tensor input, int height, int width, torch::Tensor output) -> ()");
//    ops.impl("bilinear_interpolate", torch::kCUDA, &bilinear_interpolate);
//}