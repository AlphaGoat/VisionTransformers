#ifndef TORCH_BINDING_H
#define TORCH_BINDING_H

#include <torch/extension.h>

void bilinear_interpolate(
    const torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& y,
    torch::Tensor& x
);

#endif // TORCH_BINDING_H