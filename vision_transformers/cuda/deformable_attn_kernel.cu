/* Deformable Attention CUDA Kernel */
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


__global__ void matrix_add_forward(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void matrix_add_backward(
    const float* grad_C,
    float* grad_A,
    float* grad_B,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_A[idx] = grad_C[idx];
        grad_B[idx] = grad_C[idx];
    }
}


__global__ void matrix_mul_forward(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}


__global__ void matrix_mul_backward(
    const float* grad_C,
    const float* A,
    const float* B,
    float* grad_A,
    float* grad_B,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_A[idx] = grad_C[idx] * B[idx];
        grad_B[idx] = grad_C[idx] * A[idx];
    }
}



// CUDA kernel for deformable attention
//__global__ void deformable_attn_kernel(
//    const float* query,
//    const float* key,
//    const float* value,
//    const float* reference_points,
//    const float* offsets,
//    float* output,
//    int batch_size,
//) {
//    // Kernel implementation goes here
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    // Retrieve sampling point
//    float sampling_point;
//}


__global__ void bilinear_interpolate_kernel(
    const float* input,
    float* output,
    int height,
    int width,
    float *y,
    float *x,
    int N
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    // Bilinear interpolation implementation goes here
    int x1 = floor(x[idx]);
    int x2 = ceil(x[idx]);
    int y1 = floor(y[idx]);
    int y2 = ceil(y[idx]);

    float Q11 = input[y1 * width + x1];
    float Q12 = input[y1 * width + x2];
    float Q21 = input[y2 * width + x1];
    float Q22 = input[y2 * width + x2];

    float R1 = ((x2 - x[idx]) * Q11) + ((x[idx] - x1) * Q12);
    float R2 = ((x2 - x[idx]) * Q21) + ((x[idx] - x1) * Q22);

    output[idx] = ((y2 - y[idx]) * R1) + ((y[idx] - y1) * R2);
}


void deformable_attn(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& reference_points,
    const torch::Tensor& offsets,
    torch::Tensor& output
) {
    const int batch_size = query.size(0);
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

//    deformable_attn_kernel<<<blocks, threads>>>(
//        query.data_ptr<float>(),
//        key.data_ptr<float>(),
//        value.data_ptr<float>(),
//        reference_points.data_ptr<float>(),
//        offsets.data_ptr<float>(),
//        output.data_ptr<float>(),
//        batch_size
//    );

    cudaDeviceSynchronize();
}


torch::Tensor bilinear_interpolate(
    const torch::Tensor& input,
    torch::Tensor& y,
    torch::Tensor& x
) {
    const int N = x.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    int height = input.size(0);
    int width = input.size(1);

    torch::Tensor cont_input = input.contiguous();
//    output = output.contiguous();
    y = y.contiguous();
    x = x.contiguous();

    torch::Tensor output = torch::zeros({x.numel()}, input.options());

    bilinear_interpolate_kernel<<<blocks, threads>>>(
        cont_input.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width,
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();
    return output;
}


//// Pybind11 module is a macro that creates the entry point for the Python module.
//PYBIND11_MODULE(bilinear_interpolate_cuda, m) {
//    m.def("bilinear_interpolate_cuda", &bilinear_interpolate, "Bilinear Interpolation (CUDA)");
//}