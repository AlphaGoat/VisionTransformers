// Common CUDA utilities
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


__global__ void batch_matrix_add_forward_kernel(
    const float* A,
    const float* B,
    float* C,
    int BATCH_SIZE,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * N * M) {
        int batch_num = idx / (N * M);
        int i = (idx / M) % N; // Row index
        int j = idx % M;       // Column index
        C[idx] = A[batch_num * N * M + i * M + j] + B[batch_num * N * M + i * M + j];
    }
}


__global__ void batch_matrix_add_backward_kernel(
    const float* grad_C,
    float* grad_A,
    float* grad_B,
    int BATCH_SIZE,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * N * M) {
        int batch_num = idx / (N * M);
        int i = (idx / M) % N; // Row index
        int j = idx % M;       // Column index
        grad_A[batch_num * N * M + i * M + j] = grad_C[idx];
        grad_B[batch_num * N * M + i * M + j] = grad_C[idx];
    }
}


__global__ void batch_bias_add_forward_kernel(
    const float* A,
    const float* B,
    float* C,
    int BATCH_SIZE,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * N * M) {
        int batch_num = idx / (N * M);
        int i = (idx / M) % N; // Row index
        int j = idx % M;       // Column index
        C[idx] = A[batch_num * N * M + i * M + j] + B[j];
    }
}


__global__ void batch_bias_add_backward_kernel(
    const float* grad_C,
    float* grad_A,
    float* grad_B,
    int BATCH_SIZE,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * N * M) {
        int batch_num = idx / (N * M);
        int i = (idx / M) % N; // Row index
        int j = idx % M;       // Column index
        grad_A[batch_num * N * M + i * M + j] = grad_C[idx];
        atomicAdd(&grad_B[j], grad_C[idx]);
    }
}


torch::Tensor batch_bias_add_forward(
    const torch::Tensor& A,
    const torch::Tensor& bias
) {
    // Check that the dimensions match
    TORCH_CHECK(A.size(0) == bias.size(0) && A.size(1) == bias.size(1), "Incompatible matrix dimensions");

    const int BATCH_SIZE = A.size(0);
    const int N = A.size(1);
    const int M = A.size(2);
    const int threads = 256;
    const int blocks = (BATCH_SIZE * N * M + threads - 1) / threads;

    torch::Tensor C = torch::zeros({BATCH_SIZE, N, M}, A.options());

    batch_bias_add_forward_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH_SIZE,
        N,
        M
    );

    return C;
}


torch::TensorPair batch_bias_add_backward(
    const torch::Tensor& grad_C,
    int BATCH_SIZE,
    int N,
    int M
) {
    const int threads = 256;
    const int blocks = (BATCH_SIZE * N * M + threads - 1) / threads;

    torch::Tensor grad_A = torch::zeros({BATCH_SIZE, N, M}, grad_C.options());
    torch::Tensor grad_B = torch::zeros({M}, grad_C.options());

    batch_bias_add_backward_kernel<<<blocks, threads>>>(
        grad_C.data_ptr<float>(),
        grad_A.data_ptr<float>(),
        grad_B.data_ptr<float>(),
        BATCH_SIZE,
        N,
        M
    );

    return {grad_A, grad_B};
}


__global__ void batch_softmax_forward_kernel(
    const float* A,
    float* C,
    int BATCH_SIZE,
    int N,
    int M,
    int AXIS
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * M) {
        int batch_num = idx / M;
        int j = idx % M;       // Column index
        float denom = 0.0f;
        for (int i = 0; i < N; ++i) {
            denom += expf(A[batch_num * N * M + i * M + j]);
        }
        C[batch_num * M + j] = expf(A[batch_num * N * M + i * M + j]) / denom;
    }
}


__global__ void batch_matrix_mul_forward_kernel(
    const float* A,
    const float* B,
    float* C,
    int BATCH_SIZE,
    int INNER,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * N * M) {
        int batch_num = idx / (N * M);
        int i = (idx / M) % N; // Row index
        int j = idx % M;       // Column index
        for (int k = 0; k < INNER; ++k) {
            C[idx] += A[batch_num * N * M + i * M + k] * B[batch_num * N * M + k * M + j];
        }
    }
}


__global__ void batch_matrix_mul_backward_kernel(
    const float* grad_C,
    const float* A,
    const float* B,
    float* grad_A,
    float* grad_B,
    int BATCH_SIZE,
    int INNER,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE * N * M) {
        int batch_num = idx / (N * M);
        int i = (idx / M) % N; // Row index
        int j = idx % M;       // Column index
        for (int k = 0; k < INNER; ++k) {
            grad_A[batch_num * N * M + i * M + j] = grad_C[idx] * B[batch_num * N * M + i * M + k];
            grad_B[batch_num * N * M + i * M + k] = grad_C[idx] * A[batch_num * N * M + k * M + j];
        }
    }
}


torch::Tensor batch_matrix_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    // Check that the inner dimensions match
    TORCH_CHECK(A.size(2) == B.size(1), "Incompatible matrix dimensions");

    const int BATCH_SIZE = A.size(0);
    const int INNER = A.size(2);
    const int N = A.size(1);
    const int M = B.size(2);
    const int threads = 256;
    const int blocks = (BATCH_SIZE * N * M + threads - 1) / threads;

    torch::Tensor C = torch::zeros({BATCH_SIZE, N, M}, A.options());

    batch_matrix_mul_forward_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH_SIZE,
        INNER,
        N,
        M
    );

    return C;
}


torch::TensorPair batch_matrix_mul_backward(
    const torch::Tensor& grad_C,
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    // Check that the inner dimensions match
    TORCH_CHECK(A.size(2) == B.size(1), "Incompatible matrix dimensions");

    const int BATCH_SIZE = A.size(0);
    const int INNER = A.size(2);
    const int N = A.size(1);
    const int M = B.size(2);
    const int threads = 256;
    const int blocks = (BATCH_SIZE * N * M + threads - 1) / threads;

    torch::Tensor grad_A = torch::zeros_like(A);
    torch::Tensor grad_B = torch::zeros_like(B);

    batch_matrix_mul_backward_kernel<<<blocks, threads>>>(
        grad_C.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        grad_A.data_ptr<float>(),
        grad_B.data_ptr<float>(),
        BATCH_SIZE,
        INNER,
        N,
        M
    );

    return {grad_A, grad_B};
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


__global__ void scalar_add_forward(
    const float* A,
    float scalar,
    float* C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + scalar;
    }
}


__global__ void scalar_add_backward(
    const float* grad_C,
    float* grad_A,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_A[idx] = grad_C[idx];
    }
}


__global__ void add_offsets_forward_kernel(
    const float* A,
    const float* B,
    float* C,
    int N,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * K) {
        int i = idx / K; // Row index
        C[idx] = A[i] + B[idx];
    }
}

__global__ void add_offsets_backward_kernel(
    const float* grad_C,
    float* grad_A,
    float* grad_B,
    int N,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * K) {
        int i = idx / K; // Row index
//        int j = idx % K; // Column index
        atomicAdd(&grad_A[i], grad_C[idx]);
        atomicAdd(&grad_B[idx], grad_C[idx]);
    }
}


torch::Tensor add_offsets_forward(
    const torch::Tensor& reference_points,
    const torch::Tensor& offsets
) {
    const int N = reference_points.size(0);
    const int K = offsets.size(0) / N;
    const int threads = 256;
    const int blocks = (N * K + threads - 1) / threads;

    torch::Tensor C = torch::zeros({N * K}, reference_points.options());

    broadcast_add_forward_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        K
    );

    return C;
}


torch::TensorPair add_offsets_backward(
    const torch::Tensor& grad_C,
    int N,
    int K
) {
    const int threads = 256;
    const int blocks = (N * K + threads - 1) / threads;

    torch::Tensor grad_A = torch::zeros({N}, grad_C.options());
    torch::Tensor grad_B = torch::zeros({K}, grad_C.options());

    broadcast_add_backward_kernel<<<blocks, threads>>>(
        grad_C.data_ptr<float>(),
        grad_A.data_ptr<float>(),
        grad_B.data_ptr<float>(),
        N,
        M
    );

    return {grad_A, grad_B};
}
