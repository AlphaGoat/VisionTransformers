#ifndef COMMON_CUH
#define COMMON_CUH


torch::Tensor add_offsets_forward(const torch::Tensor& reference_points, const torch::Tensor& offsets);
torch::TensorPair add_offsets_backward(const torch::Tensor& grad_C, int N, int K);
torch::Tensor batch_matrix_mul_forward(const torch::Tensor& A, const torch::Tensor& B);
torch::TensorPair batch_matrix_mul_backward(const torch::Tensor& grad_C, const torch::Tensor& A, const torch::Tensor& B);
torch::Tensor batch_bias_add_forward(const torch::Tensor& A, const torch::Tensor& bias);
torch::TensorPair batch_bias_add_backward(const torch::Tensor& grad_C, int BATCH_SIZE, int N, int M);

#endif