#ifndef COMMON_CUH
#define COMMON_CUH

using tensor_pair = std::pair<torch::Tensor, torch::Tensor>;

torch::Tensor add_offsets_forward(const torch::Tensor& reference_points, const torch::Tensor& offsets);
tensor_pair add_offsets_backward(const torch::Tensor& grad_C, int N, int K);
torch::Tensor batch_matrix_mul_forward(const torch::Tensor& A, const torch::Tensor& B);
tensor_pair batch_matrix_mul_backward(const torch::Tensor& grad_C, const torch::Tensor& A, const torch::Tensor& B);
torch::Tensor batch_bias_add_forward(const torch::Tensor& A, const torch::Tensor& bias);
tensor_pair batch_bias_add_backward(const torch::Tensor& grad_C, int BATCH_SIZE, int N, int M);
torch::Tensor batch_softmax_forward(const torch::Tensor& A);
torch::Tensor batch_softmax_backward(const torch::Tensor& grad_C, const torch::Tensor& A);

#endif