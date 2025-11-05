/* Deformable Attention CUDA Kernel */
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "common.cuh"





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


__global__ void bilinear_interpolate_forward_kernel(
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


__global__ void bilinear_interpolate_backward_kernel(
    const float* grad_output,
    float* grad_input,
    int height,
    int width,
    float *y,
    float *x,
    int N
) {
    // Backward pass implementation goes here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    // Compute gradients with respect to input
    // (This is a placeholder; actual gradient computation would be more complex)
    int x1 = floor(x[idx]);
    int x2 = ceil(x[idx]);
    int y1 = floor(y[idx]);
    int y2 = ceil(y[idx]);

    float grad = grad_output[idx] / 4.0f; // Distribute gradient equally for simplicity

    atomicAdd(&grad_input[y1 * width + x1], grad);
    atomicAdd(&grad_input[y1 * width + x2], grad);
    atomicAdd(&grad_input[y2 * width + x1], grad);
    atomicAdd(&grad_input[y2 * width + x2], grad);
}


torch::Tensor bilinear_interpolate_forward(
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


torch::Tensor bilinear_interpolate_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& y,
    const torch::Tensor& x,
    int height,
    int width
) {
    const int N = x.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    bilinear_interpolate_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        height,
        width,
        N
    );

    cudaDeviceSynchronize();
    return grad_input;
}


torch::Tensor multiscale_deformable_attn_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& reference_points,
    const torch::Tensor& feature_map,
    const torch::Tensor& W_offsets,
    const torch::Tensor& B_offsets,
    torch::Tensor& output
) {
    /* Deformable attention for a single head */

    const int batch_size = query.size(0);
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    // Get sampling offsets
    torch::Tensor offsets = batch_bias_add_forward(batch_matrix_mul_forward(W_offsets, query), B_offsets);

    // Get sampling points
    torch::Tensor sampling_points = add_offsets_forward(&reference_points, &offsets);

    // Generate attention weights
    torch::Tensor attention_weights = batch_matrix_mul_forward(query, key.transpose(1, 2));

    // sampling_points shape: (batch_size, num_sampling_points, 2)
    // Get sampling points from input feature map
    torch::Tensor sampled_features = bilinear_interpolate_forward(feature_map, sampling_points);

    // Shape: (batch_size, num_sampling_points, C)
    // Compute attention output
    output = torch::matmul(attention_weights, sampled_features);

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


torch::Tensor multi_head_attention_forward(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    int num_heads
) {
    /* Multi-head attention implementation */
    // Split queries, keys, values into multiple heads
    auto split_heads = [num_heads](const torch::Tensor& x) {
        int batch_size = x.size(0);
        int seq_len = x.size(1);
        int depth = x.size(2) / num_heads;
        return x.view({batch_size, seq_len, num_heads, depth}).permute({0, 2, 1, 3});
    };

    torch::Tensor Q = split_heads(queries);
    torch::Tensor K = split_heads(keys);
    torch::Tensor V = split_heads(values);

    // Scaled dot-product attention
    torch::Tensor scores = torch::matmul(Q, K.transpose(-2, -1)) / sqrt((float)(K.size(-1)));
    torch::Tensor attn_weights = torch::softmax(scores, -1);
    torch::Tensor attn_output = torch::matmul(attn_weights, V);

    // Concatenate heads
    attn_output = attn_output.permute({0, 2, 1, 3}).contiguous();
    attn_output = attn_output.view({attn_output.size(0), attn_output.size(1), -1});

    return attn_output;
}


torch::Tensor multi_head_attention_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& attention_weights,
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    int num_heads
) {
    /* Backward pass for multi-head attention */
    // Placeholder for backward implementation
    torch::Tensor grad_queries = torch::zeros_like(queries);
    torch::Tensor grad_keys = torch::zeros_like(keys);
    torch::Tensor grad_values = torch::zeros_like(values);

    // Actual gradient computations would go here
    grad_values = attention_weights.transpose(1, 2); // Placeholder
    grad_scores = batch_softmax_backward(grad_output, attention_weights); // Placeholder
    grad_queries = torch::matmul(grad_scores, keys); // Placeholder
    grad_keys = torch::matmul(queries.transpose(-2, -1), grad_scores); // Placeholder

    return {grad_queries, grad_keys, grad_values};
}


torch::Tensor deformable_attn_forward(
    const torch::Tensor& feature_map,
    const torch::Tensor& sample_points,
    const torch::Tensor& queries,
    const torch::Tensor& W_keys,
    const torch::Tensor& W_values,
    int num_heads
) {
    // Bilinear interpolate feature map at sampling locations
    torch::Tensor sampled_features = bilinear_interpolate_forward(&feature_map, &sample_points);

    // Generate keys and values
    torch::Tensor keys = batch_matrix_mul_forward(&W_keys, &sampled_features);
    torch::Tensor values = batch_matrix_mul_forward(&W_values, &sampled_features);

    // Feed queries, keys, and values into multi-head attention layer
    torch::Tensor output = multi_head_attention(queries, keys, values, num_heads);
    return output;
}


torch::Tensor deformable_attn_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& feature_map,
    const torch::Tensor& sample_points,
    const torch::Tensor& queries,
    const torch::Tensor& W_keys,
    const torch::Tensor& W_values,
    int num_heads
) {
    // Backward pass for deformable attention
    // Placeholder for backward implementation
    torch::Tensor grad_keys = torch::zeros_like(W_keys);
    torch::Tensor grad_values = torch::zeros_like(W_values);
    torch::Tensor grad_sampled_features = torch::zeros_like(sampled_features);
    torch::Tensor grad_offsets = torch::zeros_like(offsets);

    // Compute gradients with respect to queries, keys, values
    auto grads = multi_head_attention_backward(grad_output, queries, keys, values, num_heads);
    torch::Tensor grad_queries = grads[0];
    torch::Tensor grad_keys = grads[1];
    torch::Tensor grad_values = grads[2];

    // Compute gradients with respect to offsets
    // Placeholder for offset gradient computation
    grad_offsets = batch_matrix_mul_backward(grad_sampled_features, W_keys, sampled_features);
    grad_offsets += batch_matrix_mul_backward(grad_sampled_features, W_values, sampled_features);

    // Compute gradients with respect to feature map via bilinear interpolation backward
    torch::Tensor grad_feature_map = bilinear_interpolate_backward(grad_values, sample_points, feature_map.size(0), feature_map.size(1));

    return {grad_queries, grad_keys, grad_values, grad_offsets, grad_feature_map};
}

//// Pybind11 module is a macro that creates the entry point for the Python module.
//PYBIND11_MODULE(bilinear_interpolate_cuda, m) {
//    m.def("bilinear_interpolate_cuda", &bilinear_interpolate, "Bilinear Interpolation (CUDA)");
//}