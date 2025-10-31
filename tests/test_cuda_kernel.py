import torch
import unittest
from torch.utils import cpp_extension



class TestCudaKernel(unittest.TestCase):
    def text_batch_matrix_mul(self):
        batch_matrix_mul_module = cpp_extension.load(
            name="batch_matrix_mul",
            sources=["./vision_transformers/torch-ext/torch_binding.cpp", "./vision_transformers/cuda/deformable_attn_kernel.cu"],
            extra_cflags=["-O3"],
            extra_ldflags=["-lcudart", "-lcuda"],
            extra_include_paths=cpp_extension.include_paths(device_type="cuda"),
            verbose=True,
            with_cuda=True
        )
        test_tensor_a = torch.randn((2, 3, 4), dtype=torch.float32).cuda()
        test_tensor_b = torch.randn((2, 4, 5), dtype=torch.float32).cuda()
        output_tensor = batch_matrix_mul_module.batch_matrix_mul(test_tensor_a, test_tensor_b)
        expected_output = torch.bmm(test_tensor_a, test_tensor_b)
        self.assertTrue(torch.allclose(output_tensor, expected_output, atol=1e-6))

    def test_bilinear_interpolate(self):
        bilinear_interpolate_module = cpp_extension.load(
            name="bilinear_interpolate",
            sources=["./vision_transformers/torch-ext/torch_binding.cpp", "./vision_transformers/cuda/deformable_attn_kernel.cu"],
            extra_cflags=["-O3"],
            extra_ldflags=["-lcudart", "-lcuda"],
#            extra_include_paths=["$$(CONDA_PREFIX)/include", "$$(CONDA_PREFIX)/targets/x86_64-linux/include/"],
            extra_include_paths=cpp_extension.include_paths(device_type="cuda"),
            verbose=True,
            with_cuda=True
        )
#        from vision_transformers.cuda.deformable_attn_kernel import bilinear_interpolate
        test_matrix = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 4, 3],
            [1, 1, 1, 1, 1]
        ]
        input_tensor = torch.tensor(test_matrix, dtype=torch.float32).cuda()
#        output_tensor = torch.zeros((3,), dtype=torch.float32).cuda()

        # Define the coordinates for interpolation
#        coords = torch.tensor([[1.5, 1.5], [3.0, 3.0]], dtype=torch.float32).cuda()
        x_coords = torch.tensor([3.2, 3.8, 4.3], dtype=torch.float32).cuda()
        y_coords = torch.tensor([3.4, 2.7, 3.1], dtype=torch.float32).cuda()

        # Call the CUDA kernel (assuming it's wrapped in a PyTorch function)
        output_tensor = bilinear_interpolate_module.bilinear_interpolate(input_tensor, y_coords, x_coords)

        # Move the result back to CPU for verification
        result = output_tensor.cpu().numpy()

        # Expected results calculated manually
        expected = [[1.0, 1.0], [1.75, 2.0]]
        import pdb; pdb.set_trace()

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(result[i][j], expected[i][j], places=4)


if __name__ == "__main__":
    unittest.main()