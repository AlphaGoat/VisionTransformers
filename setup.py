from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="vision_transformers",
    version="0.1",
    description="A library for Vision Transformer models with custom CUDA kernels",
    author="Peter Thomas",
    author_email="pjtpeterjohnthomas@gmail.com",
    ext_modules=[
        CUDAExtension(
            name="deformable_attn_kernel",
            sources=[
                "vision_transformers/cuda/deformable_attn_kernel.cu",
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_70', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)