import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory of this setup.py file
this_dir = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(this_dir, 'csrc')
cpu_dir = os.path.join(csrc_dir, 'cpu')
cuda_dir = os.path.join(csrc_dir, 'cuda')
# Add the path to the kernels directory
cuda_kernels_dir = os.path.join(cuda_dir, 'kernels')
cuda_utils_dir = os.path.join(cuda_dir, 'utils')

setup(
    name='ggnn_torch_extension',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='ggnn_torch_extension._C', # This will be the name of the compiled .so file
            sources=[
                os.path.join(csrc_dir, 'ggnn.cpp'),
                os.path.join(cpu_dir, 'ggnn_cpu.cpp'),
                os.path.join(cuda_dir, 'ggnn_cuda.cu'),
            ],
            include_dirs=[
                # Add the new kernels directory to the include paths
                csrc_dir,
                cpu_dir,
                cuda_dir,
                cuda_kernels_dir, # <-- THIS IS THE NEW LINE
                cuda_utils_dir,
            ],
            extra_compile_args={
                'cxx': ['-g', '-O0', '-std=c++17', '-w'], # Added '-w'
                'nvcc': ['-O0', '--expt-relaxed-constexpr', '-std=c++17', '-w'], # Added '-w'
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)