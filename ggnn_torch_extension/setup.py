from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE_NAME = "ggnn_extension"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name=f"{PACKAGE_NAME}._C",
            sources=[
                "csrc/ggnn.cpp",
                "csrc/cpu/ggnn_cpu.cpp",
                "csrc/cuda/ggnn_cuda.cu",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-g", "-O3", "-std=c++17", "-w"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr", "-std=c++17", "-w"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)