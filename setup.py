from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

# setup(
#     name='droid_backends',
#     ext_modules=[
#         CUDAExtension('droid_backends',
#             include_dirs=[osp.join(ROOT, 'thirdparty/lietorch/eigen')],
#             sources=[
#                 'src/lib/droid.cpp',
#                 'src/lib/droid_kernels.cu',
#                 'src/lib/correlation_kernels.cu',
#                 'src/lib/altcorr_kernel.cu',
#             ],
#             extra_compile_args={
#                 'cxx': ['-O3'],
#                 'nvcc': ['-O3',
#                     '-gencode=arch=compute_60,code=sm_60',
#                     '-gencode=arch=compute_61,code=sm_61',
#                     '-gencode=arch=compute_70,code=sm_70',
#                     '-gencode=arch=compute_75,code=sm_75',
#                     '-gencode=arch=compute_80,code=sm_80',
#                     '-gencode=arch=compute_86,code=sm_86',
#                 ]
#             }),
#     ],
#     cmdclass={ 'build_ext' : BuildExtension }
# )
setup(
    name='dpvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['src/dpvo/altcorr/correlation.cpp', 'src/dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba',
            sources=['src/dpvo/fastba/ba.cpp', 'src/dpvo/fastba/ba_cuda.cu', 'src/dpvo/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            },
            include_dirs=[
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')]
            ),
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'src/dpvo/lietorch/include'), 
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')],
            sources=[
                'src/dpvo/lietorch/src/lietorch.cpp', 
                'src/dpvo/lietorch/src/lietorch_gpu.cu',
                'src/dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })