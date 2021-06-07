import os
import sys
import glob
import os.path as osp
from setuptools import setup, find_packages

import torch
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None


def get_extensions():
    Extension = CppExtension
    define_macros = []
    libraries = []
    extra_compile_args = {'cxx': []}
    extra_link_args = []

    info = parallel_info()
    if 'parallel backend: OpenMP' in info and 'OpenMP not found' not in info:
        extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
        if sys.platform == 'win32':
            extra_compile_args['cxx'] += ['/openmp']
        else:
            extra_compile_args['cxx'] += ['-fopenmp']
    else:
        print('Compiling without OpenMP...')

    if WITH_CUDA:
        Extension = CUDAExtension
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr']
        extra_compile_args['nvcc'] = nvcc_flags

    extensions_dir = osp.join('csrc')
    main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))
    extensions = []
    for main in main_files:
        name = main.split(os.sep)[-1][:-4]

        sources = [main]

        path = osp.join(extensions_dir, 'cpu', f'{name}_cpu.cpp')
        if osp.exists(path):
            sources += [path]

        path = osp.join(extensions_dir, 'cuda', f'{name}_cuda.cu')
        if WITH_CUDA and osp.exists(path):
            sources += [path]

        extension = Extension(
            'torch_geometric_autoscale._' + name,
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
        )
        extensions += [extension]

    return extensions


install_requires = ['ogb', 'hydra-core']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_geometric_autoscale',
    version='0.0.0',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    description='PyGAS: Auto-Scaling GNNs in PyTorch Geometric',
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require},
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(),
)
