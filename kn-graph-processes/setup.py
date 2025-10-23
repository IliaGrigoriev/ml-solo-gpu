from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name='msgpass_ext',
    ext_modules=[
        CUDAExtension('msgpass_ext', [
            'message_passing.cpp',
            'message_passing.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
