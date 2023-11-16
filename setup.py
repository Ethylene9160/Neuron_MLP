from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "myneuron_module",
        ["PY_MPL.cpp"],  # 替换为您的绑定代码文件名
        # 其他需要的编译器和链接器标志
    ),
]

setup(
    name="myneuron_module",
    version="0.1",
    author="Your Name",
    description="Python binding for MyNeuron class using pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
