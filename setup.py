from setuptools import Extension, find_packages, setup

import numpy as np


def find_version():
    with open("kpconv_torch/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split("=")[1].strip(" '\"")
    raise RuntimeError("Unable to find version string. Should be in __init__.py.")


with open("README.md", "rb") as f:
    readme = f.read().decode("utf-8")

subsampling_module = Extension(
    name="grid_subsampling",
    sources=[
        "cpp_wrappers/cpp_utils/cloud/cloud.cpp",
        "cpp_wrappers/cpp_subsampling/grid_subsampling/grid_subsampling.cpp",
        "cpp_wrappers/cpp_subsampling/wrapper.cpp",
    ],
    extra_compile_args=["-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
)

neighboring_module = Extension(
    name="radius_neighbors",
    sources=[
        "cpp_wrappers/cpp_utils/cloud/cloud.cpp",
        "cpp_wrappers/cpp_neighbors/neighbors/neighbors.cpp",
        "cpp_wrappers/cpp_neighbors/wrapper.cpp",
    ],
    extra_compile_args=["-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
)

setup(
    name="kpconv_torch",
    version=find_version(),
    description=(
        "An implementation of KPConv algorithm with PyTorch (initial credit to Hugues Thomas)"
    ),
    long_description=readme,
    author="Raphaël Delhome",
    author_email="raphael.delhome@oslandia.com",
    maintainer="Oslandia",
    maintainer_email="infos@oslandia.com",
    url="",
    entry_points={"console_scripts": ["kpconv=kpconv_torch.cli.__main__:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3",
    packages=find_packages(),
    ext_modules=[subsampling_module, neighboring_module],
    include_dirs=np.get_include(),
)
