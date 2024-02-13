from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


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


class LazyImportBuildExtCmd(build_ext):
    """Overload build_ext by importing numpy for compiled code

    See https://ymd_h.gitlab.io/ymd_blog/posts/setup_config_for_cython_numpy/
    """

    def run(self):
        import numpy as np

        self.include_dirs.append(np.get_include())
        super().run()


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
    install_requires=["numpy"],
    setup_requires=["numpy"],
    packages=find_packages(),
    ext_modules=[subsampling_module, neighboring_module],
    cmdclass={"build_ext": LazyImportBuildExtCmd},
)
