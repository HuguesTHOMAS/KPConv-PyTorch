from setuptools import find_packages, setup


def find_version():
    with open("kpconv_torch/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split("=")[1].strip(" '\"")
    raise RuntimeError("Unable to find version string. Should be in __init__.py.")


with open("README.md", "rb") as f:
    readme = f.read().decode("utf-8")

extra_requirements = {
    "dev": [
        "importlib-metadata<5.0",
        "pytest==5.3.0",
        "black==22.3.0",
        "flake8==3.8.3",
        "pre-commit==2.9.2",
    ]
}

setup(
    name="kpconv_torch",
    version=find_version(),
    description="An implementation of KPConv algorithm with PyTorch (initial credit to Hugues Thomas)",
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
    extras_require=extra_requirements,
    packages=find_packages(),
)
