
# Installation instructions

In order to exploit the library in optimal conditions, make sure [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) are installed.

Installing `kpconv_torch` on your system is as simple as executing the following commands in a
virtual environment:

```bash
python -m pip install numpy
python setup.py build_ext --inplace
python -m pip install -r requirements.txt
```

You should now be able to train Kernel-Point Convolution models
