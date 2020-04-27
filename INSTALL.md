
# Installation instructions

## Ubuntu 18.04
     
* Make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. One configuration has been tested: 
     - PyTorch 1.4.0, CUDA 10.1 and cuDNN 7.6
     
* Ensure all python packages are installed :

          sudo apt update
          sudo apt install python3-dev python3-pip python3-tk

* Follow <a href="https://pytorch.org/get-started/locally/">PyTorch installation procedure</a>.

* Install the other dependencies with pip:
     - numpy
     - scikit-learn
     - PyYAML
     - matplotlib (for visualization)
     - mayavi (for visualization)
     - PyQt5 (for visualization)
     
* Compile the C++ extension modules for python located in `cpp_wrappers`. Open a terminal in this folder, and run:

          sh compile_wrappers.sh

You should now be able to train Kernel-Point Convolution models

## Windows 10
     
* Make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. One configuration has been tested: 
     - PyTorch 1.4.0, CUDA 10.1 and cuDNN 7.5
     
* Follow <a href="https://pytorch.org/get-started/locally/">PyTorch installation procedure</a>.
     
* We used the PyCharm IDE to pip install all python dependencies (including PyTorch) in a venv:
     - torch
     - torchvision
     - numpy
     - scikit-learn
     - PyYAML
     - matplotlib (for visualization)
     - mayavi (for visualization)
     - PyQt5 (for visualization)
     
* Compile the C++ extension modules for python located in `cpp_wrappers`. You just have to execute two .bat files:

        cpp_wrappers/cpp_neighbors/build.bat
        
  and
        
        cpp_wrappers/cpp_subsampling/build.bat
        
You should now be able to train Kernel-Point Convolution models

