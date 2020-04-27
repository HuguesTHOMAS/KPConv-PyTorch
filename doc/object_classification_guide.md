
## Object classification on ModelNet40

### Data

We consider our experiment folder is located at `XXXX/Experiments/KPConv-PyTorch`. And we use a common Data folder 
loacated at `XXXX/Data`. Therefore the relative path to the Data folder is `../../Data`.

Regularly sampled clouds from ModelNet40 dataset can be downloaded 
<a href="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip">here (1.6 GB)</a>. 
Uncompress the data and move it inside the folder `../../Data/ModelNet40`.

N.B. If you want to place your data anywhere else, you just have to change the variable 
`self.path` of `ModelNet40Dataset` class ([here](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/e9d328135c0a3818ee0cf1bb5bb63434ce15c22e/datasets/ModelNet40.py#L113)).


### Training a model

Simply run the following script to start the training:

        python3 training_ModelNet40.py
        
This file contains a configuration subclass `ModelNet40Config`, inherited from the general configuration class `Config` defined in `utils/config.py`. The value of every parameter can be modified in the subclass. The first run of this script will precompute structures for the dataset which might take some time.
        
### Plot a logged training

When you start a new training, it is saved in a `results` folder. A dated log folder will be created, containing many information including loss values, validation metrics, model checkpoints, etc.

In `plot_convergence.py`, you will find detailed comments explaining how to choose which training log you want to plot. Follow them and then run the script :

        python3 plot_convergence.py


### Test the trained model

The test script is the same for all models (segmentation or classification). In `test_any_model.py`, you will find detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the script :

        python3 test_any_model.py
