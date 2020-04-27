
## Scene Segmentation on S3DIS

### Data

We consider our experiment folder is located at `XXXX/Experiments/KPConv-PyTorch`. And we use a common Data folder 
loacated at `XXXX/Data`. Therefore the relative path to the Data folder is `../../Data`.

S3DIS dataset can be downloaded <a href="https://goo.gl/forms/4SoGp4KtH1jfRqEj2">here (4.8 GB)</a>. 
Download the file named `Stanford3dDataset_v1.2.zip`, uncompress the data and move it to `../../Data/S3DIS`.

N.B. If you want to place your data anywhere else, you just have to change the variable 
`self.path` of `S3DISDataset` class ([here](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/afa18c92f00c6ed771b61cb08b285d2f93446ea4/datasets/S3DIS.py#L88)).

### Training

Simply run the following script to start the training:

        python3 training_S3DIS.py
        
Similarly to ModelNet40 training, the parameters can be modified in a configuration subclass called `S3DISConfig`, and the first run of this script might take some time to precompute dataset structures.


### Plot a logged training

When you start a new training, it is saved in a `results` folder. A dated log folder will be created, containing many information including loss values, validation metrics, model checkpoints, etc.

In `plot_convergence.py`, you will find detailed comments explaining how to choose which training log you want to plot. Follow them and then run the script :

        python3 plot_convergence.py


### Test the trained model

The test script is the same for all models (segmentation or classification). In `test_any_model.py`, you will find detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the script :

        python3 test_any_model.py
