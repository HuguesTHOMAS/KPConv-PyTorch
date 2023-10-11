
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

        kpconv train -d <path-of-your-dataset> -l <path-of-your-model-log>

Similarly to ModelNet40 training, the parameters can be modified in a configuration subclass called `S3DISConfig`, and the first run of this script might take some time to precompute dataset structures.

### Plot a logged training

When you start a new training, it is saved in a `results` folder. A dated log folder will be created, containing many information including loss values, validation metrics, model checkpoints, etc.

In `plot_convergence.py`, you will find detailed comments explaining how to choose which training log you want to plot. Follow them and then run the script :

        kpconv plotconv -d <path-of-your-dataset> -l <path-of-your-model-log>

### Test the trained model

The test script is the same for all models (segmentation or classification). It contains detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the script:

        kpconv test -d <path-of-your-dataset> -l <path-of-your-model-log>

You will see the performance (on the subsampled input clouds) increase as the test goes on.

        Confusion on sub clouds
        65.08 | 92.11 98.40 81.83  0.00 18.71 55.41 68.65 90.93 79.79 74.83 65.31 63.41 56.62


After a few minutes, the script will reproject the results form the subsampled input clouds to the real data and get you the real score

        Reproject Vote #9
        Done in 2.6 s

        Confusion on full clouds
        Done in 2.1 s

        --------------------------------------------------------------------------------------
        65.38 | 92.62 98.39 81.77  0.00 18.87 57.80 67.93 91.52 80.27 74.24 66.14 64.01 56.42
        --------------------------------------------------------------------------------------

 The test script creates a folder `test/name-of-your-log`, where it saves the predictions, potentials, and probabilities per class. You can load them with CloudCompare for visualization.
