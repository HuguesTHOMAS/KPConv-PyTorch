

## Scene Segmentation on S3DIS

### Data

S3DIS dataset can be downloaded <a href="https://goo.gl/forms/4SoGp4KtH1jfRqEj2">here (4.8 GB)</a>. Download the file named `Stanford3dDataset_v1.2.zip`, uncompress the folder and move it to `Data/S3DIS/Stanford3dDataset_v1.2`.

### Training

Simply run the following script to start the training:

        python3 training_S3DIS.py
        
Similarly to ModelNet40 training, the parameters can be modified in a configuration subclass called `S3DISConfig`, and the first run of this script might take some time to precompute dataset structures.


## Scene Segmentation on Scannet

Incoming

## Scene Segmentation on Semantic3D

### Data

Semantic3D dataset can be found <a href="http://www.semantic3d.net/view_dbase.php?chl=2">here</a>. Download and unzip every point cloud as ascii files and place them in a folder called `Data/Semantic3D/original_data`. You also have to download and unzip the groundthruth labels as ascii files in the same folder


### Training

Simply run the following script to start the training:

        python3 training_Semantic3D.py
        
Similarly to ModelNet40 training, the parameters can be modified in a configuration subclass called `Semantic3DConfig`, and the first run of this script might take some time to precompute dataset structures.


## Scene Segmentation on NPM3D

Incoming

       
## Plot and test trained models

### Plot a logged training

When you start a new training, it is saved in a `results` folder. A dated log folder will be created, containing many information including loss values, validation metrics, model snapshots, etc.

In `plot_convergence.py`, you will find detailed comments explaining how to choose which training log you want to plot. Follow them and then run the script :

        python3 plot_convergence.py


### Test the trained model

The test script is the same for all models (segmentation or classification). In `test_any_model.py`, you will find detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the script :

        python3 test_any_model.py
