
## Scene Segmentation on SemanticKitti

### Data

We consider our experiment folder is located at `XXXX/Experiments/KPConv-PyTorch`. And we use a common Data folder 
loacated at `XXXX/Data`. Therefore the relative path to the Data folder is `../../Data`.

SemanticKitti dataset can be downloaded <a href="http://semantic-kitti.org/dataset.html#download">here (80 GB)</a>. 
Download the three file named:
 * [`data_odometry_velodyne.zip` (80 GB)](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip)
 * [`data_odometry_calib.zip` (1 MB)](http://www.cvlibs.net/download.php?file=data_odometry_calib.zip)
 * [`data_odometry_labels.zip` (179 MB)](http://semantic-kitti.org/assets/data_odometry_labels.zip)

uncompress the data and move it to `../../Data/SemanticKitti`.

You also need to download the files 
[`semantic-kitti-all.yaml`](https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti-all.yaml)
and
[`semantic-kitti.yaml`](https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml).
Place them in your `../../Data/SemanticKitti` folder.

N.B. If you want to place your data anywhere else, you just have to change the variable 
`self.path` of `SemanticKittiDataset` class ([here](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/c32e6ce94ed34a3dd9584f98d8dc0be02535dfb4/datasets/SemanticKitti.py#L65)).

### Training

Simply run the following script to start the training:

        python3 training_SemanticKitti.py
        
Similarly to ModelNet40 training, the parameters can be modified in a configuration subclass called `SemanticKittiConfig`, and the first run of this script might take some time to precompute dataset structures.


### Plot a logged training

When you start a new training, it is saved in a `results` folder. A dated log folder will be created, containing many information including loss values, validation metrics, model checkpoints, etc.

In `plot_convergence.py`, you will find detailed comments explaining how to choose which training log you want to plot. Follow them and then run the script :

        python3 plot_convergence.py


### Test the trained model

The test script is the same for all models (segmentation or classification). In `test_any_model.py`, you will find detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the script :

        python3 test_any_model.py
