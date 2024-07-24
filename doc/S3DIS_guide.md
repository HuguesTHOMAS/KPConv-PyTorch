
## Scene Segmentation on S3DIS

S3DIS is used to train models on indoor large spaces.

### Data

We consider our experiment folder is located at `XXXX/Experiments/KPConv-PyTorch`. And we use a common Data folder
loacated at `XXXX/Data`. Therefore the relative path to the Data folder is `../../Data`.

S3DIS dataset can be downloaded <a href="https://goo.gl/forms/4SoGp4KtH1jfRqEj2">here (4.8 GB)</a>.
Download the file named `Stanford3dDataset_v1.2.zip`, uncompress the data and move it to `../../data/S3DIS`.

> If you want to place your data anywhere else, you just have to change the variable `self.path` of `S3DISDataset` class ([here](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/afa18c92f00c6ed771b61cb08b285d2f93446ea4/datasets/S3DIS.py#L88)).

S3DIS dataset is documented [here](http://buildingparser.stanford.edu/dataset.html). It contains:
- 6 large-scale indoor areas, 3 buildings, 273 million points;
- annotated with 13 semantic classes:
    - 0: ceiling;
    - 1: floor;
    - 2: wall;
    - 3: beam;
    - 4: column;
    - 5: window;
    - 6: door;
    - 7: chair;
    - 8: table;
    - 9: bookcase;
    - 10: sofa;
    - 11: board;
    - 12: clutter.

The dataset Area-5 is used as test scene to better measure the generalization ability of the method.

When you use a S3DIS dataset, you must check that the folder contains a tree architecture, having the following organisation, for each sample area $x$ and each room of type $y$.

```bash
Area_x
├── Annotations
│   ├── cat-z_m.txt
└── room-type-y_n.txt
```

- `room-type-y_n.txt` is the file containing all (unlabeled) data points describing the room. $y$ is a user-defined room type.

- `cat-s_m.txt` is a file containing the labeled data points corresponding to the $z$ semantic categories (among the 13 available).

There are $M$ `cat-z_m.txt` files, each one corresponding to one labeled object and $m$  varying from 1 to $M$.

### Preprocessing

```bash
kpconv preprocess -c ./config_S3DIS.yml -d ~/data/S3DIS
```

### Training

Simply run the following script to start the training:
```bash
mkdir kpconv_trained_models
kpconv train -s S3DIS -c ./config_S3DIS.yml -d ~/data/S3DIS -o ~/data/kpconv_trained_models
```
The `kpconv_trained_models` folder will be the parent folder of the log folder containing the trained model, which will have the following form: `Log_YYYY-MM-DD_HH-MM-SS`, the horodate corresponding to the launching moment.

To restart the training of an already trained model, at the next iteration, do the following:

```bash
kpconv train -s S3DIS -c ./config_S3DIS.yml -d ~/data/S3DIS -l ~/data/kpconv_trained_models/Log_YYYY-MM-DD_HH-MM-SS
```

Similarly to ModelNet40 training, the parameters can be modified in a configuration subclass called `S3DISConfig`, and the first run of this script might take some time to precompute dataset structures.

### Plot a logged training

When you start a new training, it is saved in a `results` folder. A dated log folder will be created, containing many information including loss values, validation metrics, model checkpoints, etc.

In `plot_convergence.py`, you will find detailed comments explaining how to choose which training log you want to plot. Follow them and then run the script:

```bash
kpconv plotconv -c ./config_S3DIS.yml -d ~/data/S3DIS -l ~/data/kpconv_trained_models/Log_YYYY-MM-DD_HH-MM-SS
```

### Test the trained model

The test script includes the preprocessing of the entry file. It is the same for all models (segmentation or classification). It contains detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the following scripts.

For any file `~/data/cloud.xyz`:
```bash
kpconv test -s S3DIS -c ./config_S3DIS.yml -d ~/data/S3DIS -f ~/data/cloud.xyz -l ~/data/kpconv_trained_models/Log_YYYY-MM-DD_HH-MM-SS
```

For the validation file `Area_5.ply` of the S3DIS dataset used to train the model:
```bash
kpconv test -s S3DIS -c ./config_S3DIS.yml -d ~/data/S3DIS -l ~/data/kpconv_trained_models/Log_YYYY-MM-DD_HH-MM-SS
```

You will see the performance (on the subsampled input clouds) increase as the test goes on.

```bash
Confusion on sub clouds
65.08 | 92.11 98.40 81.83  0.00 18.71 55.41 68.65 90.93 79.79 74.83 65.31 63.41 56.62
```

After a few minutes, the script will reproject the results form the subsampled input clouds to the real data and get you the real score

```bash
Reproject Vote #9
Done in 2.6 s

Confusion on full clouds
Done in 2.1 s

--------------------------------------------------------------------------------------
65.38 | 92.62 98.39 81.77  0.00 18.87 57.80 67.93 91.52 80.27 74.24 66.14 64.01 56.42
--------------------------------------------------------------------------------------
```

The test script creates a folder `test/name-of-your-log`, where it saves the predictions, potentials, and probabilities per class. You can load them with CloudCompare for visualization.

## S3DIS Pretrained Models

### Models

We provide pretrained weights for S3DIS dataset. The raw weights come with a parameter file describing the architecture and network hyperparameters. The code can thus load the network automatically.


| Name (link) | KPConv Type | Description | Score |
|:-------------|:-------------:|:-----|:-----:|
| [Light_KPFCNN](https://drive.google.com/file/d/14sz0hdObzsf_exxInXdOIbnUTe0foOOz/view?usp=sharing) | rigid | A network with small `in_radius` for light GPU consumption (~8GB) | 65.4% |
| [Heavy_KPFCNN](https://drive.google.com/file/d/1ySQq3SRBgk2Vt5Bvj-0N7jDPi0QTPZiZ/view?usp=sharing) | rigid | A network with better performances but needing bigger GPU (>18GB). | 66.4% |
| [Deform_KPFCNN](https://drive.google.com/file/d/1ObGr2Srfj0f7Bd3bBbuQzxtjf0ULbpSA/view?usp=sharing) | deform | Deformable convolution network needing big GPU (>20GB). | 67.3% |
| [Deform_Light_KPFCNN](https://drive.google.com/file/d/1gZfv6q6lUT9STFh7Fk4qVa5IVTgwmWIr/view?usp=sharing) | deform | Lighter version of the deformable architecture (~8GB). | 66.7% |



### Instructions

1. Unzip and place the folder in your 'results' folder.

2. In the test script `test_any_model.py`, set the variable `chosen_log` to the path were you placed the folder.

3. Run the test script

        kpconv test -d <path-of-your-dataset> -l <path-of-your-model-log>

4. You will see the performance (on the subsampled input clouds) increase as the test goes on.

        Confusion on sub clouds
        65.08 | 92.11 98.40 81.83  0.00 18.71 55.41 68.65 90.93 79.79 74.83 65.31 63.41 56.62


5. After a few minutes, the script will reproject the results form the subsampled input clouds to the real data and get you the real score

        Reproject Vote #9
        Done in 2.6 s

        Confusion on full clouds
        Done in 2.1 s

        --------------------------------------------------------------------------------------
        65.38 | 92.62 98.39 81.77  0.00 18.87 57.80 67.93 91.52 80.27 74.24 66.14 64.01 56.42
        --------------------------------------------------------------------------------------

6. The test script creates a folder `test/name-of-your-log`, where it saves the predictions, potentials, and probabilities per class. You can load them with CloudCompare for visualization.
