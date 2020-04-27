
## Object Part Segmentation on ShapeNetPart

### Data

ShapeNetPart dataset can be downloaded <a href="https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip">here (635 MB)</a>. Uncompress the folder and move it to `Data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0`.

### Training

Simply run the following script to start the training:

        python3 training_ShapeNetPart.py
        
Similarly to ModelNet40 training, the parameters can be modified in a configuration subclass called `ShapeNetPartConfig`, and the first run of this script might take some time to precompute dataset structures.
       
### Plot a logged training

When you start a new training, it is saved in a `results` folder. A dated log folder will be created, containing many information including loss values, validation metrics, model snapshots, etc.

In `plot_convergence.py`, you will find detailed comments explaining how to choose which training log you want to plot. Follow them and then run the script :

        python3 plot_convergence.py


### Test the trained model

The test script is the same for all models (segmentation or classification). In `test_any_model.py`, you will find detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the script :

        python3 test_any_model.py
