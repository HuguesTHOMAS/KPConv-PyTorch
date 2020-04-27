

## Visualize learned features 

### Intructions

In order to visualize features you need a dataset and a pretrained model. You can use one of our pretrained models 
provided in the [pretrained models guide](./pretrained_models_guide.md), and the corresponding dataset.

To start this visualization run the script:

        python3 visualize_features.py

### Details

The visualization script has to main parts, separated in two different methods of the visualizer class in 
`visualizer.py`. 

* In the first part, implemented in the method `top_relu_activations`, the script runs the model on test examples 
(forward pass). At the chosen Relu layer, you have N output features that are going to be visualized. For each feature,
the script keeps the top 5 examples that activated it the most, and saves them in a `visu` folder. 

* In the second part, implemented in the method `top_relu_activations`, the script just shows the saved examples for 
each feature with the level of activation as color. You can navigate through examples with keys 'g' and 'h'.

N.B. This second part of the code can be started without doing the first part again if the top examples have already 
been computed. See details in the code. Alternatively you can visualize the saved example with a point cloud software 
like CloudCompare.


## Visualize kernel deformations

### Intructions

In order to visualize features you need a dataset and a pretrained model that uses deformable KPConv. You can use our 
NPM3D pretrained model provided in the [pretrained models guide](./pretrained_models_guide.md).

To start this visualization run the script:

        python3 visualize_deformations.py

### Details

The visualization script runs the model runs the model on a batch of test examples (forward pass), and then show these 
examples in an interactive window. Here is a list of all keyborad shortcuts:

- 'b' / 'n': smaller or larger point size.
- 'g' / 'h': previous or next example in current batch.
- 'k': switch between the rigid kenrel (original kernel points positions) and the deformed kernel (position of the 
kernel points after shift are applied)
- 'z': Switch between the points displayed (input points, current layer points or both).
- '0': Saves the example and deformed kernel as ply files.
- mouse left click: select a point and show kernel at its location.
- exit window: compute next batch.


## visualize Effective Receptive Fields 

### Intructions

In order to visualize features you need a dataset and a pretrained model. You can use one of our pretrained models 
provided in the [pretrained models guide](./pretrained_models_guide.md), and the corresponding dataset.

To start this visualization run the script:

        python3 visualize_ERFs.py
        
**Warning: This cript currently only works on the following datasets: NPM3D, Semantic3D, S3DIS, Scannet**

### Details

The visualization script show the Effective receptive fields of a network layer at one location. If you chose another 
location (with left click), it has to rerun the model on the whole input point cloud to get new gradient values. Here a 
list of all keyborad shortcuts:

- 'b' / 'n': smaller or larger point size.
- 'g' / 'h': lower or higher ceiling limit. A functionality that remove points from the ceiling. Very handy for indoor 
point clouds.
- 'z': Switch between the points displayed (input points, current layer points or both).
- 'x': Go to the next input point cloud.
- '0': Saves the input point cloud with ERF values and the center point used as origin of the ERF.
- mouse left click: select a point and show ERF at its location.
- exit window: End script.