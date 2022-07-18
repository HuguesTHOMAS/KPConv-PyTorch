

## Visualize kernel deformations

### Intructions

In order to visualize features you need a dataset and a pretrained model that uses deformable KPConv.

To start this visualization run the script:

        python3 visualize_deformations.py

### Details

The visualization script runs the model on a batch of test examples (forward pass), and then show these 
examples in an interactive window. Here is a list of all keyboard shortcuts:

- 'b' / 'n': smaller or larger point size.
- 'g' / 'h': previous or next example in current batch.
- 'k': switch between the rigid kernel (original kernel points positions) and the deformed kernel (position of the 
kernel points after shift are applied)
- 'z': Switch between the points displayed (input points, current layer points or both).
- '0': Saves the example and deformed kernel as ply files.
- mouse left click: select a point and show kernel at its location.
- exit window: compute next batch.
