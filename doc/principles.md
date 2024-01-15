# Principles

## Introduction

Based on [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch), this library offers a deep learning model for points clouds classification. It is based on a Kernel Points Convolution (KPConv) algorithm, inspired by image-based convolution. The entry is made of a set of 3D-points, associated to a *feature* or *attribute*, which is here the nature of the point, in a pre-defined number of categories classification. 

The objective of the model is to be able to classify a new 3D-points set with these categories of materials or objects.

Each kernel point is associated to a weight function (called *kernel function*), applied to all points located in a radius around it. It is based on a correlation function quantifying the influence of the kernel point on the base of its distance to each point.

The number of kernel points is not constrained and nor are their positions (in particular, they are not on a grid), which makes the design adaptable to the shape of the points cloud. That is why it is called a *deformable* KPConv algorithm. The position of kernel points is determined by an optimization problem, where each point applies a repulsive force on others. The process is calibrated so that there is a small overlap between each kernel area of influence and a good space coverage.

As an intermediary result, there is a neighborhood matrix, containing for each point of the 3D-set (*N'* line) the reference to all points located in the neighborhood (*nmax* columns). Some colums can be empty, when the neighborhood contains less points.

This library has been developed by Hugues Thomas and improved by Oslandia, in the particular context of building information modeling, for indoor and outdoor point segmentation (IASBIM project).

## Indoor and outdoor available annotated data

This algorithm has been used in a particular context of segmentation of indoor and outdoor data. 4 datasets have been used to test the neural network model:
- Scannet: for indoor cluttered scenes;
    - 1513 training scenes;
    - 100 test scenes;
    - annotated with 20 semantic classes;
- S3DIS: for indoor large spaces;
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
        - 12: clutter;
    - Area-5 is used as test scene to better measure the generalization ability of the method;
- Semantic3D: for outdoor fixed scans;
    - fixed lidar scans of different outdoor scenes;
    - more than 4 billion points;
    - annotated with 8 classes : mostly covering ground, building or vegetation, with fewer object instances than in the other datasets;
- Paris-Lille-3D: for outdoor mobile scans;
    - 2 km of streets in 4 different cities;
    - 160 million points;
    - 10 semantic classes.

## Implementation details

3D datasets are too big to be segmented as a whole. The model architecture is used to segment small subclouds contained in spheres, of data variable radius. 

At training:
- Spheres are picked randomly in the scenes.

At testing:
- Spheres are regularly picked randomly in the point clouds, ensuring that each point is tested several times by different sphere locations.
- The predicted probabilities are averaged for each point.
- Datasets are colored, using three color channels as feature. A point with all features equal to 0 is equivalent to empty space.

![Schéma de fonctionnement du modèle](../_static/schema_neural_network.svg)
