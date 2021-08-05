

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

        python3 test_any_model.py

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

