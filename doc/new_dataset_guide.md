

## Creating your own dataset

### Overview of the pipeline

The training script initiates a bunch of variables and classes before starting the training on a dataset. Here are the 
initialization steps:

* Create an instance of the `Config` class. This instance will hold all the parameters defining the network.

* Create an instance of your dataset class. This instance will handle the data, and the input pipeline. **This is the 
class you have to implement to train our network on your own data**.

* Load the input point cloud in memory. Most datasets will fit in a 32GB RAM computer. If you don't have enough memory
for your dataset, you will have to redesign the input pipeline.

* Initialize the tensorflow input pipeline, which is a `tf.dataset` object that will create and feed the input batches
to the network.

* Create an instance of the network model class. This class contains the tensorflow operations defining the network.

* Create an instance of our generic `ModelTrainer` class. This class handles the training of the model

Then the training can start.


### The dataset class

This class has several roles. First this is where you define your dataset parameters (class names, data path, nature 
of the data...). Then this class will hold the point clouds loaded in memory. Eventually, it also defines the 
Tensorflow input pipeline. For efficiency, our implementation uses a parallel input queue, feeding batches to the 
network.

Here we give you a description of each essential method that need to be implemented in your new dataset class. For more
details, follow the implementation of the current datasets, which contains a lot of indications as comments.


* The **\_\_init\_\_** method: Here you have to define the parameters of your dataset. Notice that your dataset class 
has to be a child of the common `Dataset` class, where generic methods are implemented. Their are a few thing that has 
to be defined here:
    - The labels: define a dictionary `self.label_to_names`, call the `self.init_labels()` method, and define which 
    label should be ignored in `self.ignored_labels`.
    - The network model: the type of model that will be used on this dataset ("classification", "segmentation", 
    "multi_segmentation" or "cloud_segmentation").
    - The number of CPU threads used in the parallel input queue.
    - Data paths and splits: you can manage your data as you wish, these variables are only used in methods that you 
    will implement, so you do not have to follow exactly the notations of the other dataset classes.
    
    
* The **load_subsampled_clouds** method: Here you load your data in memory. Depending on your dataset (if this is a 
classification or segmentation task, 3D scenes or 3D models) you will not have to load the same variables. Just follow 
the implementation of the existing datasets.


* The **get_batch_gen** method: This method should return a python generator. This will be the base generator for the
`tf.dataset` object. It is called in the generic `self.init_input_pipeline` or `self.init_test_input_pipeline` methods. 
Along with the generator, it also has to return the generated types and shapes. You can redesign the generators or used 
the ones we implemented. The generator returns np.arrays, but from this point of the pipeline, they will be converted 
to tensorflow tensors.


* The **get_tf_mapping** method: This method return a mapping function that takes the generated batches and creates all 
the variables for the network. Remember that from this point we are defining a tensorflow graph of operations. There is 
not much to implement here as most of the work is done by two generic function `self.tf_augment_input` and 
`self.tf_xxxxxxxxx_inputs` where xxxxxxxxx can be "classification" of "segmentation" depending on the task. The only 
important thing to do here is to define the features that will be fed to the network.


### The training script and configuration class

In the training script you have to create a class that inherits from the `Config` class. This is where you will define 
all the network parameters by overwriting the attributes



