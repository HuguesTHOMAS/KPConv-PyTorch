"""
ModelNet40Sampler class

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0903

from torch.utils.data import get_worker_info


class ModelNet40WorkerInitDebug:
    """
    Callable class that Initializes workers.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self):
        # Print workers info
        worker_info = get_worker_info()
        print(worker_info)

        # Get associated dataset
        dataset = worker_info.dataset  # the dataset copy in this worker process

        # In windows, each worker has its own copy of the dataset. In Linux, this is shared in
        # memory
        print(dataset.input_labels.__array_interface__["data"])
        print(worker_info.dataset.input_labels.__array_interface__["data"])
        print(self.dataset.input_labels.__array_interface__["data"])

        # configure the dataset to only process the split workload
