import os
import platform

def get_dataset_directory():
    """
    Auxiliary function that returns the dataset directory which is different according to the platform:
    Windows or Linux
    """
    dataset_relative = "../../WebCamT/data"
    if platform.system() == "Windows":
        return dataset_relative
    else:
        abs_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(abs_path, dataset_relative)

DATASET_DIRECTORY = get_dataset_directory()
USE_BIG_BUS = False
NUM_DATASETS = 3
DATASETS = [403, 410, 511]
IMAGE_SHAPE = (60,88)
