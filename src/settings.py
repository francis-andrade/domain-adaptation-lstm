import os
import platform

def get_dataset_directory():
    """
    Auxiliary function that returns the dataset directory which is different according to the platform:
    Windows or Linux
    """
    dataset_relative = "../../Dataset"
    if platform.system() == "Windows":
        return dataset_relative
    else:
        abs_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(abs_path, dataset_relative)

DATASET_DIRECTORY = get_dataset_directory()
USE_BIG_BUS = False

DATASETS = [403, 410, 511]
#DATASETS = [403]
#DATASETS = [164, 166, 170, 173, 181, 253, 398, 403, 410, 511, 551, 572, 691, 846, 928]
NUM_DATASETS = len(DATASETS)
IMAGE_NEW_SHAPE = (120, 176)
IMAGE_ORIGINAL_SHAPE = (240, 352)
TEMPORAL = False
USE_GAUSSIAN = True
SEQUENCE_SIZE = 10
LOAD_MULTIPLE_FILES = True