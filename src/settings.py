"""
Module that defines the main settings.
"""

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
UCSPEDS_PREPROCESSED_DIRECTORY = 'UCSPeds_spacing5'
WEBCAMT_PREPROCESSED_DIRECTORY = 'WebCamT_complete'
USE_BIG_BUS = False

WEBCAMT_DOMAINS = [511, 551, 691, 846]
NUM_WEBCAMT_DOMAINS = len(WEBCAMT_DOMAINS)
WEBCAMT_NEW_SHAPE = (120, 176)
WEBCAMT_SHAPE = (240, 352)
TEMPORAL = False
USE_GAUSSIAN = True
SEQUENCE_SIZE = 3
LOAD_DATA_AUGMENTATION = False
VALIDATION_TEST_RATIO = 0.3
PREFIX_DATA = 'const4'
PREFIX_DENSITIES = 'const4'

UCSPEDS_DOMAINS = ['vidd', 'vidf']
DATASET = 'webcamt'
UCSPEDS_NEW_SHAPE = (158, 238)
USE_MASK = False


def get_new_shape():
    if DATASET == 'webcamt':
        return WEBCAMT_NEW_SHAPE
    elif DATASET == 'ucspeds':
        return UCSPEDS_NEW_SHAPE
