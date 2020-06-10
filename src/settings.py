import os
import platform
import transformations

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

WEBCAMT_DOMAINS = [403, 410, 511, 551]
#WEBCAMT_DOMAINS = [410]
#WEBCAMT_DOMAINS = [164, 166, 170, 173, 181, 253, 398, 403, 410, 511, 551, 572, 691, 846, 928]
NUM_WEBCAMT_DOMAINS = len(WEBCAMT_DOMAINS)
WEBCAMT_NEW_SHAPE = (120, 176)
WEBCAMT_SHAPE = (240, 352)
TEMPORAL = True
USE_GAUSSIAN = True
SEQUENCE_SIZE = 4
LOAD_MULTIPLE_FILES = True
LOAD_DATA_AUGMENTATION = False
VALIDATION_TEST_RATIO = 0.3
PREFIX_DATA = 'first_mask'
PREFIX_DENSITIES = 'proportional_mask'

UCSPEDS_DOMAINS = ['vidd', 'vidf']
DATASET = 'webcamt'
UCSPEDS_NEW_SHAPE = (158, 238)
USE_MASK = True
STORE_MASK = True

def get_new_shape():
    if DATASET == 'webcamt':
        return WEBCAMT_NEW_SHAPE
    elif DATASET == 'ucspeds':
        return UCSPEDS_NEW_SHAPE

TRANSFORMS = []
hor_sym = lambda matrix : transformations.transform_matrix_channels(matrix, transformations.symmetric, 90)
TRANSFORMS.append([hor_sym, hor_sym])