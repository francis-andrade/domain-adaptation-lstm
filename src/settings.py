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
UCSPEDS_PREPROCESSED_DIRECTORY = 'UCSPeds_spacing5'
WEBCAMT_PREPROCESSED_DIRECTORY = 'WebCamT_complete'
USE_BIG_BUS = False

#WEBCAMT_DOMAINS = [403, 410, 511, 551]
#WEBCAMT_DOMAINS = [403, 511, 551, 572]
WEBCAMT_DOMAINS = [691]
#WEBCAMT_DOMAINS = [164, 166, 170, 173, 181, 253, 398, 403, 410, 511, 551, 572, 691, 846, 928] 181 does not have mask
#WEBCAMT_DOMAINS = [164, 166, 170, 173, 253, 398, 403, 410, 511, 551, 572, 691, 846, 928]
NUM_WEBCAMT_DOMAINS = len(WEBCAMT_DOMAINS)
WEBCAMT_NEW_SHAPE = (120, 176)
WEBCAMT_SHAPE = (240, 352)
TEMPORAL = True
USE_GAUSSIAN = False
SEQUENCE_SIZE = 3
LOAD_DATA_AUGMENTATION = False
VALIDATION_TEST_RATIO = 0.3
PREFIX_DATA = 'proportional'
PREFIX_DENSITIES = 'proportional'

UCSPEDS_DOMAINS = ['vidd', 'vidf']
DATASET = 'webcamt'
UCSPEDS_NEW_SHAPE = (158, 238)
USE_MASK = True


def get_new_shape():
    if DATASET == 'webcamt':
        return WEBCAMT_NEW_SHAPE
    elif DATASET == 'ucspeds':
        return UCSPEDS_NEW_SHAPE
