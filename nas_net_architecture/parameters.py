import os

IMG_SIZE = 64

HIDDEN_STATE_OPERATIONS = [
    "identity",
    "1x3 then 3x1 convolution",
    "1x7 then 7x1 convolution",
    "3x3 dilated convolution",
    "3x3 average pooling",
    "3x3 max pooling",
    "5x5 max pooling",
    "7x7 max pooling",
    "1x1 convolution",
    "3x3 convolution",
    "3x3 depthwise-separable conv",
    "5x5 depthwise-seperable conv",
    "7x7 depthwise-separable conv"
]

EPISODES = 800

B = 2  # In Zoph et al. (2018), B is the number of child models sampled and trained in parallel before updating the controller (policy network).
MOTIF_REPETITIONS = 3

POLICY_NETWORK_UNITS = 35 # Zoph and Le (2017)
POLICY_NETWORK_LEARNING_RATE = 6e-4  # Zoph and Le (2017)
POLICY_TEMPERATURE_DECAY = 0.995  # Exponential decay per episode
# Initial baseline in case reward=accuracy (mock training)
# POLICY_NETWORK_INITIAL_BASELINE = 0.65
# Initial baseline in case reward is the cube of the best validation accuracy (as in Zoph and Le 2017)
POLICY_NETWORK_INITIAL_BASELINE = 0.65**3

ACTOR_TRINING_BATCH_SIZE=7

CHILD_NETWORK_EPOCHS = 50 # Zoph and Le (2017)
CHILD_NETWORK_LEARNING_RATE = 0.1 # Zoph and Le (2017)
CHILD_NETWORK_LEARNING_RATE_MOMENTUM = 0.9
CHILD_NETWORK_LEARNING_RATE_DECAY = 1e-4
CHILD_NETWORK_BATCH_SIZE = 32

SEARCH_DATASET_FRACTION = 0.2

DATASET_DIR = os.environ.get('NAS_DATADIR', '/opt/ml/input/data/training') # Default to SageMaker input path
OUTPUT_DIR = os.environ.get('NAS_OUTPUTDIR', '/opt/ml/model/') # Default to SageMaker output path