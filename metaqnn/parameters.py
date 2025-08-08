import os


EPSILON_TO_NUM_MODELS_MAP = {
    1.0: 150, # 1500
    0.9: 10, # 100
    0.8: 10, # 100
    0.7: 10, # 100
    0.6: 15, # 150
    0.5: 15, # 150
    0.4: 15, # 150
    0.3: 15, # 150,
    0.2: 15, # 150
    0.1: 15 # 150
}

""" Agent parameters"""
RL_AGENT_Q_LEARNING_RATE = 0.01  # Baker et al. (2017, p. 5) uses 0.01 - It's the Q-learning rate of the RL Q-learning agent
DISCOUNT_FACTOR = 1 # Baker et al. (2017, p. 5) uses 1
DEFAULT_Q_VALUE = 0.5
INITIAL_R_SIZE = 64.0  # Initial representation size based on input size (64x64)
SAMPLED_MODELS_FROM_EXPERIENCE_REPLAY = 10 # Number of models to sample from experience replay for Q-value updates (Baker et al. (2017, p. 6) uses 100)
LAYERS_DEPTH_LIMIT = 12 # As per Baker et al. (2017) - Maximum depth of the architecture

""" Dataset parameters """
DATASET_DIR = os.environ.get('META_MQQ_DATADIR', '/opt/ml/input/data/training')
NUM_CLASSES = 2

""" Architectures training parameters """
IMG_SIZE = (64, 64)
LEARNING_RATE = 0.001  # Learning rate for the optimizer
EPOCHS = 15 # Number of epochs for training, Baker et al. (2017, p. 6) uses 20 epochs, here I use 15 for quicker testing
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 6  # Patience for early stopping based on validation accuracy

""" Output results """
OUTPUT_DIR = os.environ.get('NAS_OUTPUTDIR', '/opt/ml/model/') # Default to SageMaker output path