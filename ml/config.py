"""
Configuration for keypoint estimation training
"""
# Data paths
DATA_PATH = "/home/spjor/PycharmProjects/dat255-pose-estimation/data/coco"
MAX_SAMPLES = None  # dataset size for training/validation, None for full dataset

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 3e-6

# Model hyperparameters
IMG_SIZE = 256
HEATMAP_SIZE = 64
NUM_KEYPOINTS = 17

# Training settings
NUM_WORKERS = 4
LR_SCHEDULER_STEP_SIZE = 5
LR_SCHEDULER_GAMMA = 0.5

# Checkpoint settings
CHECKPOINT_DIR = '/home/spjor/PycharmProjects/dat255-pose-estimation/out/checkpoints'
MODEL_SAVE_NAME = 'best_keypoint_model.pth'
# Visualization settings
VIS_DIR = "/home/spjor/PycharmProjects/dat255-pose-estimation/out/vis"
VIS_INTERVAL = 1  # Save 1 visualization image every X epochs