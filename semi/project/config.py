import datetime
import time

# Configuration (update paths)
CSV_PATH = "/home/francescomarchetto/semi/DatasetSemi.csv"
IMAGES_FOLDER = "/home/francescomarchetto/semi/segmented_seeds_ordered"

# Architettura
ARCHITECTURE = 'convnext_tiny'  # 'custom', 'alexnet', 'vgg16', 'resnet50', 'efficientnet_b0', 'convnext_tiny', 'mobilenet_v3_small'
PRETRAINED = True

FILTER_ZEROS = False
MAX_POST_GERM_TIMEPOINTS = 19

# Training
EPOCHS = 12
BATCH_SIZE = 32
INITIAL_LR = 2e-5

# Class weighting
CLASS_WEIGHT_MODE = "none"
NEG_WEIGHT = 1.0
POS_WEIGHT = 15.0

USE_BALANCED_SAMPLING = False

# Threshold tuning
THRESHOLD_STRATEGY = "f1"
TARGET_PRECISION = 0.60

# Data augmentation
USE_AUGMENT = True

# Directories
HISTORY_DIR = "histories"
PLOTS_DIR = "plots"

# Dynamic paths setup
timestamp = time.strftime("%Y%m%d_%H%M%S")
variant_str = f"bs{BATCH_SIZE}_lr{INITIAL_LR}_gt{MAX_POST_GERM_TIMEPOINTS}"
MODEL_SAVE_PATH = f"/home/francescomarchetto/semi/seed_germination_model_{ARCHITECTURE}_{variant_str}_{timestamp}.pth"
REPORT_PATH = f"model_evaluation_report_{ARCHITECTURE}_{variant_str}_{timestamp}.md"