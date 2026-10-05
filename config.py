# -*- coding: utf-8 -*-
"""
Enhanced configuration with three-level hierarchy support
"""

import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------
# Path to output folder
_C.OUTPUT_DIR = ""
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Fixed random seed
_C.SEED = 1
# Number of folds to run
_C.NUM_FOLDS = 10
# Whether to show individual scores for each task
_C.SHOW_EACH_SCORES = False
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Frequency to show training epoch
_C.SHOW_FREQ = 5

# Hyperopt setting
_C.HYPER = False
_C.HYPER_COUNT = 1
_C.HYPER_REMOVE = None
# Number of hyperparameters choice to try
_C.NUM_ITERS = 20

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size, overwritten by command line argument
_C.DATA.BATCH_SIZE = 64
# Path to dataset, overwritten by command line argument
_C.DATA.DATA_PATH = '../data/'
_C.DATA.TEST_PATH = ""
_C.DATA.TRAIN_VAL_RATIO = 0.9
# Dataset name
_C.DATA.DATASET = 'ht'
# Tasks name, override by ~get_task_names~(utils.py 152) function
_C.DATA.TASK_NAME = None
# Dataset type, 'classification' or 'regression'
_C.DATA.TASK_TYPE = 'regression'
# Metric, choice from ['auc', 'prc', 'rmse', 'mae']
_C.DATA.METRIC = 'rmse'
# How to split data, 'random', 'scaffold', 'scaffoldrandom', 'molecular_weight', 'similarity', or 'noise'
_C.DATA.SPLIT_TYPE = 'random'
# Molecular weight split bins: list of MW thresholds for binning, e.g. [200, 300, 400, 500]
_C.DATA.MW_BINS = [200, 300, 400, 500]
# anti-noise rate for hiv dataset, only works when DATA.SPLIT_TYPE is 'noise'
_C.DATA.RATE = None

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Core graph model: one of ['NTN', 'GIN', 'GAT', 'GCN']
_C.MODEL.TYPE = 'NTN'
# Hidden size of HiGNN model
_C.MODEL.HID = 128
# Output size of HiGNN model, override by dataset.py 474
_C.MODEL.OUT_DIM = None
# Number of layers
_C.MODEL.DEPTH = 5
# Number of heads
_C.MODEL.SLICES = 2
# Dropout
_C.MODEL.DROPOUT = 0.2
# Feature attention
_C.MODEL.F_ATT = True
# reduction value
_C.MODEL.R = 4
# Whether to use BRICS information, if set to False, the option LOSS.CL_LOSS is set to False
_C.MODEL.BRICS = True

# -----------------------------------------------------------------------------
# Three-level hierarchy settings (NEW)
# -----------------------------------------------------------------------------
_C.MODEL.THREE_LEVEL = CN()
# Whether to enable three-level hierarchy (atom-pharmacophore-molecule)
_C.MODEL.THREE_LEVEL.ENABLE = False
# Number of attention heads for multi-view attention
_C.MODEL.THREE_LEVEL.NUM_HEADS = 8
# Number of node types (atom, pharmacophore, molecule)
_C.MODEL.THREE_LEVEL.NUM_NODE_TYPES = 3
# Number of edge types (bond, BRICS, junction, pharm-pharm, atom-pharm)
_C.MODEL.THREE_LEVEL.NUM_EDGE_TYPES = 5
# Distance threshold for pharmacophore connectivity
_C.MODEL.THREE_LEVEL.PHARM_DISTANCE_THRESHOLD = 5.0
# Number of pharmacophore convolution layers
_C.MODEL.THREE_LEVEL.PHARM_LAYERS = 2
# Junction view settings
_C.MODEL.THREE_LEVEL.USE_JUNCTION_VIEW = True
# Cross-scale fusion strategy: 'sequential', 'parallel', 'hierarchical'
_C.MODEL.THREE_LEVEL.FUSION_STRATEGY = 'hierarchical'

# -----------------------------------------------------------------------------
# Molecular descriptor settings for multi-task learning
# -----------------------------------------------------------------------------
_C.DESCRIPTOR = CN()
# Whether to enable descriptor reconstruction task
_C.DESCRIPTOR.ENABLE = False
# Descriptor types to compute and reconstruct
_C.DESCRIPTOR.TYPES = [
    'mw',           # Molecular weight
    'logp',         # Partition coefficient
    'tpsa',         # Topological polar surface area
    'hbd',          # Hydrogen bond donors
    'hba',          # Hydrogen bond acceptors  
    'nrb',          # Number of rotatable bonds
    'aromatic_rings', # Number of aromatic rings
    'heteroatoms',   # Number of heteroatoms
    'flexibility',   # Molecular flexibility index
    'complexity'     # Molecular complexity
]
# Hidden dimension for descriptor reconstructor
_C.DESCRIPTOR.HIDDEN_DIM = 64
# Dropout for descriptor reconstructor
_C.DESCRIPTOR.DROPOUT = 0.1

# -----------------------------------------------------------------------------
# Loss settings for multi-task learning
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Whether to adopt focal loss
_C.LOSS.FL_LOSS = False
# Whether to adopt molecule-fragment contrastive learning
_C.LOSS.CL_LOSS = False
# Alpha for contrastive loss
_C.LOSS.ALPHA = 0.1
# Scale logits by the inverse of the temperature
_C.LOSS.TEMPERATURE = 0.1
# Whether to use Huber loss instead of MSE for regression (more robust to outliers)
_C.LOSS.USE_HUBER = False
# Delta parameter for Huber loss
_C.LOSS.HUBER_DELTA = 1.0

# Multi-task learning weights
_C.LOSS.MULTITASK = CN()
# Weight for main task (permeability prediction)
_C.LOSS.MULTITASK.LAMBDA_MAIN = 1.0
# Weight for contrastive learning task
_C.LOSS.MULTITASK.LAMBDA_CONTRASTIVE = 0.1  
# Weight for descriptor reconstruction task
_C.LOSS.MULTITASK.LAMBDA_DESCRIPTOR = 0.5
# Whether to use adaptive weighting
_C.LOSS.MULTITASK.ADAPTIVE_WEIGHTS = False
# Temperature for adaptive weighting
_C.LOSS.MULTITASK.ADAPTIVE_TEMP = 2.0

# Three-level specific loss settings (NEW)
_C.LOSS.THREE_LEVEL = CN()
# Weight for pharmacophore-level contrastive loss
_C.LOSS.THREE_LEVEL.LAMBDA_PHARM_CONTRASTIVE = 0.1
# Weight for cross-scale alignment loss
_C.LOSS.THREE_LEVEL.LAMBDA_CROSS_SCALE = 0.05
# Whether to use hierarchical contrastive learning
_C.LOSS.THREE_LEVEL.HIERARCHICAL_CL = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Checkpoint to resume, overwritten by command line argument
_C.TRAIN.RESUME = None
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCHS = 100
# early stopping
_C.TRAIN.EARLY_STOP = -1
# Gradient clipping max norm (0 = disabled)
_C.TRAIN.GRADIENT_CLIP = 0.0

# Tensorboard
_C.TRAIN.TENSORBOARD = CN()
_C.TRAIN.TENSORBOARD.ENABLE = True

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.BASE_LR = 1e-3
# FPN Learning rate
_C.TRAIN.OPTIMIZER.FP_LR = 4e-5
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# Weight decay
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = 'reduce'
# NoamLR parameters
_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS = 2
_C.TRAIN.LR_SCHEDULER.INIT_LR = 1e-4
_C.TRAIN.LR_SCHEDULER.MAX_LR = 1e-2
_C.TRAIN.LR_SCHEDULER.FINAL_LR = 1e-4
# ReduceLRonPlateau
_C.TRAIN.LR_SCHEDULER.FACTOR = 0.7
_C.TRAIN.LR_SCHEDULER.PATIENCE = 10
_C.TRAIN.LR_SCHEDULER.MIN_LR = 1e-5


# -----------------------------------------------------------------------------
# Predict settings
# -----------------------------------------------------------------------------
_C.PREDICT = CN()
# Path to the trained checkpoint (.pth). Empty string = auto-resolve from OUTPUT_DIR.
_C.PREDICT.CKPT_PATH = ''
# Path to the independent test CSV (not seen during training).
# Empty string = auto-resolve to <DATA.DATA_PATH>/raw/<DATA.DATASET>_test.csv
_C.PREDICT.TEST_CSV = ''
# Directory where prediction results are saved.
# Empty string = auto-resolve to <OUTPUT_DIR>/predictions/
_C.PREDICT.OUTPUT_DIR = ''
# Batch size for inference (0 = fall back to DATA.BATCH_SIZE)
_C.PREDICT.BATCH_SIZE = 0
# Device: 'auto', 'cpu', or 'cuda'
_C.PREDICT.DEVICE = 'auto'
# Whether to save a per-sample CSV with predictions (and true labels if available)
_C.PREDICT.SAVE_PREDICTIONS = True
# Whether to save a metrics summary CSV (skipped automatically if no labels in test file)
_C.PREDICT.SAVE_METRICS = True


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(cfg, args):
    _update_config_from_file(cfg, args.cfg)

    cfg.defrost()
    if args.opts:
        cfg.merge_from_list(args.opts)
    # merge from specific arguments
    if args.batch_size:
        cfg.DATA.BATCH_SIZE = args.batch_size
    if getattr(args, 'model_type', None):
        cfg.MODEL.TYPE = str(args.model_type).upper()
    if args.lr_scheduler:
        cfg.TRAIN.LR_SCHEDULER.TYPE = args.lr_scheduler
    if args.resume:
        cfg.TRAIN.RESUME = args.resume
    if args.tag:
        cfg.TAG = args.tag
    if args.eval:
        cfg.EVAL_MODE = True

    # output folder
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.TAG)

    cfg.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    update_config(cfg, args)

    return cfg