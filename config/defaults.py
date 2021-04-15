from yacs.config import CfgNode as CN


cfg = CN()

cfg.MODEL = CN()

# TODO: Fill in right values for training
cfg.MODEL.IN_CHANNELS = 1
cfg.MODEL.OUT_CHANNELS = 2

# cfg.MODEL.THRESHOLD = 0.5
# cfg.MODEL.NUM_CLASSES = 21
# # Hard negative mining
# cfg.MODEL.NEG_POS_RATIO = 3
# cfg.MODEL.CENTER_VARIANCE = 0.1
# cfg.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
# cfg.MODEL.BACKBONE = CN()
# cfg.MODEL.BACKBONE.NAME = 'vgg'
# cfg.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
# cfg.MODEL.BACKBONE.PRETRAINED = True
# cfg.MODEL.BACKBONE.INPUT_CHANNELS = 3


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Image size
cfg.INPUT.IMAGE_SIZE = [300, 300]
# Values to be used for image normalization, RGB layout
#cfg.INPUT.PIXEL_MEAN = [123.675, 116.280, 103.530]
#cfg.INPUT.PIXEL_STD = [1, 1, 1]

# TODO: Find the correct valeus for this

# -----------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------
cfg.PREPROCESSING = CN()
# Base transformation
cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE = CN()
cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.ENABLE = True
cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.SIZE = 384
cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.PROBABILITY = 1

cfg.PREPROCESSING.HORIZONTALFLIP = CN()
cfg.PREPROCESSING.HORIZONTALFLIP.ENABLE = True
cfg.PREPROCESSING.HORIZONTALFLIP.PROBABILITY = 1

cfg.PREPROCESSING.GAUSSIANSMOOTH = CN()
cfg.PREPROCESSING.GAUSSIANSMOOTH.ENABLE = False
cfg.PREPROCESSING.GAUSSIANSMOOTH.BLURLIMIT = 51 #kan også være en liste med to tall [tall1, tall2]
cfg.PREPROCESSING.GAUSSIANSMOOTH.SIGMALIMIT = 0
cfg.PREPROCESSING.GAUSSIANSMOOTH.PROBABILITY = 1



cfg.PREPROCESSING.IMAGE_FILTERING = True


# Transformations in data augmentation
cfg.PREPROCESSING.DATA_AUGMENTATION = CN()

# Random sample crop
cfg.PREPROCESSING.DATA_AUGMENTATION.RANDOM_SAMPLE_CROP = CN()
cfg.PREPROCESSING.DATA_AUGMENTATION.RANDOM_SAMPLE_CROP.ENABLED = False
# TODO: Add hyperparams here

# Random rotation
cfg.PREPROCESSING.DATA_AUGMENTATION.RANDOM_ROTATION = CN()
cfg.PREPROCESSING.DATA_AUGMENTATION.RANDOM_ROTATION.ENABLED = False
# TODO: Add hyperparams here

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()
# List of the dataset names for training, as present in pathscfgatalog.py
cfg.DATASETS.TRAIN_IMAGES = '../../../../work/datasets/medical_project/CAMUS'
cfg.DATASETS.GT_IMAGES = ''
# # List of the dataset names for testing, as present in pathscfgatalog.py
# cfg.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
cfg.DATA_LOADER = CN()
# Number of data loading threads
cfg.DATA_LOADER.NUM_WORKERS = 4
cfg.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# train configs
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.MAX_MINUTES = 600
# cfg.SOLVER.GAMMA = 0.1
# cfg.SOLVER.BATCH_SIZE = 32
cfg.SOLVER.LR = 1e-2
# cfg.SOLVER.MOMENTUM = 0.9
# cfg.SOLVER.WEIGHT_DECAY = 5e-4

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #

# TODO: Fill in correct values here: 
cfg.TEST = CN()
# cfg.TEST.NMS_THRESHOLD = 0.45
# cfg.TEST.CONFIDENCE_THRESHOLD = 0.01
# cfg.TEST.MAX_PER_CLASS = -1
# cfg.TEST.MAX_PER_IMAGE = 100
cfg.TEST.BATCH_SIZE = 12
cfg.TEST.NUM_EPOCHS = 50

cfg.EVAL_STEP = 500 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.MODEL_SAVE_STEP = 50 # Save checkpoint every save_step
cfg.LOG_STEP = 10 # Print logs every log_stepPrint logs every log_step

cfg.OUTPUT_DIR = "outputs"
# cfg.DATASET_DIR = "datasets"

# ---------------------------------------------------------------------------- #
# Logger options
# ---------------------------------------------------------------------------- #
cfg.LOGGER = CN()
cfg.LOGGER.NAME = 'UNET'
cfg.LOGGER.SAVE_DIR = False
cfg.LOGGER.VISUAL_DEBUG = True