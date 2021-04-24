from yacs.config import CfgNode as CN
import cv2


cfg = CN()

cfg.MODEL = CN()

# TODO: Fill in right values for training
cfg.MODEL.IN_CHANNELS = 1
cfg.MODEL.OUT_CHANNELS = 4
cfg.MODEL.CLASSES = [1, 2, 3]


#use of backbone models



# cfg.MODEL.THRESHOLD = 0.5
# cfg.MODEL.NUM_CLASSES = 21
# # Hard negative mining
# cfg.MODEL.NEG_POS_RATIO = 3
# cfg.MODEL.CENTER_VARIANCE = 0.1
# cfg.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
cfg.MODEL.BACKBONE = CN()
# cfg.MODEL.BACKBONE.NAME = 'vgg'
# cfg.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
# cfg.MODEL.BACKBONE.PRETRAINED = True
# cfg.MODEL.BACKBONE.INPUT_CHANNELS = 3
cfg.MODEL.BACKBONE.USE = False
cfg.MODEL.BACKBONE.NET = 'resnet18'
cfg.MODEL.BACKBONE.PRETRAINED = True
cfg.MODEL.BACKBONE.ENCODER_FREZE=False
cfg.MODEL.BACKBONE.DECODER_FILTERS = (256, 128, 64, 32, 16)
cfg.MODEL.BACKBONE.PARAMETRIC_UPSAMPLING = True
cfg.MODEL.BACKBONE.SHORTCUT_FEATURES = 'default'
cfg.MODEL.BACKBONE.DECODER_USE_BATCHNORM = True
cfg.MODEL.BACKBONE.USE_NORMALIZATION = False
cfg.MODEL.BACKBONE.NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
cfg.MODEL.BACKBONE.NORMALIZATION_STD = [0.229, 0.224, 0.225]




# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Values to be used for image normalization, RGB layout
#cfg.INPUT.PIXEL_MEAN = [123.675, 116.280, 103.530]
#cfg.INPUT.PIXEL_STD = [1, 1, 1]

# TODO: Find the correct valeus for this

# -----------------------------------------------------------------------------
# UNETSTRUCTURE
# -----------------------------------------------------------------------------
cfg.UNETSTRUCTURE = CN()
# in_channels, out_channels, kernel_size, padding
cfg.UNETSTRUCTURE.CONTRACTBLOCK = [[cfg.MODEL.IN_CHANNELS, 32, 7, 3], 
                                    [32, 64, 3, 1],
                                    [64, 128, 3, 1]]

cfg.UNETSTRUCTURE.EXPANDBLOCK = [[128, 64, 3, 1],
                                [64*2, 32, 3, 1],
                                [32*2, cfg.MODEL.OUT_CHANNELS, 3, 1]]

# -----------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------

cfg.PREPROCESSING = CN()

# Base transformation
cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE = CN()
cfg.PREPROCESSING.ISOTROPIC_PIXEL_SIZE.ENABLE = True
cfg.PREPROCESSING.RESIZE = CN()
cfg.PREPROCESSING.RESIZE.FX = 0.5
cfg.PREPROCESSING.RESIZE.FY = 0.5
cfg.PREPROCESSING.RESIZE.X = 0 #Ikke rør denne
cfg.PREPROCESSING.RESIZE.Y = 0 #Ikke denne heller!!!!!

cfg.PREPROCESSING.NORMALIZE = CN()
cfg.PREPROCESSING.NORMALIZE.ENABLE = False
cfg.PREPROCESSING.NORMALIZE.MEAN = 0.0
cfg.PREPROCESSING.NORMALIZE.STD = 0.5

cfg.PREPROCESSING.ROTATE = CN()
cfg.PREPROCESSING.ROTATE.ENABLE = False
cfg.PREPROCESSING.ROTATE.LIMIT = 15
cfg.PREPROCESSING.ROTATE.BORDER_MODE = 0 #cv2.BORDER_CONSTANT
cfg.PREPROCESSING.ROTATE.PROB = 0.5



cfg.PREPROCESSING.HORIZONTALFLIP = CN()
cfg.PREPROCESSING.HORIZONTALFLIP.ENABLE = False
cfg.PREPROCESSING.HORIZONTALFLIP.PROBABILITY = 1

cfg.PREPROCESSING.GAUSSIANSMOOTH = CN()
cfg.PREPROCESSING.GAUSSIANSMOOTH.ENABLE = False
cfg.PREPROCESSING.GAUSSIANSMOOTH.BLURLIMIT = (3,11) #kan også være en liste med to tall [tall1, tall2]
cfg.PREPROCESSING.GAUSSIANSMOOTH.SIGMALIMIT = (0,5)
cfg.PREPROCESSING.GAUSSIANSMOOTH.PROBABILITY = 1

cfg.PREPROCESSING.ELASTICDEFORM = CN()
cfg.PREPROCESSING.ELASTICDEFORM.ENABLE = False
cfg.PREPROCESSING.ELASTICDEFORM.ALPHA = 300
cfg.PREPROCESSING.ELASTICDEFORM.SIGMA = 20
cfg.PREPROCESSING.ELASTICDEFORM.ALPHA_AFFINE = 1
cfg.PREPROCESSING.ELASTICDEFORM.PROBABILITY = 0.5

cfg.PREPROCESSING.GRIDDISTORTIAN = CN()
cfg.PREPROCESSING.GRIDDISTORTIAN.ENABLE = False
cfg.PREPROCESSING.GRIDDISTORTIAN.MUM_STEPS = 3
cfg.PREPROCESSING.GRIDDISTORTIAN.DISTORT_LIMIT = 0.2
cfg.PREPROCESSING.GRIDDISTORTIAN.PROB = 0.5

cfg.PREPROCESSING.RANDOMCROP = CN()
cfg.PREPROCESSING.RANDOMCROP.ENABLE = False
cfg.PREPROCESSING.RANDOMCROP.X_RESIZE = 0.8
cfg.PREPROCESSING.RANDOMCROP.Y_RESIZE = 0.8
cfg.PREPROCESSING.RANDOMCROP.PROB = 0.5


cfg.DROPOUT = CN()
cfg.DROPOUT.ENABLE = False
cfg.DROPOUT.PROB = 0.15


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()

cfg.DATASETS.CYBELE = True
if cfg.DATASETS.CYBELE:
    cfg.DATASETS.BASE_PATH = '../../../lhome/jorghaal/'
else:
    cfg.DATASETS.BASE_PATH = '../../../../work/datasets/medical_project/'

cfg.DATASETS.CAMUS = 'CAMUS'
cfg.DATASETS.TEE = 'TEE/DataTEEGroundTruth'


# -----------------------------------------------------------------------------
# LOSS_function
# -----------------------------------------------------------------------------
cfg.LOSS = CN()

cfg.LOSS.DIFFRENT = False

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
cfg.SOLVER.DIFFRENT = False

# train configs
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.MAX_MINUTES = 120
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
cfg.TEST.EARLY_STOPPING_COUNT = 10
cfg.TEST.EARLY_STOPPING_TOL = 10e-7


cfg.EVAL_EPOCH = 2 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.SAVE_EPOCH = 3*cfg.EVAL_EPOCH
cfg.FIND_LR_ITERATION  = 0 
# cfg.MODEL_SAVE_STEP = 100 # Save checkpoint every save_step
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