from yacs.config import CfgNode as CN


_C = CN()

_C.EXP = "newexp" # Experiment name
_C.DEBUG = False

_C.INFER = CN()
_C.INFER.TTA = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 42
_C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O2"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 8

_C.DIRS = CN()
_C.DIRS.DATA = "/home/tungthanhlee/CGIAR/data"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"
_C.DIRS.SUB = "/home/tungthanhlee/CGIAR/src/submission/"

_C.DATA = CN()
_C.DATA.MIXUP = False
_C.DATA.CUTMIX = False
_C.DATA.CM_ALPHA = 1.0
_C.DATA.INP_CHANNEL = 3
_C.DATA.IMG_SIZE = 512
_C.DATA.CSV = "/home/tungthanhlee/CGIAR/data/Folds"
_C.DATA.FOLD = 0




_C.OPT = CN()
_C.OPT.GD_STEPS = 1 
_C.OPT.WARMUP_EPOCHS = 2
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.CLR = False 

_C.TRAIN = CN()
_C.TRAIN.MODEL = "resnet101" # Model name
_C.TRAIN.EPOCHS = 20
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.NUM_CLASSES = 3
_C.TRAIN.DROPOUT = 0.0
_C.TRAIN.POOL_TYPE = 'avg'
_C.TRAIN.PRETRAINED = True
_C.TRAIN.AUG = True
_C.TRAIN.ACTIVATION = 'relu'
_C.TRAIN.RANDAUG_N = 0
_C.TRAIN.RANDAUG_M = 0

_C.MODEL = CN()
_C.MODEL.DROP_CONNECT = 0.0
_C.MODEL.WEIGHT = "" 
_C.MODEL.SWA = False
_C.MODEL.SWA_CIRCLES = 1
_C.MODEL.SWA_EVAL_FREQ = 5





_C.VAL = CN()
_C.VAL.BATCH_SIZE = 32



_C.CONST = CN()
_C.CONST.LABELS = ['leaf_rust','stem_rust','healthy_wheat'] 

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`