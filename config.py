                       


import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

                                                                               
                     
                                                                               
                       
_C.OUTPUT_DIR = ""
                                                         
_C.TAG = 'default'
                   
_C.SEED = 1
                        
_C.NUM_FOLDS = 10
                                                 
_C.SHOW_EACH_SCORES = False
                                                               
_C.EVAL_MODE = False
                                  
_C.SHOW_FREQ = 5

                  
_C.HYPER = False
_C.HYPER_COUNT = 1
_C.HYPER_REMOVE = None
                                         
_C.NUM_ITERS = 20

                                                                               
               
                                                                               
_C.DATA = CN()
                                                  
_C.DATA.BATCH_SIZE = 64
                                                       
_C.DATA.DATA_PATH = '../data/'
_C.DATA.TEST_PATH = ""
_C.DATA.TRAIN_VAL_RATIO = 0.9
              
_C.DATA.DATASET = 'ht'
                                                                 
_C.DATA.TASK_NAME = None
                                                
_C.DATA.TASK_TYPE = 'regression'
                                                   
_C.DATA.METRIC = 'rmse'
                                                                                                         
_C.DATA.SPLIT_TYPE = 'random'
                                                                                           
_C.DATA.MW_BINS = [200, 300, 400, 500]
                                                                             
_C.DATA.RATE = None

                                                                               
                
                                                                               
_C.MODEL = CN()
                                                       
_C.MODEL.TYPE = 'NTN'
                            
_C.MODEL.HID = 128
                                                        
_C.MODEL.OUT_DIM = None
                  
_C.MODEL.DEPTH = 5
                 
_C.MODEL.SLICES = 2
         
_C.MODEL.DROPOUT = 0.2
                   
_C.MODEL.F_ATT = True
                 
_C.MODEL.R = 4
                                                                                            
_C.MODEL.BRICS = True

                                                                               
                                      
                                                                               
_C.MODEL.THREE_LEVEL = CN()
                                                                       
_C.MODEL.THREE_LEVEL.ENABLE = False
                                                    
_C.MODEL.THREE_LEVEL.NUM_HEADS = 8
                                                      
_C.MODEL.THREE_LEVEL.NUM_NODE_TYPES = 3
                                                                       
_C.MODEL.THREE_LEVEL.NUM_EDGE_TYPES = 5
                                                   
_C.MODEL.THREE_LEVEL.PHARM_DISTANCE_THRESHOLD = 5.0
                                            
_C.MODEL.THREE_LEVEL.PHARM_LAYERS = 2
                        
_C.MODEL.THREE_LEVEL.USE_JUNCTION_VIEW = True
                                                                       
_C.MODEL.THREE_LEVEL.FUSION_STRATEGY = 'hierarchical'

                                                                               
                                                       
                                                                               
_C.DESCRIPTOR = CN()
                                                  
_C.DESCRIPTOR.ENABLE = False
                                             
_C.DESCRIPTOR.TYPES = [
    'mw',                             
    'logp',                                
    'tpsa',                                         
    'hbd',                                
    'hba',                                     
    'nrb',                                     
    'aromatic_rings',                           
    'heteroatoms',                          
    'flexibility',                                
    'complexity'                           
]
                                               
_C.DESCRIPTOR.HIDDEN_DIM = 64
                                      
_C.DESCRIPTOR.DROPOUT = 0.1

                                                                               
                                       
                                                                               
_C.LOSS = CN()
                             
_C.LOSS.FL_LOSS = False
                                                         
_C.LOSS.CL_LOSS = False
                            
_C.LOSS.ALPHA = 0.1
                                                
_C.LOSS.TEMPERATURE = 0.1
                                                                                   
_C.LOSS.USE_HUBER = False
                                
_C.LOSS.HUBER_DELTA = 1.0

                             
_C.LOSS.MULTITASK = CN()
                                                
_C.LOSS.MULTITASK.LAMBDA_MAIN = 1.0
                                      
_C.LOSS.MULTITASK.LAMBDA_CONTRASTIVE = 0.1  
                                           
_C.LOSS.MULTITASK.LAMBDA_DESCRIPTOR = 0.5
                                   
_C.LOSS.MULTITASK.ADAPTIVE_WEIGHTS = False
                                    
_C.LOSS.MULTITASK.ADAPTIVE_TEMP = 2.0

                                          
_C.LOSS.THREE_LEVEL = CN()
                                                 
_C.LOSS.THREE_LEVEL.LAMBDA_PHARM_CONTRASTIVE = 0.1
                                       
_C.LOSS.THREE_LEVEL.LAMBDA_CROSS_SCALE = 0.05
                                                  
_C.LOSS.THREE_LEVEL.HIERARCHICAL_CL = True

                                                                               
                   
                                                                               
_C.TRAIN = CN()
                                                            
_C.TRAIN.RESUME = None
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCHS = 100
                
_C.TRAIN.EARLY_STOP = -1
                                           
_C.TRAIN.GRADIENT_CLIP = 0.0

             
_C.TRAIN.TENSORBOARD = CN()
_C.TRAIN.TENSORBOARD.ENABLE = True

           
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'adam'
               
_C.TRAIN.OPTIMIZER.BASE_LR = 1e-3
                   
_C.TRAIN.OPTIMIZER.FP_LR = 4e-5
              
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
              
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

              
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = 'reduce'
                   
_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS = 2
_C.TRAIN.LR_SCHEDULER.INIT_LR = 1e-4
_C.TRAIN.LR_SCHEDULER.MAX_LR = 1e-2
_C.TRAIN.LR_SCHEDULER.FINAL_LR = 1e-4
                   
_C.TRAIN.LR_SCHEDULER.FACTOR = 0.7
_C.TRAIN.LR_SCHEDULER.PATIENCE = 10
_C.TRAIN.LR_SCHEDULER.MIN_LR = 1e-5


                                                                               
                  
                                                                               
_C.PREDICT = CN()
                                                                                     
_C.PREDICT.CKPT_PATH = ''
                                                              
                                                                             
_C.PREDICT.TEST_CSV = ''
                                               
                                                          
_C.PREDICT.OUTPUT_DIR = ''
                                                             
_C.PREDICT.BATCH_SIZE = 0
                                  
_C.PREDICT.DEVICE = 'auto'
                                                                                  
_C.PREDICT.SAVE_PREDICTIONS = True
                                                                                         
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

                   
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.TAG)

    cfg.freeze()


def get_config(args):

                                                             
                                                  
    cfg = _C.clone()
    update_config(cfg, args)

    return cfg