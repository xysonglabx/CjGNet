                       


import os
import yaml
import numpy as np
from copy import deepcopy
from hyperopt import fmin, hp, tpe

import torch

from model import build_model
from train import train, parse_args
from utils import create_logger, get_task_names


                                         
                         
                                         
def cross_validate(cfg, logger):


                                   
    init_seed = cfg.SEED
    out_dir = cfg.OUTPUT_DIR
    task_names = get_task_names(os.path.join(cfg.DATA.DATA_PATH, 'raw/{}.csv'.format(cfg.DATA.DATASET)))

    logger.info(f"Model type: {cfg.MODEL.TYPE}")

                                               
    use_three_level = hasattr(cfg.MODEL, 'THREE_LEVEL') and cfg.MODEL.THREE_LEVEL.ENABLE
    
    if use_three_level:
        logger.info("=" * 60)
        logger.info("THREE-LEVEL HIERARCHY CROSS-VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Architecture: Atom-Pharmacophore-Molecule")
        logger.info(f"Fusion Strategy: {cfg.MODEL.THREE_LEVEL.FUSION_STRATEGY}")
        logger.info(f"Number of Heads: {cfg.MODEL.THREE_LEVEL.NUM_HEADS}")
        logger.info(f"Pharmacophore Layers: {cfg.MODEL.THREE_LEVEL.PHARM_LAYERS}")
        logger.info("=" * 60)
    else:
        logger.info("Standard two-level cross-validation")

                                                          
    all_scores = []
    fold_details = []
    
    for fold_num in range(cfg.NUM_FOLDS):
        cfg.defrost()
        cfg.SEED = init_seed + fold_num
        cfg.OUTPUT_DIR = os.path.join(out_dir, f'fold_{fold_num}')
        cfg.freeze()
        
        logger.info(f'Fold {fold_num} (Seed: {cfg.SEED})')
        
                                    
        model_scores = train(cfg, logger)
        all_scores.append(model_scores)
        
                                          
        fold_info = {
            'fold': fold_num,
            'seed': cfg.SEED,
            'scores': model_scores,
            'mean_score': np.nanmean(model_scores)
        }
        fold_details.append(fold_info)
        
    all_scores = np.array(all_scores)

                    
    cfg.defrost()
    cfg.OUTPUT_DIR = out_dir
    cfg.freeze()
    
    logger.info("=" * 60)
    logger.info(f'{cfg.NUM_FOLDS}-FOLD CROSS VALIDATION RESULTS')
    logger.info("=" * 60)

                                 
    for fold_num, scores in enumerate(all_scores):
        fold_mean = np.nanmean(scores)
        logger.info(f'Fold {fold_num} (Seed {init_seed + fold_num}) ==> '
                   f'test {cfg.DATA.METRIC} = {fold_mean:.3f}')
        
        if cfg.SHOW_EACH_SCORES:
            for task_name, score in zip(task_names, scores):
                logger.info(f'  Task "{task_name}": {cfg.DATA.METRIC} = {score:.3f}')

                                        
    avg_scores = np.nanmean(all_scores, axis=1)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    
    logger.info("=" * 60)
    logger.info(f'Overall test {cfg.DATA.METRIC} = {mean_score:.3f} ± {std_score:.3f}')
    logger.info("=" * 60)

    if cfg.SHOW_EACH_SCORES:
        for task_num, task_name in enumerate(task_names):
            task_mean = np.nanmean(all_scores[:, task_num])
            task_std = np.nanstd(all_scores[:, task_num])
            logger.info(f'Task "{task_name}": {cfg.DATA.METRIC} = {task_mean:.3f} ± {task_std:.3f}')

                                   
    results_path = os.path.join(out_dir, 'cv_results.yaml')
    cv_results = {
        'configuration': {
            'dataset': cfg.DATA.DATASET,
            'model_type': cfg.MODEL.TYPE,
            'task_type': cfg.DATA.TASK_TYPE,
            'metric': cfg.DATA.METRIC,
            'num_folds': cfg.NUM_FOLDS,
            'three_level_enabled': use_three_level
        },
        'overall': {
            'mean': float(mean_score),
            'std': float(std_score)
        },
        'fold_details': fold_details
    }
    
    if use_three_level:
        cv_results['three_level_config'] = {
            'fusion_strategy': cfg.MODEL.THREE_LEVEL.FUSION_STRATEGY,
            'num_heads': cfg.MODEL.THREE_LEVEL.NUM_HEADS,
            'pharm_layers': cfg.MODEL.THREE_LEVEL.PHARM_LAYERS,
            'use_junction_view': cfg.MODEL.THREE_LEVEL.USE_JUNCTION_VIEW
        }
    
    with open(results_path, 'w') as f:
        yaml.dump(cv_results, f, indent=4, sort_keys=False)
    logger.info(f"Cross-validation results saved to {results_path}")

    return mean_score, std_score


                                         
                              
                                         

                            
THREE_LEVEL_SPACE = {
                                             
    'MODEL.HID': hp.choice('dim', [128, 256, 384, 512]),
    'MODEL.SLICES': hp.choice('slices', [2, 4, 8]),
    'MODEL.DROPOUT': hp.quniform('dropout', low=0.05, high=0.4, q=0.05),
    'MODEL.DEPTH': hp.choice('depth', [3, 4, 5, 6]),
    
                                          
    'MODEL.THREE_LEVEL.NUM_HEADS': hp.choice('num_heads', [4, 8, 12, 16]),
    'MODEL.THREE_LEVEL.PHARM_LAYERS': hp.choice('pharm_layers', [1, 2, 3, 4]),
    'MODEL.THREE_LEVEL.PHARM_DISTANCE_THRESHOLD': hp.quniform('pharm_dist', low=3.0, high=8.0, q=0.5),
    
                         
    'LOSS.THREE_LEVEL.LAMBDA_PHARM_CONTRASTIVE': hp.quniform('lambda_pharm', low=0.05, high=0.3, q=0.05),
    'LOSS.THREE_LEVEL.LAMBDA_CROSS_SCALE': hp.quniform('lambda_cross', low=0.0, high=0.15, q=0.05),
    
                         
    'LOSS.MULTITASK.LAMBDA_DESCRIPTOR': hp.quniform('lambda_desc', low=0.1, high=0.5, q=0.05),
    
                                        
    'TRAIN.OPTIMIZER.BASE_LR': hp.loguniform('lr', np.log(1e-5), np.log(5e-3)),
    'TRAIN.OPTIMIZER.WEIGHT_DECAY': hp.choice('l2', [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]),
    
                             
    'LOSS.HUBER_DELTA': hp.quniform('huber_delta', low=0.5, high=2.0, q=0.25),
}

                          
STANDARD_SPACE = {
    'MODEL.HID': hp.choice('dim', [64, 128, 256, 512]),
    'MODEL.SLICES': hp.choice('slices', [1, 2, 4, 8]),
    'MODEL.DROPOUT': hp.quniform('dropout', low=0.0, high=0.5, q=0.05),
    'MODEL.DEPTH': hp.choice('depth', [2, 3, 4, 5, 6]),
    'TRAIN.OPTIMIZER.BASE_LR': hp.loguniform('lr', np.log(1e-5), np.log(5e-3)),
    'TRAIN.OPTIMIZER.WEIGHT_DECAY': hp.choice('l2', [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]),
}

INT_KEYS_THREE_LEVEL = [
    'MODEL.HID', 'MODEL.DEPTH', 'MODEL.SLICES',
    'MODEL.THREE_LEVEL.NUM_HEADS', 'MODEL.THREE_LEVEL.PHARM_LAYERS'
]

INT_KEYS_STANDARD = ['MODEL.HID', 'MODEL.DEPTH', 'MODEL.SLICES']


def hyperopt(cfg, logger):


    
                                     
    use_three_level = hasattr(cfg.MODEL, 'THREE_LEVEL') and cfg.MODEL.THREE_LEVEL.ENABLE
    
    if use_three_level:
        logger.info("=" * 60)
        logger.info("THREE-LEVEL HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 60)
        SPACE = THREE_LEVEL_SPACE.copy()
        INT_KEYS = INT_KEYS_THREE_LEVEL.copy()
    else:
        logger.info("Standard hyperparameter optimization")
        SPACE = STANDARD_SPACE.copy()
        INT_KEYS = INT_KEYS_STANDARD.copy()
    
                                                    
    if cfg.MODEL.F_ATT:
        SPACE['MODEL.R'] = hp.choice('R', [1, 2, 4, 8])
        INT_KEYS.append('MODEL.R')
    
    if cfg.MODEL.BRICS and cfg.LOSS.CL_LOSS:
        SPACE['LOSS.ALPHA'] = hp.quniform('alpha', low=0.05, high=0.3, q=0.05)
        SPACE['LOSS.TEMPERATURE'] = hp.choice('temperature', [0.05, 0.07, 0.1, 0.15, 0.2])
    
                                      
    if hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE and not use_three_level:
        SPACE['LOSS.MULTITASK.LAMBDA_DESCRIPTOR'] = hp.quniform('lambda_desc', low=0.1, high=0.5, q=0.05)
    
                                                 
    if cfg.HYPER_REMOVE is not None:
        for param in cfg.HYPER_REMOVE:
            if param in SPACE:
                del SPACE[param]
        INT_KEYS = [k for k in INT_KEYS if k not in cfg.HYPER_REMOVE]
    
                                        
    suffix = "_three_level" if use_three_level else ""
    yaml_name = f"best_{cfg.DATA.DATASET}_{cfg.TAG}{suffix}.yaml"
    cfg_save_path = os.path.join(cfg.OUTPUT_DIR, yaml_name)
    
                                     
    results = []
    best_score = float('inf') if cfg.DATA.TASK_TYPE == 'regression' else 0.0

                                                  
    def objective(hyperparams):
        nonlocal best_score
        
                                                              
        for key in INT_KEYS:
            if key in hyperparams:
                hyperparams[key] = int(hyperparams[key])

                                                   
        hyper_cfg = deepcopy(cfg)
        if hyper_cfg.OUTPUT_DIR is not None:
            folder_name = f'round_{hyper_cfg.HYPER_COUNT}'
            hyper_cfg.defrost()
            hyper_cfg.OUTPUT_DIR = os.path.join(hyper_cfg.OUTPUT_DIR, folder_name)
            hyper_cfg.freeze()
        
        hyper_cfg.defrost()
        opts = []
        for key, value in hyperparams.items():
            opts.append(key)
            opts.append(value)
        hyper_cfg.merge_from_list(opts)
        hyper_cfg.freeze()

                                
        cfg.defrost()
        cfg.HYPER_COUNT += 1
        cfg.freeze()
        
        logger.info(f'Round {hyper_cfg.HYPER_COUNT - 1}')
        logger.info('Hyperparameters:')
        for key, value in hyperparams.items():
            logger.info(f'  {key}: {value}')

                        
        mean_score, std_score = cross_validate(hyper_cfg, logger)

                                    
        temp_model = build_model(hyper_cfg)
        num_params = sum(param.numel() for param in temp_model.parameters() if param.requires_grad)
        
        logger.info(f'Model parameters: {num_params:,}')
        logger.info(f'Score: {mean_score:.3f} ± {std_score:.3f} {hyper_cfg.DATA.METRIC}')

                        
        result_info = {
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams.copy(),
            'num_params': num_params,
            'round': hyper_cfg.HYPER_COUNT - 1
        }
        results.append(result_info)
        
                           
        if cfg.DATA.TASK_TYPE == 'regression':
            if mean_score < best_score:
                best_score = mean_score
                logger.info(f'New best score: {best_score:.3f}')
        else:
            if mean_score > best_score:
                best_score = mean_score
                logger.info(f'New best score: {best_score:.3f}')

                       
        if np.isnan(mean_score):
            if hyper_cfg.DATA.TASK_TYPE == 'classification':
                mean_score = 0
            else:
                logger.warning('NaN score encountered, using large penalty')
                mean_score = 1e6

                                                                                       
        return mean_score if cfg.DATA.TASK_TYPE == 'regression' else -mean_score

                                     
    logger.info(f"Starting hyperparameter search with {cfg.NUM_ITERS} iterations...")
    logger.info(f"Search space contains {len(SPACE)} parameters:")
    for key in sorted(SPACE.keys()):
        logger.info(f"  {key}")
    
    best_hyperparams = fmin(
        objective, 
        SPACE, 
        algo=tpe.suggest, 
        max_evals=cfg.NUM_ITERS, 
        verbose=False
    )

                                    
    results = [r for r in results if not np.isnan(r['mean_score'])]
    
    if cfg.DATA.TASK_TYPE == 'regression':
        best_result = min(results, key=lambda r: r['mean_score'])
    else:
        best_result = max(results, key=lambda r: r['mean_score'])
    
    logger.info("=" * 60)
    logger.info('BEST HYPERPARAMETERS')
    logger.info("=" * 60)
    
    for key, value in best_result['hyperparams'].items():
        logger.info(f'{key}: {value}')
    
    logger.info(f'Model parameters: {best_result["num_params"]:,}')
    logger.info(f'Score: {best_result["mean_score"]:.3f} ± {best_result["std_score"]:.3f} {cfg.DATA.METRIC}')
    logger.info("=" * 60)

                                       
    save_config = {
        'dataset': cfg.DATA.DATASET,
        'model_type': cfg.MODEL.TYPE,
        'task_type': cfg.DATA.TASK_TYPE,
        'metric': cfg.DATA.METRIC,
        'three_level_enabled': use_three_level,
        'best_score': float(best_result['mean_score']),
        'std_score': float(best_result['std_score']),
        'num_params': int(best_result['num_params']),
        'hyperparameters': best_result['hyperparams']
    }
    
    with open(cfg_save_path, 'w') as f:
        yaml.dump(save_config, f, indent=4, sort_keys=True)
    
    logger.info(f"Best hyperparameters saved to {cfg_save_path}")
    
                                   
    all_results_path = os.path.join(cfg.OUTPUT_DIR, f'all_results{suffix}.yaml')
    with open(all_results_path, 'w') as f:
        yaml.dump(results, f, indent=4)
    
    logger.info(f"All results saved to {all_results_path}")


if __name__ == '__main__':
    _, cfg = parse_args()

    logger = create_logger(cfg)

                       
    if torch.cuda.is_available():
        logger.info('GPU mode enabled')
    else:
        logger.info('CPU mode (GPU not available)')

                         
    use_three_level = hasattr(cfg.MODEL, 'THREE_LEVEL') and cfg.MODEL.THREE_LEVEL.ENABLE
    
    if use_three_level:
        logger.info("Three-level hierarchy is ENABLED")
        logger.info(f"  Fusion strategy: {cfg.MODEL.THREE_LEVEL.FUSION_STRATEGY}")
        logger.info(f"  Number of heads: {cfg.MODEL.THREE_LEVEL.NUM_HEADS}")
        logger.info(f"  Pharmacophore layers: {cfg.MODEL.THREE_LEVEL.PHARM_LAYERS}")
    else:
        logger.info("Using standard two-level hierarchy")

                                                 
    if cfg.HYPER:
        hyperopt(cfg, logger)
    else:
        logger.info(cfg.dump())
        cross_validate(cfg, logger)