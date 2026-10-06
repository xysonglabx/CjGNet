                       


import os
import time
import math
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from config import get_config
from utils import create_logger, seed_set
from utils import NoamLR, build_scheduler, build_optimizer, get_metric_func
from utils import load_checkpoint, save_best_checkpoint, load_best_result
from dataset import build_loader
from loss import bulid_loss
from model import build_model

from evaluate_utils import generate_complete_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced HiGNN with Three-Level Hierarchy")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/example.yaml",
        type=str,
    )

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

                              
    parser.add_argument('--batch-size', type=int, help="batch size for training")
    parser.add_argument(
        '--model-type', '--model_type',
        dest='model_type',
        type=str,
        choices=['NTN', 'GIN', 'GAT', 'GCN', 'ntn', 'gin', 'gat', 'gcn'],
        help='core model type: NTN, GIN, GAT, or GCN',
    )
    parser.add_argument('--lr_scheduler', type=str, help='learning rate scheduler')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    args = parser.parse_args()
    cfg = get_config(args)

    return args, cfg


def compute_regression_metrics(y_true, y_pred):

    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    return metrics


                                     
class CosineWarmupScheduler:


    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6,
                 steps_per_epoch=1, base_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)
        self.total_steps = int(max_epochs * steps_per_epoch)
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self.current_step = 0

                          
        self._set_lr(self.min_lr)

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'base_lr': self.base_lr,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        self.base_lr = state_dict['base_lr']
                    
        lr = self._get_lr()
        self._set_lr(lr)

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        if self.current_step < self.warmup_steps:
                           
            progress = self.current_step / max(self.warmup_steps, 1)
            return self.min_lr + progress * (self.base_lr - self.min_lr)
        else:
                              
            progress = (self.current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self, val_loss=None):

        self.current_step += 1
        lr = self._get_lr()
        self._set_lr(lr)


def build_lr_scheduler(cfg, optimizer, steps_per_epoch):


    scheduler_type = cfg.TRAIN.LR_SCHEDULER.TYPE.lower()

    if scheduler_type == 'cosine_warmup':
        warmup_epochs = getattr(cfg.TRAIN.LR_SCHEDULER, 'WARMUP_EPOCHS', 20)
        min_lr = getattr(cfg.TRAIN.LR_SCHEDULER, 'MIN_LR', 1e-6)
        base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=cfg.TRAIN.MAX_EPOCHS,
            min_lr=min_lr,
            steps_per_epoch=1,              
            base_lr=base_lr
        )
        return scheduler
    else:
                              
        return build_scheduler(cfg, optimizer, steps_per_epoch=steps_per_epoch)


def extract_model_outputs(model_output):


    if isinstance(model_output, dict):
                                                         
        main_output = model_output.get('main')
        mol_vec = model_output.get('mol_vec')

                                                                                
        fra_vec = model_output.get('fra_vec')
        if fra_vec is None:
            fra_vec = model_output.get('pharm_vec')                            

        atom_vec = model_output.get('atom_vec')                       
        descriptors = model_output.get('descriptors')
        return main_output, mol_vec, fra_vec, atom_vec, descriptors
    elif isinstance(model_output, tuple) and len(model_output) == 3:
                                   
        return model_output[0], model_output[1], model_output[2], None, None
    else:
                                     
        return model_output, None, None, None, None


def train_one_epoch(cfg, model, criterion, trainloader, optimizer, lr_scheduler, device, logger):
    model.train()

    losses = []
    loss_breakdown = {
        'main': [],
        'contrastive': [],
        'descriptor': [],
        'hierarchical_cl': [],
        'cross_scale': [],
        'cl_atom_pharm': [],
        'cl_pharm_mol': [],
        'cl_cross_scale': []
    }
    y_pred_list = {}
    y_label_list = {}

                                     
    use_three_level = hasattr(cfg.MODEL, 'THREE_LEVEL') and cfg.MODEL.THREE_LEVEL.ENABLE

                      
    gradient_clip = getattr(cfg.TRAIN, 'GRADIENT_CLIP', 0.0)

    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        model_output = model(data)

                                           
        main_output, mol_vec, fra_vec, atom_vec, pred_descriptors = extract_model_outputs(model_output)

                                      
        true_descriptors = None
        if hasattr(data, 'descriptors') and data.descriptors is not None:
            batch_size = data.batch.max().item() + 1
            num_descriptors = len(cfg.DESCRIPTOR.TYPES) if hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE else 5
            true_descriptors = data.descriptors.view(batch_size, num_descriptors)

                                          
        if batch_idx == 0 and use_three_level:
            logger.info(f"Three-level mode active")
            if atom_vec is not None:
                logger.info(f"Atom features shape: {atom_vec.shape}")
            if fra_vec is not None:
                logger.info(f"Pharm/Fra features shape: {fra_vec.shape}")
            if mol_vec is not None:
                logger.info(f"Mol features shape: {mol_vec.shape}")

        total_loss = 0

        for i in range(len(cfg.DATA.TASK_NAME)):
            if cfg.DATA.TASK_TYPE == 'classification':
                y_pred = main_output[:, i * 2:(i + 1) * 2]
                y_label = data.y[:, i].squeeze()
                validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]

                if len(validId) == 0:
                    continue
                if y_label.dim() == 0:
                    y_label = y_label.unsqueeze(0)

                y_pred = y_pred[torch.tensor(validId).to(device)]
                y_label = y_label[torch.tensor(validId).to(device)]

                                                     
                if isinstance(model_output, dict):
                    task_output = {
                        'main': y_pred,
                        'mol_vec': mol_vec,
                        'fra_vec': fra_vec,
                        'pharm_vec': fra_vec,                         
                        'atom_vec': atom_vec,
                        'descriptors': pred_descriptors
                    }
                    loss_result = criterion(task_output, y_label, descriptors=true_descriptors)

                                                                          
                    if isinstance(loss_result, dict):
                        task_loss = loss_result['total']
                                                    
                        for key in loss_breakdown.keys():
                            if key in loss_result:
                                loss_breakdown[key].append(float(loss_result[key]))
                    else:
                        task_loss = loss_result
                else:
                                 
                    if isinstance(criterion, list):
                        task_loss = criterion[i](y_pred, y_label, mol_vec, fra_vec)
                    else:
                        task_loss = criterion(y_pred, y_label, mol_vec, fra_vec)

                total_loss += task_loss
                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()

            else:              
                y_pred = main_output[:, i]
                y_label = data.y[:, i]

                if isinstance(model_output, dict):
                    task_output = {
                        'main': y_pred,
                        'mol_vec': mol_vec,
                        'fra_vec': fra_vec,
                        'pharm_vec': fra_vec,         
                        'atom_vec': atom_vec,
                        'descriptors': pred_descriptors
                    }
                    loss_result = criterion(task_output, y_label, descriptors=true_descriptors)

                                                                          
                    if isinstance(loss_result, dict):
                        task_loss = loss_result['total']
                                                    
                        for key in loss_breakdown.keys():
                            if key in loss_result:
                                loss_breakdown[key].append(float(loss_result[key]))
                    else:
                        task_loss = loss_result
                else:
                                 
                    task_loss = criterion(y_pred, y_label, mol_vec, fra_vec)

                total_loss += task_loss
                y_pred = y_pred.detach().cpu().numpy()

            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)

        optimizer.zero_grad()
        total_loss.backward()

                          
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()

        if isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step()

        losses.append(total_loss.item())

                     
    results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if cfg.DATA.TASK_TYPE == 'classification':
            nan = False
            if all(target == 0 for target in y_label_list[i]) or all(target == 1 for target in y_label_list[i]):
                nan = True
                logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while training')

            if nan:
                results.append(float('nan'))
                continue

        if len(y_label_list[i]) == 0:
            continue

        results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_results = np.nanmean(results)
    trn_loss = np.array(losses).mean()

                                 
    if any(loss_breakdown['main']):
        log_msg = "Training loss breakdown - "
        components = []

        if loss_breakdown['main']:
            components.append(f"Main: {np.mean(loss_breakdown['main']):.4f}")

        if use_three_level:
            if loss_breakdown['hierarchical_cl']:
                components.append(f"Hierarchical CL: {np.mean(loss_breakdown['hierarchical_cl']):.4f}")
            if loss_breakdown['cross_scale']:
                components.append(f"Cross-scale: {np.mean(loss_breakdown['cross_scale']):.4f}")

                                   
            if loss_breakdown['cl_atom_pharm']:
                components.append(f"Atom-Pharm CL: {np.mean(loss_breakdown['cl_atom_pharm']):.4f}")
            if loss_breakdown['cl_pharm_mol']:
                components.append(f"Pharm-Mol CL: {np.mean(loss_breakdown['cl_pharm_mol']):.4f}")
        else:
            if loss_breakdown['contrastive']:
                components.append(f"Contrastive: {np.mean(loss_breakdown['contrastive']):.4f}")

        if loss_breakdown['descriptor']:
            components.append(f"Descriptor: {np.mean(loss_breakdown['descriptor']):.4f}")

        logger.info(log_msg + ", ".join(components))

    return trn_loss, avg_results


@torch.no_grad()
def validate(cfg, model, criterion, dataloader, epoch, device, logger, eval_mode=False):
    model.eval()

    losses = []
    loss_breakdown = {
        'main': [],
        'contrastive': [],
        'descriptor': [],
        'hierarchical_cl': [],
        'cross_scale': [],
        'cl_atom_pharm': [],
        'cl_pharm_mol': [],
        'cl_cross_scale': []
    }
    y_pred_list = {}
    y_label_list = {}

    use_three_level = hasattr(cfg.MODEL, 'THREE_LEVEL') and cfg.MODEL.THREE_LEVEL.ENABLE

    for data in dataloader:
        data = data.to(device)
        model_output = model(data)

                         
        main_output, mol_vec, fra_vec, atom_vec, pred_descriptors = extract_model_outputs(model_output)

                                      
        true_descriptors = None
        if hasattr(data, 'descriptors') and data.descriptors is not None:
            batch_size = data.batch.max().item() + 1
            num_descriptors = len(cfg.DESCRIPTOR.TYPES) if hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE else 5
            true_descriptors = data.descriptors.view(batch_size, num_descriptors)

        total_loss = 0

        for i in range(len(cfg.DATA.TASK_NAME)):
            if cfg.DATA.TASK_TYPE == 'classification':
                y_pred = main_output[:, i * 2:(i + 1) * 2]
                y_label = data.y[:, i].squeeze()
                validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
                if len(validId) == 0:
                    continue
                if y_label.dim() == 0:
                    y_label = y_label.unsqueeze(0)

                y_pred = y_pred[torch.tensor(validId).to(device)]
                y_label = y_label[torch.tensor(validId).to(device)]

                if isinstance(model_output, dict):
                    task_output = {
                        'main': y_pred,
                        'mol_vec': mol_vec,
                        'fra_vec': fra_vec,
                        'pharm_vec': fra_vec,
                        'atom_vec': atom_vec,
                        'descriptors': pred_descriptors
                    }
                    loss_result = criterion(task_output, y_label, descriptors=true_descriptors)

                                                                          
                    if isinstance(loss_result, dict):
                        task_loss = loss_result['total']
                        for key in loss_breakdown.keys():
                            if key in loss_result:
                                loss_breakdown[key].append(float(loss_result[key]))
                    else:
                        task_loss = loss_result
                else:
                    if isinstance(criterion, list):
                        task_loss = criterion[i](y_pred, y_label, mol_vec, fra_vec)
                    else:
                        task_loss = criterion(y_pred, y_label, mol_vec, fra_vec)

                total_loss += task_loss
                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()

            else:              
                y_pred = main_output[:, i]
                y_label = data.y[:, i]

                if isinstance(model_output, dict):
                    task_output = {
                        'main': y_pred,
                        'mol_vec': mol_vec,
                        'fra_vec': fra_vec,
                        'pharm_vec': fra_vec,
                        'atom_vec': atom_vec,
                        'descriptors': pred_descriptors
                    }
                    loss_result = criterion(task_output, y_label, descriptors=true_descriptors)

                                                                          
                    if isinstance(loss_result, dict):
                        task_loss = loss_result['total']
                        for key in loss_breakdown.keys():
                            if key in loss_result:
                                loss_breakdown[key].append(float(loss_result[key]))
                    else:
                        task_loss = loss_result
                else:
                    task_loss = criterion(y_pred, y_label, mol_vec, fra_vec)

                total_loss += task_loss
                y_pred = y_pred.detach().cpu().numpy()

            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            losses.append(total_loss.item())

                     
    val_results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if cfg.DATA.TASK_TYPE == 'classification':
            nan = False
            if all(target == 0 for target in y_label_list[i]) or all(target == 1 for target in y_label_list[i]):
                nan = True
                logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while validating')

            if nan:
                val_results.append(float('nan'))
                continue

        if len(y_label_list[i]) == 0:
            continue

        val_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_val_results = np.nanmean(val_results)
    val_loss = np.array(losses).mean()

                                 
    if any(loss_breakdown['main']):
        log_msg = "Validation loss breakdown - "
        components = []

        if loss_breakdown['main']:
            components.append(f"Main: {np.mean(loss_breakdown['main']):.4f}")

        if use_three_level:
            if loss_breakdown['hierarchical_cl']:
                components.append(f"Hierarchical CL: {np.mean(loss_breakdown['hierarchical_cl']):.4f}")
            if loss_breakdown['cross_scale']:
                components.append(f"Cross-scale: {np.mean(loss_breakdown['cross_scale']):.4f}")
        else:
            if loss_breakdown['contrastive']:
                components.append(f"Contrastive: {np.mean(loss_breakdown['contrastive']):.4f}")

        if loss_breakdown['descriptor']:
            components.append(f"Descriptor: {np.mean(loss_breakdown['descriptor']):.4f}")

        logger.info(log_msg + ", ".join(components))

                                                  
    if eval_mode:
        result_dir = os.path.join(cfg.OUTPUT_DIR, "predictions")
        os.makedirs(result_dir, exist_ok=True)

        if cfg.DATA.TASK_TYPE == 'regression':
                                                                            
            all_results = []
            for i, task in enumerate(cfg.DATA.TASK_NAME):
                df = pd.DataFrame({
                    'sample_id': range(len(y_label_list[i])),
                    f'{task}_true': y_label_list[i],
                    f'{task}_pred': y_pred_list[i]
                })
                all_results.append(df)
                df.to_csv(os.path.join(result_dir, f"{task}_predictions.csv"), index=False)

            combined_df = pd.concat(all_results, axis=1)
                                                
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
            combined_df.to_csv(os.path.join(result_dir, "all_predictions.csv"), index=False)

                                                   
            metrics_list = []
            for i, task in enumerate(cfg.DATA.TASK_NAME):
                metrics = compute_regression_metrics(y_label_list[i], y_pred_list[i])
                metrics['task'] = task
                metrics_list.append(metrics)

            avg_metrics = {
                'task': 'average',
                'r2': np.mean([m['r2'] for m in metrics_list]),
                'mse': np.mean([m['mse'] for m in metrics_list]),
                'rmse': np.mean([m['rmse'] for m in metrics_list]),
                'mae': np.mean([m['mae'] for m in metrics_list])
            }
            metrics_list.append(avg_metrics)

            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(os.path.join(result_dir, "regression_metrics.csv"), index=False)

            logger.info(f"Saved regression predictions to {result_dir}")
            logger.info("Regression Metrics:")
            logger.info(metrics_df.to_string(index=False))

        else:
                                                                                
            all_results = []
            for i, task in enumerate(cfg.DATA.TASK_NAME):
                df = pd.DataFrame({
                    'sample_id': range(len(y_label_list[i])),
                    f'{task}_true': [int(v) for v in y_label_list[i]],
                    f'{task}_pred_prob': y_pred_list[i],
                    f'{task}_pred_label': [int(v >= 0.5) for v in y_pred_list[i]]
                })
                all_results.append(df)
                df.to_csv(os.path.join(result_dir, f"{task}_predictions.csv"), index=False)

            combined_df = pd.concat(all_results, axis=1)
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
            combined_df.to_csv(os.path.join(result_dir, "all_predictions.csv"), index=False)

            logger.info(f"Saved classification predictions to {result_dir}")

    if eval_mode:
        logger.info(f'Seed {cfg.SEED} Dataset {cfg.DATA.DATASET} ==> '
                    f'The best epoch:{epoch} test_loss:{val_loss:.3f} test_scores:{avg_val_results:.3f}')
        return val_results

    return val_loss, avg_val_results


def train(cfg, logger):
    seed_set(cfg.SEED)

                                
    train_loader, val_loader, test_loader, weights = build_loader(cfg, logger)

                           
    model = build_model(cfg)
    logger.info(f"Selected model type: {cfg.MODEL.TYPE}")
    logger.info(model)

                 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

                               
    optimizer = build_optimizer(cfg, model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")

                                           
    lr_scheduler = build_lr_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

                                   
    if weights is not None:
        criterion = [bulid_loss(cfg, torch.Tensor(w).to(device)) for w in weights]
    else:
        criterion = bulid_loss(cfg)

                                 
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        tensorboard_dir = os.path.join(cfg.OUTPUT_DIR, "tensorboard")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        writer = None

                            
    best_epoch, best_score = 0, 0 if cfg.DATA.TASK_TYPE == 'classification' else float('inf')
    if cfg.TRAIN.RESUME:
        best_epoch, best_score = load_checkpoint(cfg, model, optimizer, lr_scheduler, logger)
        validate(cfg, model, criterion, val_loader, best_epoch, device, logger)

        if cfg.EVAL_MODE:
            return

                           
    logger.info("Start training")

                               
    use_three_level = hasattr(cfg.MODEL, 'THREE_LEVEL') and cfg.MODEL.THREE_LEVEL.ENABLE
    multitask_enabled = use_three_level or (hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE) or cfg.LOSS.CL_LOSS

    if use_three_level:
        logger.info("Three-level hierarchy ENABLED (atom-pharmacophore-molecule)")
        logger.info(f"  Pharmacophore layers: {cfg.MODEL.THREE_LEVEL.PHARM_LAYERS}")
        logger.info(f"  Number of attention heads: {cfg.MODEL.THREE_LEVEL.NUM_HEADS}")
        logger.info(f"  Fusion strategy: {cfg.MODEL.THREE_LEVEL.FUSION_STRATEGY}")
        logger.info(f"  Junction view: {cfg.MODEL.THREE_LEVEL.USE_JUNCTION_VIEW}")

        if hasattr(cfg.LOSS, 'THREE_LEVEL'):
            logger.info(f"  Hierarchical CL: {cfg.LOSS.THREE_LEVEL.HIERARCHICAL_CL}")
            logger.info(f"  Lambda pharm contrastive: {cfg.LOSS.THREE_LEVEL.LAMBDA_PHARM_CONTRASTIVE}")
            logger.info(f"  Lambda cross-scale: {cfg.LOSS.THREE_LEVEL.LAMBDA_CROSS_SCALE}")

    if multitask_enabled:
        logger.info(f"Multi-task learning enabled: CL={cfg.LOSS.CL_LOSS}, "
                    f"DESC={hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE}, "
                    f"THREE_LEVEL={use_three_level}")

        if hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE:
            logger.info(f"Descriptor types: {cfg.DESCRIPTOR.TYPES}")

        if hasattr(cfg, 'LOSS') and hasattr(cfg.LOSS, 'MULTITASK'):
            logger.info(f"Task weights - Main: {cfg.LOSS.MULTITASK.LAMBDA_MAIN}, "
                        f"Contrastive: {cfg.LOSS.MULTITASK.LAMBDA_CONTRASTIVE}, "
                        f"Descriptor: {cfg.LOSS.MULTITASK.LAMBDA_DESCRIPTOR}")

                        
    gradient_clip = getattr(cfg.TRAIN, 'GRADIENT_CLIP', 0.0)
    if gradient_clip > 0:
        logger.info(f"Gradient clipping enabled: max_norm={gradient_clip}")

    use_huber = getattr(cfg.LOSS, 'USE_HUBER', False)
    if use_huber:
        logger.info(f"Using Huber loss (delta={getattr(cfg.LOSS, 'HUBER_DELTA', 1.0)}) instead of MSE")

    logger.info(f"LR scheduler: {cfg.TRAIN.LR_SCHEDULER.TYPE}")
    if cfg.TRAIN.LR_SCHEDULER.TYPE == 'cosine_warmup':
        logger.info(f"  Warmup epochs: {getattr(cfg.TRAIN.LR_SCHEDULER, 'WARMUP_EPOCHS', 20)}")
        logger.info(f"  Min LR: {getattr(cfg.TRAIN.LR_SCHEDULER, 'MIN_LR', 1e-6)}")

    early_stop_cnt = 0
    start_time = time.time()

                                                            
    epoch_records = []

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCHS):
                                             
        trn_loss, trn_score = train_one_epoch(cfg, model, criterion, train_loader, optimizer,
                                              lr_scheduler, device, logger)
        val_loss, val_score = validate(cfg, model, criterion, val_loader, epoch, device, logger)
                                                                
        test_loss, test_score = validate(cfg, model, criterion, test_loader, epoch, device, logger)

                                 
                                        
        if isinstance(lr_scheduler, NoamLR):
            pass                                
        elif isinstance(lr_scheduler, CosineWarmupScheduler):
            lr_scheduler.step()                              
        else:
            lr_scheduler.step(val_loss)                               

                          
        lr_cur = optimizer.param_groups[0]['lr']
        if epoch % cfg.SHOW_FREQ == 0 or epoch == cfg.TRAIN.MAX_EPOCHS - 1:
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} trn_loss:{trn_loss:.3f} '
                        f'trn_{cfg.DATA.METRIC}:{trn_score:.3f} lr:{lr_cur:.6f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} val_loss:{val_loss:.3f} '
                        f'val_{cfg.DATA.METRIC}:{val_score:.3f} lr:{lr_cur:.6f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} test_loss:{test_loss:.3f} '
                        f'test_{cfg.DATA.METRIC}:{test_score:.3f} lr:{lr_cur:.6f}')

                                             
        epoch_records.append({
            'epoch': epoch,
            'lr': lr_cur,
            'train_loss': trn_loss,
            f'train_{cfg.DATA.METRIC}': trn_score,
            'val_loss': val_loss,
            f'val_{cfg.DATA.METRIC}': val_score,
            'test_loss': test_loss,
            f'test_{cfg.DATA.METRIC}': test_score,
        })

                                                   
        loss_dict = {"train_loss": trn_loss}
        acc_dict = {f"train_{cfg.DATA.METRIC}": trn_score}
        loss_dict["valid_loss"] = val_loss
        acc_dict[f"valid_{cfg.DATA.METRIC}"] = val_score

        if cfg.TRAIN.TENSORBOARD.ENABLE and writer is not None:
            writer.add_scalars(f"scalar/{cfg.DATA.METRIC}", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
                                 
            writer.add_scalar("scalar/lr", optimizer.param_groups[0]['lr'], epoch)

                              
        if cfg.DATA.TASK_TYPE == 'classification' and val_score > best_score or \
                cfg.DATA.TASK_TYPE == 'regression' and val_score < best_score:
            best_score, best_epoch = val_score, epoch
            save_best_checkpoint(cfg, epoch, model, best_score, best_epoch, optimizer, lr_scheduler, logger)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

                           
        if early_stop_cnt > cfg.TRAIN.EARLY_STOP > 0:
            logger.info('Early stop hitted!')
            break

                                                   
    if epoch_records:
        epoch_csv_path = os.path.join(cfg.OUTPUT_DIR, 'epoch_losses.csv')
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        pd.DataFrame(epoch_records).to_csv(epoch_csv_path, index=False)
        logger.info(f"Epoch loss history saved to {epoch_csv_path}")

    if cfg.TRAIN.TENSORBOARD.ENABLE and writer is not None:
        writer.close()

                             
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')

                   
    model, best_epoch = load_best_result(cfg, model, logger)
    score = validate(cfg, model, criterion, test_loader, best_epoch, device, logger=logger, eval_mode=True)

              
    eval_dir = generate_complete_evaluation_report(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=cfg.OUTPUT_DIR,
        logger=logger)

    logger.info(f"Training completed. Full evaluation report saved to {eval_dir}")

    return score


if __name__ == "__main__":
    _, cfg = parse_args()

    logger = create_logger(cfg)

                  
    logger.info(cfg.dump())

                       
    if torch.cuda.is_available():
        logger.info('GPU mode...')
    else:
        logger.info('CPU mode...')

              
    train(cfg, logger)