# -*- coding: utf-8 -*-
"""
Enhanced training script with Bayesian Optimization support.
This file is designed as a drop-in replacement for train.py.
"""

import os
import time
import math
import json
import copy
import random
import shutil
import datetime
import argparse
import traceback
import numpy as np
import pandas as pd
from math import erf, sqrt, exp, pi
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from optimized_config import get_config
from utils import create_logger, seed_set
from utils import NoamLR, build_scheduler, build_optimizer, get_metric_func
from utils import load_checkpoint, save_best_checkpoint, load_best_result
from dataset import build_loader
from loss import bulid_loss
from model import build_model
from evaluate_utils import generate_complete_evaluation_report



def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced HiGIN with Bayesian Optimization")

    parser.add_argument(
        "--cfg",
        help="configuration file",
        required=False,
        default="configs/example.yaml",
        type=str,
    )

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
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
    return {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

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
        progress = (self.current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self, val_loss=None):
        self.current_step += 1
        self._set_lr(self._get_lr())



def build_lr_scheduler(cfg, optimizer, steps_per_epoch):
    scheduler_type = cfg.TRAIN.LR_SCHEDULER.TYPE.lower()
    if scheduler_type == 'cosine_warmup':
        warmup_epochs = getattr(cfg.TRAIN.LR_SCHEDULER, 'WARMUP_EPOCHS', 20)
        min_lr = getattr(cfg.TRAIN.LR_SCHEDULER, 'MIN_LR', 1e-6)
        base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR
        return CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=cfg.TRAIN.MAX_EPOCHS,
            min_lr=min_lr,
            steps_per_epoch=1,
            base_lr=base_lr
        )
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
    if isinstance(model_output, tuple) and len(model_output) == 3:
        return model_output[0], model_output[1], model_output[2], None, None
    return model_output, None, None, None, None



def train_one_epoch(cfg, model, criterion, trainloader, optimizer, lr_scheduler, device, logger):
    model.train()
    losses = []
    loss_breakdown = {
        'main': [], 'contrastive': [], 'descriptor': [], 'hierarchical_cl': [],
        'cross_scale': [], 'cl_atom_pharm': [], 'cl_pharm_mol': [], 'cl_cross_scale': []
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
            logger.info("Three-level mode active")
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
                    task_loss = criterion[i](y_pred, y_label, mol_vec, fra_vec) if isinstance(criterion, list) else criterion(y_pred, y_label, mol_vec, fra_vec)

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

            if i not in y_label_list:
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
            nan = all(target == 0 for target in y_label_list.get(i, [])) or all(target == 1 for target in y_label_list.get(i, []))
            if nan:
                logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while training')
                results.append(float('nan'))
                continue
        if len(y_label_list.get(i, [])) == 0:
            continue
        results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_results = np.nanmean(results)
    trn_loss = np.array(losses).mean()

    if any(loss_breakdown['main']):
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
        logger.info("Training loss breakdown - " + ", ".join(components))

    return trn_loss, avg_results


@torch.no_grad()
def validate(cfg, model, criterion, dataloader, epoch, device, logger, eval_mode=False):
    model.eval()
    losses = []
    loss_breakdown = {
        'main': [], 'contrastive': [], 'descriptor': [], 'hierarchical_cl': [],
        'cross_scale': [], 'cl_atom_pharm': [], 'cl_pharm_mol': [], 'cl_cross_scale': []
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
                    task_loss = criterion[i](y_pred, y_label, mol_vec, fra_vec) if isinstance(criterion, list) else criterion(y_pred, y_label, mol_vec, fra_vec)
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

            if i not in y_label_list:
                y_label_list[i] = []
                y_pred_list[i] = []
            y_label_list[i].extend(y_label.cpu().numpy())
            y_pred_list[i].extend(y_pred)
            losses.append(total_loss.item())

    val_results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if cfg.DATA.TASK_TYPE == 'classification':
            nan = all(target == 0 for target in y_label_list.get(i, [])) or all(target == 1 for target in y_label_list.get(i, []))
            if nan:
                logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while validating')
                val_results.append(float('nan'))
                continue
        if len(y_label_list.get(i, [])) == 0:
            continue
        val_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_val_results = np.nanmean(val_results)
    val_loss = np.array(losses).mean()

    if any(loss_breakdown['main']):
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
        logger.info("Validation loss breakdown - " + ", ".join(components))

    if eval_mode:
        result_dir = os.path.join(cfg.OUTPUT_DIR, "predictions")
        os.makedirs(result_dir, exist_ok=True)
        if cfg.DATA.TASK_TYPE == 'regression':
            all_results = []
            for i, task in enumerate(cfg.DATA.TASK_NAME):
                df = pd.DataFrame({'sample_id': range(len(y_label_list[i])), f'{task}_true': y_label_list[i], f'{task}_pred': y_pred_list[i]})
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
            pd.DataFrame(metrics_list).to_csv(os.path.join(result_dir, "regression_metrics.csv"), index=False)
            logger.info(f"Saved regression predictions to {result_dir}")
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

        logger.info(f'Seed {cfg.SEED} Dataset {cfg.DATA.DATASET} ==> The best epoch:{epoch} test_loss:{val_loss:.3f} test_scores:{avg_val_results:.3f}')
        return val_results

    return val_loss, avg_val_results


# -----------------------------------------------------------------------------
# Bayesian optimization utilities
# -----------------------------------------------------------------------------

def _normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class BayesianSearchSpace:
    def __init__(self, cfg):
        s = cfg.BO.SEARCH
        self.dim_specs = [
            ('lr', 'float_log', s.LR_MIN, s.LR_MAX),
            ('weight_decay', 'float_log', s.WD_MIN, s.WD_MAX),
            ('dropout', 'float', s.DROPOUT_MIN, s.DROPOUT_MAX),
            ('alpha', 'float', s.ALPHA_MIN, s.ALPHA_MAX),
            ('lambda_descriptor', 'float', s.LAMBDA_DESC_MIN, s.LAMBDA_DESC_MAX),
            ('lambda_pharm', 'float', s.LAMBDA_PHARM_MIN, s.LAMBDA_PHARM_MAX),
            ('lambda_cross', 'float', s.LAMBDA_CROSS_MIN, s.LAMBDA_CROSS_MAX),
            ('grad_clip', 'float', s.GRAD_CLIP_MIN, s.GRAD_CLIP_MAX),
            ('hid', 'choice', list(s.HID_CHOICES)),
            ('depth', 'choice', list(s.DEPTH_CHOICES)),
            ('batch_size', 'choice', list(s.BATCH_CHOICES)),
        ]

    def sample_random(self, rng):
        x = []
        for spec in self.dim_specs:
            kind = spec[1]
            if kind == 'choice':
                choices = spec[2]
                x.append(rng.uniform(0.0, 1.0))
            else:
                x.append(rng.uniform(0.0, 1.0))
        return np.array(x, dtype=np.float64)

    def decode(self, x):
        x = np.asarray(x, dtype=np.float64)
        params = {}
        for i, spec in enumerate(self.dim_specs):
            name, kind = spec[0], spec[1]
            v = float(np.clip(x[i], 0.0, 1.0))
            if kind == 'float':
                lo, hi = spec[2], spec[3]
                params[name] = lo + (hi - lo) * v
            elif kind == 'float_log':
                lo, hi = spec[2], spec[3]
                params[name] = float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * v))
            elif kind == 'choice':
                choices = spec[2]
                idx = min(int(round(v * (len(choices) - 1))), len(choices) - 1)
                params[name] = choices[idx]
            else:
                raise ValueError(f'Unknown search type: {kind}')
        return params



def apply_trial_params(cfg, params, trial_tag=None):
    cfg = cfg.clone()
    cfg.defrost()
    cfg.TRAIN.OPTIMIZER.BASE_LR = float(params['lr'])
    cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY = float(params['weight_decay'])
    cfg.MODEL.DROPOUT = float(params['dropout'])
    cfg.MODEL.HID = int(params['hid'])
    cfg.MODEL.DEPTH = int(params['depth'])
    cfg.DATA.BATCH_SIZE = int(params['batch_size'])
    cfg.TRAIN.GRADIENT_CLIP = float(params['grad_clip'])

    cfg.LOSS.ALPHA = float(params['alpha'])
    if hasattr(cfg.LOSS, 'MULTITASK'):
        cfg.LOSS.MULTITASK.LAMBDA_DESCRIPTOR = float(params['lambda_descriptor'])
    if hasattr(cfg.LOSS, 'THREE_LEVEL'):
        cfg.LOSS.THREE_LEVEL.LAMBDA_PHARM_CONTRASTIVE = float(params['lambda_pharm'])
        cfg.LOSS.THREE_LEVEL.LAMBDA_CROSS_SCALE = float(params['lambda_cross'])

    if trial_tag is not None:
        cfg.TAG = trial_tag
        cfg.OUTPUT_DIR = os.path.join(os.path.dirname(cfg.OUTPUT_DIR), trial_tag)
    cfg.freeze()
    return cfg



def is_higher_better(cfg):
    return cfg.DATA.TASK_TYPE == 'classification'



def score_to_objective(cfg, score):
    return score if is_higher_better(cfg) else -score



def objective_to_score(cfg, objective):
    return objective if is_higher_better(cfg) else -objective



def expected_improvement(X, X_obs, y_obs, gp, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    y_best = np.max(y_obs)
    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * np.vectorize(_normal_cdf)(z) + sigma * np.vectorize(_normal_pdf)(z)
    ei[sigma < 1e-12] = 0.0
    return ei



def suggest_next_point(space, X_obs, y_obs, rng, n_candidates=256):
    if len(X_obs) < 3:
        return space.sample_random(rng)

    X_obs = np.asarray(X_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_obs.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=rng.randint(0, 10**6), n_restarts_optimizer=2)

    try:
        gp.fit(X_obs, y_obs)
    except Exception:
        return space.sample_random(rng)

    X_cand = np.vstack([space.sample_random(rng) for _ in range(n_candidates)])
    ei = expected_improvement(X_cand, X_obs, y_obs, gp)
    best_idx = int(np.argmax(ei))
    return X_cand[best_idx]



def train_single_run(cfg, logger, final_eval=True):
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

    writer = None
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        tensorboard_dir = os.path.join(cfg.OUTPUT_DIR, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)

    best_epoch, best_score = 0, 0 if cfg.DATA.TASK_TYPE == 'classification' else float('inf')
    if cfg.TRAIN.RESUME:
        best_epoch, best_score = load_checkpoint(cfg, model, optimizer, lr_scheduler, logger)
        validate(cfg, model, criterion, val_loader, best_epoch, device, logger)
        if cfg.EVAL_MODE:
            return {
                'best_epoch': best_epoch,
                'best_score': best_score,
                'final_score': best_score,
                'test_score': None,
            }

    logger.info("Start training")
    early_stop_cnt = 0
    start_time = time.time()
    epoch_records = []

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCHS):
        trn_loss, trn_score = train_one_epoch(cfg, model, criterion, train_loader, optimizer, lr_scheduler, device, logger)
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
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} trn_loss:{trn_loss:.3f} trn_{cfg.DATA.METRIC}:{trn_score:.3f} lr:{lr_cur:.6f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} val_loss:{val_loss:.3f} val_{cfg.DATA.METRIC}:{val_score:.3f} lr:{lr_cur:.6f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} test_loss:{test_loss:.3f} test_{cfg.DATA.METRIC}:{test_score:.3f} lr:{lr_cur:.6f}')

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

        if writer is not None:
            writer.add_scalars(f"scalar/{cfg.DATA.METRIC}", {f"train_{cfg.DATA.METRIC}": trn_score, f"valid_{cfg.DATA.METRIC}": val_score}, epoch)
            writer.add_scalars("scalar/loss", {"train_loss": trn_loss, "valid_loss": val_loss}, epoch)
            writer.add_scalar("scalar/lr", optimizer.param_groups[0]['lr'], epoch)

        improved = (cfg.DATA.TASK_TYPE == 'classification' and val_score > best_score) or (cfg.DATA.TASK_TYPE == 'regression' and val_score < best_score)
        if improved:
            best_score, best_epoch = val_score, epoch
            save_best_checkpoint(cfg, epoch, model, best_score, best_epoch, optimizer, lr_scheduler, logger)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt > cfg.TRAIN.EARLY_STOP > 0:
            logger.info('Early stop hitted!')
            break

    if epoch_records:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        pd.DataFrame(epoch_records).to_csv(os.path.join(cfg.OUTPUT_DIR, 'epoch_losses.csv'), index=False)

    if writer is not None:
        writer.close()

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'Training time {total_time_str}')

    if not final_eval:
        return {
            'best_epoch': best_epoch,
            'best_score': best_score,
            'final_score': best_score,
            'test_score': None,
        }

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
        logger=logger
    )
    logger.info(f"Training completed. Full evaluation report saved to {eval_dir}")
    return {
        'best_epoch': best_epoch,
        'best_score': best_score,
        'final_score': score,
        'test_score': score,
        'eval_dir': eval_dir,
    }



def run_bayesian_optimization(cfg, logger):
    logger.info("=" * 80)
    logger.info("STARTING BAYESIAN OPTIMIZATION")
    logger.info("=" * 80)

    bo_root = os.path.join(cfg.OUTPUT_DIR, 'bayes_opt')
    os.makedirs(bo_root, exist_ok=True)

    n_trials = cfg.NUM_ITERS if cfg.BO.USE_LEGACY_NUM_ITERS else cfg.BO.N_TRIALS
    init_points = min(cfg.BO.INIT_POINTS, n_trials)
    rng = random.Random(cfg.BO.RANDOM_STATE)
    space = BayesianSearchSpace(cfg)

    X_obs, y_obs, trial_records = [], [], []
    best_obj = -float('inf')
    best_params = None
    best_trial = None

    for trial_idx in range(n_trials):
        logger.info("-" * 80)
        logger.info(f"BO trial {trial_idx + 1}/{n_trials}")

        if trial_idx < init_points:
            x = space.sample_random(rng)
            propose_mode = 'random_init'
        else:
            x = suggest_next_point(space, X_obs, y_obs, rng, n_candidates=cfg.BO.N_CANDIDATES)
            propose_mode = 'gp_ei'

        params = space.decode(x)
        trial_tag = f"{cfg.TAG}_bo_trial_{trial_idx:03d}"
        trial_cfg = apply_trial_params(cfg, params, trial_tag=trial_tag)
        trial_cfg.defrost()
        trial_cfg.HYPER = False
        trial_cfg.BO.ENABLE = False
        trial_cfg.freeze()

        trial_log_name = f"{cfg.DATA.DATASET}_{trial_tag}"
        trial_logger = create_logger(trial_cfg)
        trial_logger.info(f"Propose mode: {propose_mode}")
        trial_logger.info(json.dumps(params, ensure_ascii=False, indent=2))

        try:
            result = train_single_run(trial_cfg, trial_logger, final_eval=False)
            val_score = float(result['best_score'])
            objective_value = score_to_objective(cfg, val_score)
            status = 'ok'
            error_msg = ''
        except Exception as e:
            val_score = float('inf') if cfg.DATA.TASK_TYPE == 'regression' else 0.0
            objective_value = score_to_objective(cfg, val_score)
            status = 'failed'
            error_msg = ''.join(traceback.format_exception_only(type(e), e)).strip()
            trial_logger.error(traceback.format_exc())

        X_obs.append(x)
        y_obs.append(objective_value)

        record = {
            'trial': trial_idx,
            'status': status,
            'mode': propose_mode,
            'objective': objective_value,
            'val_score': val_score,
            **params,
        }
        if error_msg:
            record['error'] = error_msg
        trial_records.append(record)
        pd.DataFrame(trial_records).to_csv(os.path.join(bo_root, 'bo_trials.csv'), index=False)

        better = objective_value > best_obj
        if better and status == 'ok':
            best_obj = objective_value
            best_params = params
            best_trial = record
            with open(os.path.join(bo_root, 'best_params.json'), 'w', encoding='utf-8') as f:
                json.dump({'best_trial': best_trial, 'best_params': best_params}, f, ensure_ascii=False, indent=2)

        logger.info(f"Trial {trial_idx} finished: val_{cfg.DATA.METRIC}={val_score:.6f}, objective={objective_value:.6f}, status={status}")

    if best_params is None:
        raise RuntimeError('Bayesian optimization failed: no successful trial was completed.')

    logger.info("=" * 80)
    logger.info(f"Best BO validation score: {objective_to_score(cfg, best_obj):.6f}")
    logger.info(json.dumps(best_params, ensure_ascii=False, indent=2))
    logger.info("=" * 80)

    if not cfg.BO.RETRAIN_BEST:
        return {
            'best_params': best_params,
            'best_trial': best_trial,
            'bo_root': bo_root,
        }

    best_tag = f"{cfg.TAG}_bo_best"
    best_cfg = apply_trial_params(cfg, best_params, trial_tag=best_tag)
    best_cfg.defrost()
    best_cfg.HYPER = False
    best_cfg.BO.ENABLE = False
    best_cfg.freeze()

    best_logger = create_logger(best_cfg)
    best_logger.info("Retraining best BO configuration from scratch")
    best_logger.info(json.dumps(best_params, ensure_ascii=False, indent=2))
    final_result = train_single_run(best_cfg, best_logger, final_eval=True)

    with open(os.path.join(bo_root, 'final_retrain_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': best_params,
            'bo_best_val_score': objective_to_score(cfg, best_obj),
            'final_result': {
                'best_epoch': int(final_result['best_epoch']) if final_result.get('best_epoch') is not None else None,
                'best_score': float(final_result['best_score']) if final_result.get('best_score') is not None else None,
            }
        }, f, ensure_ascii=False, indent=2)

    return {
        'best_params': best_params,
        'best_trial': best_trial,
        'bo_root': bo_root,
        'final_result': final_result,
        'best_run_output_dir': best_cfg.OUTPUT_DIR,
    }



def main(cfg):
    logger = create_logger(cfg)
    logger.info(cfg.dump())
    logger.info('GPU mode...' if torch.cuda.is_available() else 'CPU mode...')

    if cfg.HYPER or cfg.BO.ENABLE:
        return run_bayesian_optimization(cfg, logger)
    return train_single_run(cfg, logger, final_eval=True)


if __name__ == "__main__":
    _, cfg = parse_args()
    main(cfg)
