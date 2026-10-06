                       


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve, auc

import torch
import torch.nn.functional as F


def extract_model_outputs(model_output):

    if isinstance(model_output, dict):
        return model_output.get('main')
    elif isinstance(model_output, tuple):
        return model_output[0]
    else:
        return model_output


def evaluate_and_collect_predictions(cfg, model, dataloader, device):


    model.eval()

    all_smiles = []
    all_labels = {i: [] for i in range(len(cfg.DATA.TASK_NAME))}
    all_preds = {i: [] for i in range(len(cfg.DATA.TASK_NAME))}

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)

                              
            model_output = model(data)
            main_output = extract_model_outputs(model_output)

                                                                       
            batch_smiles = data.smiles if hasattr(data, 'smiles') else [None] * main_output.shape[0]

                               
            for i in range(len(cfg.DATA.TASK_NAME)):
                if cfg.DATA.TASK_TYPE == 'classification':
                    y_pred = main_output[:, i * 2:(i + 1) * 2]
                    y_label = data.y[:, i].squeeze()

                                       
                    validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]

                    if len(validId) > 0:
                        y_pred = F.softmax(y_pred[validId], dim=-1)[:, 1].cpu().numpy()
                        y_label = y_label[validId].cpu().numpy()

                                                      
                        if i == 0:
                            valid_smiles = [batch_smiles[idx] for idx in validId]
                            all_smiles.extend(valid_smiles)
                    else:
                        continue

                else:              
                    y_pred = main_output[:, i].cpu().numpy()
                    y_label = data.y[:, i].cpu().numpy()

                                             
                    if i == 0:
                        all_smiles.extend(batch_smiles)

                all_labels[i].extend(y_label)
                all_preds[i].extend(y_pred)

    return {
        'smiles': all_smiles,
        'labels': all_labels,
        'predictions': all_preds
    }


def compute_metrics(y_true, y_pred, task_type):

    metrics = {}

    if task_type == 'classification':
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['auc'] = np.nan

        y_pred_binary = (y_pred > 0.5).astype(int)
        metrics['accuracy'] = np.mean(y_true == y_pred_binary)

    else:              
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)

    return metrics


def save_predictions_csv(results, task_names, task_type, output_path):

    df_data = {'smiles': results['smiles']}

    for i, task_name in enumerate(task_names):
        df_data[f'{task_name}_true'] = results['labels'][i]
        df_data[f'{task_name}_pred'] = results['predictions'][i]

        if task_type == 'classification':
            df_data[f'{task_name}_pred_binary'] = (np.array(results['predictions'][i]) > 0.5).astype(int)

    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)

    return df


def save_metrics_csv(results, task_names, task_type, dataset_name, output_path):

    metrics_list = []

    for i, task_name in enumerate(task_names):
        y_true = np.array(results['labels'][i])
        y_pred = np.array(results['predictions'][i])

        metrics = compute_metrics(y_true, y_pred, task_type)
        metrics['task'] = task_name
        metrics['dataset'] = dataset_name
        metrics_list.append(metrics)

                               
    if task_type == 'regression':
        avg_metrics = {
            'task': 'average',
            'dataset': dataset_name,
            'r2': np.mean([m['r2'] for m in metrics_list]),
            'mse': np.mean([m['mse'] for m in metrics_list]),
            'rmse': np.mean([m['rmse'] for m in metrics_list]),
            'mae': np.mean([m['mae'] for m in metrics_list])
        }
    else:
        avg_metrics = {
            'task': 'average',
            'dataset': dataset_name,
            'auc': np.mean([m['auc'] for m in metrics_list if not np.isnan(m['auc'])]),
            'accuracy': np.mean([m['accuracy'] for m in metrics_list])
        }

    metrics_list.append(avg_metrics)

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(output_path, index=False)

    return df_metrics


def create_regression_plots(train_df, val_df, test_df, task_names, output_dir):

    n_tasks = len(task_names)

                                                               
    fig, axes = plt.subplots(n_tasks, 3, figsize=(15, 5 * n_tasks))

    if n_tasks == 1:
        axes = axes.reshape(1, -1)

    datasets = [
        ('Train', train_df, 'blue'),
        ('Validation', val_df, 'green'),
        ('Test', test_df, 'red')
    ]

    for i, task_name in enumerate(task_names):
        for j, (dataset_name, df, color) in enumerate(datasets):
            ax = axes[i, j]

            y_true = df[f'{task_name}_true'].values
            y_pred = df[f'{task_name}_pred'].values

                               
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

                          
            ax.scatter(y_true, y_pred, alpha=0.6, s=25, color=color, edgecolors='black', linewidth=0.5)

                                     
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

                              
            ax.set_xlabel('True Value', fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted Value', fontsize=11, fontweight='bold')
            ax.set_title(f'{task_name} - {dataset_name}\nR²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}',
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'regression_individual_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

                                                  
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5))

    if n_tasks == 1:
        axes = [axes]

    for i, task_name in enumerate(task_names):
        ax = axes[i]

                           
        for df, label, color, marker in [
            (train_df, 'Train', 'blue', 'o'),
            (val_df, 'Validation', 'green', 's'),
            (test_df, 'Test', 'red', '^')
        ]:
            y_true = df[f'{task_name}_true'].values
            y_pred = df[f'{task_name}_pred'].values

            ax.scatter(y_true, y_pred, alpha=0.5, s=35,
                       color=color, marker=marker, label=label, edgecolors='black', linewidth=0.5)

                                                      
        all_true = np.concatenate([
            train_df[f'{task_name}_true'].values,
            val_df[f'{task_name}_true'].values,
            test_df[f'{task_name}_true'].values
        ])
        all_pred = np.concatenate([
            train_df[f'{task_name}_pred'].values,
            val_df[f'{task_name}_pred'].values,
            test_df[f'{task_name}_pred'].values
        ])

                                 
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.5)

        ax.set_xlabel('True Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{task_name} - All Datasets Combined', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'regression_combined_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_classification_plots(train_df, val_df, test_df, task_names, output_dir):

    n_tasks = len(task_names)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5))

    if n_tasks == 1:
        axes = [axes]

    datasets = [
        ('Train', train_df, 'blue'),
        ('Validation', val_df, 'green'),
        ('Test', test_df, 'red')
    ]

    for i, task_name in enumerate(task_names):
        ax = axes[i]

        for dataset_name, df, color in datasets:
            y_true = df[f'{task_name}_true'].values.astype(int)
            y_pred = df[f'{task_name}_pred'].values

                                 
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f'{dataset_name} (AUC = {roc_auc:.4f})')

                       
        ax.plot([0, 1], [0, 1], 'k--', lw=2)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC Curve - {task_name}', fontsize=13, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'classification_roc_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_complete_evaluation_report(cfg, model, train_loader, val_loader, test_loader,
                                        device, output_dir, logger):


    logger.info("=" * 70)
    logger.info("GENERATING COMPLETE EVALUATION REPORT")
    logger.info("=" * 70)

                             
    eval_dir = os.path.join(output_dir, 'final_evaluation')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    logger.info(f"Evaluation results will be saved to: {eval_dir}")

                                              
    logger.info("\n[1/4] Collecting predictions from all datasets...")

    logger.info("  - Evaluating on training set...")
    train_results = evaluate_and_collect_predictions(cfg, model, train_loader, device)
    logger.info(f"    Collected {len(train_results['smiles'])} training samples")

    logger.info("  - Evaluating on validation set...")
    val_results = evaluate_and_collect_predictions(cfg, model, val_loader, device)
    logger.info(f"    Collected {len(val_results['smiles'])} validation samples")

    logger.info("  - Evaluating on test set...")
    test_results = evaluate_and_collect_predictions(cfg, model, test_loader, device)
    logger.info(f"    Collected {len(test_results['smiles'])} test samples")

                                
    logger.info("\n[2/4] Saving predictions to CSV files...")

    train_csv = os.path.join(eval_dir, 'train_predictions.csv')
    train_df = save_predictions_csv(train_results, cfg.DATA.TASK_NAME, cfg.DATA.TASK_TYPE, train_csv)
    logger.info(f"  - Saved: {train_csv}")

    val_csv = os.path.join(eval_dir, 'validation_predictions.csv')
    val_df = save_predictions_csv(val_results, cfg.DATA.TASK_NAME, cfg.DATA.TASK_TYPE, val_csv)
    logger.info(f"  - Saved: {val_csv}")

    test_csv = os.path.join(eval_dir, 'test_predictions.csv')
    test_df = save_predictions_csv(test_results, cfg.DATA.TASK_NAME, cfg.DATA.TASK_TYPE, test_csv)
    logger.info(f"  - Saved: {test_csv}")

                            
    logger.info("\n[3/4] Calculating and saving metrics...")

    train_metrics_csv = os.path.join(eval_dir, 'train_metrics.csv')
    train_metrics = save_metrics_csv(train_results, cfg.DATA.TASK_NAME, cfg.DATA.TASK_TYPE,
                                     'train', train_metrics_csv)
    logger.info(f"  - Saved: {train_metrics_csv}")

    val_metrics_csv = os.path.join(eval_dir, 'validation_metrics.csv')
    val_metrics = save_metrics_csv(val_results, cfg.DATA.TASK_NAME, cfg.DATA.TASK_TYPE,
                                   'validation', val_metrics_csv)
    logger.info(f"  - Saved: {val_metrics_csv}")

    test_metrics_csv = os.path.join(eval_dir, 'test_metrics.csv')
    test_metrics = save_metrics_csv(test_results, cfg.DATA.TASK_NAME, cfg.DATA.TASK_TYPE,
                                    'test', test_metrics_csv)
    logger.info(f"  - Saved: {test_metrics_csv}")

                         
    all_metrics = pd.concat([train_metrics, val_metrics, test_metrics], ignore_index=True)
    all_metrics_csv = os.path.join(eval_dir, 'all_metrics_summary.csv')
    all_metrics.to_csv(all_metrics_csv, index=False)
    logger.info(f"  - Saved: {all_metrics_csv}")

                              
    logger.info("\n[4/4] Creating visualization plots...")

    if cfg.DATA.TASK_TYPE == 'regression':
        create_regression_plots(train_df, val_df, test_df, cfg.DATA.TASK_NAME, eval_dir)
        logger.info(f"  - Saved: regression_individual_plots.png")
        logger.info(f"  - Saved: regression_combined_plot.png")
    else:                  
        create_classification_plots(train_df, val_df, test_df, cfg.DATA.TASK_NAME, eval_dir)
        logger.info(f"  - Saved: classification_roc_curves.png")

                           
    logger.info("\n" + "=" * 70)
    logger.info("METRICS SUMMARY")
    logger.info("=" * 70)
    logger.info("\n" + all_metrics.to_string(index=False))

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION REPORT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"All results saved to: {eval_dir}")
    logger.info("=" * 70)

    return eval_dir


                     


