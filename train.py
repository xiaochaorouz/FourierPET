import argparse
import os
import torch
import matplotlib.pyplot as plt
import shutil
import time

from utils.config import create_config
from trains.unrolling_train import unrolling_train, unrolling_val
from utils.common_config import (get_train_dataset, get_val_dataset, get_projection_simulation,
                                 get_train_transformations, get_model, get_criterion, get_optimizer, get_train_dataloader,
                                 get_val_transformations, get_val_dataloader, get_train_dataloader_LOOCV, get_val_dataloader_LOOCV, get_scheduler)
from termcolor import colored, cprint
import json
import numpy as np


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

FLAGS = argparse.ArgumentParser(description='FourierPET')
FLAGS.add_argument('--config_exp', required=True,
                   help='Path to experiment config file')


def main():
    # Fix random seed
    torch.manual_seed(0)

    # Parse command-line arguments and load YAML configuration
    args = FLAGS.parse_args()
    p = create_config(args.config_exp)
    es_patience = p.get('ES_PATIENCE', 20)
    # Determine if running LOOCV or single-model training
    if 'LOOCV' in p and p['LOOCV']:
        # ----------------------
        # LOOCV TRAINING BRANCH
        # ----------------------

        # base_output_dir: root output directory for all folds
        if 'output_dir' not in p or not p['output_dir']:
            raise ValueError("Please set 'output_dir' in the configuration as the common parent directory for all folds.")
        base_output_dir = p['output_dir']
        os.makedirs(base_output_dir, exist_ok=True)

        # Copy original YAML files into base_output_dir for reproducibility
        try:
            if args.config_exp and os.path.exists(args.config_exp):
                shutil.copy2(args.config_exp,
                             os.path.join(base_output_dir, os.path.basename(args.config_exp)))
                cprint(f"Copied experiment config to: {base_output_dir}", 'green')
        except Exception as e:
            cprint(f"Warning: Failed to copy config files: {e}", 'yellow')

        SSIM_by_fold = []
        PSNR_by_fold = []
        RMSE_by_fold = []
        # Loop over each LOOCV fold
        for fold_idx in range(p['LOOCV_num']):
            # 1) Shallow copy p and override paths for this fold
            fold_config = dict(p)

            # 2) Define the top-level directory for the current fold: LOOCV_0, LOOCV_1, …
            fold_name = f"LOOCV_{fold_idx}"
            fold_dir = os.path.join(base_output_dir, fold_name)
            os.makedirs(fold_dir, exist_ok=True)

            # Within each fold, create subdirectories: logs, figures, checkpoints, models
            logs_dir = os.path.join(fold_dir, "logs")
            figures_dir = os.path.join(fold_dir, "figures")
            checkpoints_dir = os.path.join(fold_dir, "checkpoints")
            os.makedirs(logs_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)
            os.makedirs(checkpoints_dir, exist_ok=True)

            # 3) Override fold_config's paths to point to the fold-specific subdirectories
            fold_config['output_dir'] = fold_dir
            fold_config['figures_base'] = figures_dir
            # "best" checkpoint file (overwrite only when validation improves)
            fold_config['checkpoint'] = os.path.join(checkpoints_dir, f"best_checkpoint_fold{fold_idx}.pth")
            fold_config['checkpoint_last'] = os.path.join(checkpoints_dir, f"last_checkpoint_fold{fold_idx}.pth")
            # Final model file (save at end of training)
            fold_config['model'] = os.path.join(checkpoints_dir, f"final_model_fold{fold_idx}.pth")
            # Tab-separated metrics log file (append one line per epoch)
            fold_config['list_log'] = os.path.join(logs_dir, f"metrics_fold{fold_idx}.txt")

            # Create the fold's main log file to record detailed run info
            fold_log_file = os.path.join(logs_dir, f"log_{fold_idx}.out")
            with open(fold_log_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"LOOCV Fold {fold_idx} — Experiment Configuration:\n")
                f.write(json.dumps(fold_config, indent=4, ensure_ascii=False) + "\n")
                f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")

            # 4) Set up device, model, criterion, optimizer
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            torch.backends.cudnn.benchmark = True

            cprint(f"[Fold {fold_idx}] Loading model...", 'cyan')
            model = get_model(fold_config)
            model_params = sum(param.numel() for param in model.parameters()) / 1e6
            cprint(f"[Fold {fold_idx}] Model: {model.__class__.__name__}, Parameters: {model_params:.2f}M", 'cyan')
            with open(fold_log_file, 'a') as f:
                f.write("\nModel Information:\n")
                f.write(f"  Model Class: {model.__class__.__name__}\n")
                f.write(f"  Parameter Count: {model_params:.2f}M\n")
                f.write(f"  Device: {device}\n")
                f.write("="*80 + "\n")
            model = model.to(device)

            cprint(f"[Fold {fold_idx}] Retrieving criterion...", 'cyan')
            criterion = get_criterion(fold_config)
            cprint(f"[Fold {fold_idx}] Criterion: {criterion.__class__.__name__}", 'cyan')
            criterion = criterion.to(device)

            cprint(f"[Fold {fold_idx}] Retrieving optimizer...", 'cyan')
            optimizer = get_optimizer(fold_config, model)
            with open(fold_log_file, 'a') as f:
                f.write("\nOptimizer Information:\n")
                f.write(f"  Criterion: {criterion.__class__.__name__}\n")
                f.write(f"  Optimizer: {str(optimizer)}\n")
                f.write("="*80 + "\n")

            # Learning Rate Scheduler: ReduceLROnPlateau based on validation SSIM
            if p.get('scheduler', None) is not None:
                scheduler = get_scheduler(p, optimizer)
                cprint(f"[Fold {fold_idx}] Learning Rate Scheduler: {scheduler.__class__.__name__}", 'cyan')
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                mode='max',       # Monitor SSIM (higher is better), so use 'max'
                factor=0.5,       # Multiply LR by 0.5
                patience=5,       # If no improvement for 5 epochs, reduce LR
                verbose=True,
                min_lr=1e-8       # Lower bound for LR (won't go below this)
            )
            with open(fold_log_file, 'a') as f:
                f.write("\nLearning Rate Scheduler: ReduceLROnPlateau (monitor SSIM)\n")
                f.write("  factor=0.5, patience=5, min_lr=1e-8\n")
                f.write("="*80 + "\n")

            # 5) Check if a checkpoint exists to resume training
            last_checkpoint_path = fold_config['checkpoint_last']
            best_checkpoint_path = fold_config['checkpoint']
            
            if os.path.exists(last_checkpoint_path):
                checkpoint = torch.load(last_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                scheduler.load_state_dict(checkpoint['scheduler_state']) if scheduler and checkpoint.get('scheduler_state') else None
                best_ssim = checkpoint.get('best_ssim', 0.0)
                no_improve_count = checkpoint.get('no_improve_count', 0)
                start_epoch = checkpoint.get('epoch', 0)
                fold_idx = checkpoint.get('fold_idx', fold_idx)
                cprint(f"[Fold {fold_idx}] Found last checkpoint, loading...", 'blue')

            elif os.path.exists(best_checkpoint_path):
                checkpoint = torch.load(best_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                scheduler.load_state_dict(checkpoint['scheduler_state']) if scheduler and checkpoint.get('scheduler_state') else None
                best_ssim = checkpoint.get('best_ssim', 0.0)
                no_improve_count = checkpoint.get('no_improve_count', 0)
                start_epoch = checkpoint.get('epoch', 0)
                fold_idx = checkpoint.get('fold_idx', fold_idx)
                cprint(f"[Fold {fold_idx}] Found best checkpoint, loading...", 'blue')

            else:
                cprint(f"[Fold {fold_idx}] No checkpoint found, initializing from scratch...", 'blue')
                best_ssim = 0.0
                no_improve_count = 0
                start_epoch = 0

            cprint(f"[Fold {fold_idx}] Resuming from epoch {start_epoch}, previous best SSIM: {best_ssim:.4f}, no-improve count: {no_improve_count}", 'blue')


            # 6) Build projection simulation, datasets, and dataloaders
            cprint(f"[Fold {fold_idx}] Building projection simulation...", 'magenta')
            projection = get_projection_simulation(fold_config)

            cprint(f"[Fold {fold_idx}] Getting data augmentation transforms...", 'magenta')
            train_transforms = get_train_transformations(fold_config)
            val_transforms = get_val_transformations(fold_config)

            cprint(f"[Fold {fold_idx}] Building datasets and dataloaders...", 'magenta')
            train_dataset = get_train_dataset(fold_config, projection, train_transforms)
            val_dataset = get_val_dataset(fold_config, projection, val_transforms)
            spc = fold_config.get('slices_per_case', len(train_dataset) // fold_config['LOOCV_num'])
            train_dataloader = get_train_dataloader_LOOCV(fold_config, train_dataset, fold_idx, slices_per_case=spc)
            val_dataloader = get_val_dataloader_LOOCV(fold_config, val_dataset, fold_idx, slices_per_case=spc)

            with open(fold_log_file, 'a') as f:
                f.write("\nDataset Information:\n")
                f.write(f"  Training Set Size:   {len(train_dataset)}\n")
                f.write(f"  Validation Set Size: {len(val_dataset)}\n")
                f.write("="*80 + "\n")

            # 7) If metrics log file does not exist, create it and write header
            if not os.path.exists(fold_config['list_log']):
                with open(fold_config['list_log'], 'w') as f:
                    f.write("epoch\tloss\tssim\tpsnr\tssim_val\tpsnr_val\n")

            # 8) Main training loop (with Early Stopping & best-checkpoint saving)
            cprint(f"[Fold {fold_idx}] Starting training from epoch {start_epoch}/{fold_config['epochs']}...", 'green')

            # Containers for plotting metrics if storing in memory
            loss_list = []
            ssim_list = []
            psnr_list = []
            rmse_list = []
            ssim_v_list = []
            psnr_v_list = []
            rmse_v_list = []

            last_epoch = None  # set each iteration; None if range(start_epoch, epochs) is empty
            for epoch in range(start_epoch, fold_config['epochs']):
                last_epoch = epoch
                epoch_start_time = time.time()
                cprint(f"[Fold {fold_idx}][Epoch {epoch+1}/{fold_config['epochs']}", 'yellow')
                cprint("-" * 40, 'yellow')

                # 8.1) Training phase
                cprint(f"[Fold {fold_idx}][Epoch {epoch+1}] Training...", 'white', 'on_blue')
                train_start = time.time()
                loss, ssim, psnr, rmse = unrolling_train(
                    train_dataloader, model, criterion, optimizer, epoch, device, fold_config, pbeam=projection
                )
                train_duration = time.time() - train_start

                loss_list.append(loss)
                ssim_list.append(ssim)
                psnr_list.append(psnr)
                rmse_list.append(rmse)

                # 8.2) Validation phase
                cprint(f"[Fold {fold_idx}][Epoch {epoch+1}] Validation...", 'white', 'on_blue')
                val_start = time.time()
                ssim_v, psnr_v, rmse_v = unrolling_val(
                        val_dataloader, model, criterion, optimizer, epoch, device, fold_config, pbeam=projection
                )

                if (epoch + 1) % p.get('save_every_epoch', 5) == 0:
                    last_checkpoint_path = fold_config['checkpoint_last']
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'best_ssim': best_ssim,
                        'no_improve_count': no_improve_count,
                        'fold_idx': fold_idx
                    }, last_checkpoint_path)
                    cprint(f"[Fold {fold_idx}][Epoch {epoch+1}] Saved last checkpoint → {last_checkpoint_path}", 'blue')
                val_duration = time.time() - val_start

                ssim_v_list.append(ssim_v)
                psnr_v_list.append(psnr_v)
                rmse_v_list.append(rmse_v)
                epoch_duration = time.time() - epoch_start_time

                # Write this epoch's metrics to the fold's main log file
                with open(fold_log_file, 'a') as f:
                    f.write(f"\n[Fold {fold_idx}][Epoch {epoch+1}/{fold_config['epochs']}]\n")
                    f.write(f"  Training Time:   {train_duration:.2f} s\n")
                    f.write(f"  Validation Time: {val_duration:.2f} s\n")
                    f.write(f"  Total Time:      {epoch_duration:.2f} s\n")
                    f.write(f"  Training Metrics -> Loss: {loss:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}, RMSE: {rmse:.4f}\n")
                    f.write(f"  Validation Metrics -> SSIM_val: {ssim_v:.4f}, PSNR_val: {psnr_v:.4f}, RMSE_val: {rmse_v:.4f}\n")
                    f.write("-" * 80 + "\n")

                # Append metrics to the tab-separated file for plotting later
                with open(fold_config['list_log'], 'a') as f:
                    f.write(f"{epoch+1}\t{loss:.6f}\t{ssim:.6f}\t{psnr:.6f}\t{rmse:.6f}\t{ssim_v:.6f}\t{psnr_v:.6f}\t{rmse_v:.6f}\n")

                # 8.3) Learning rate scheduling based on validation SSIM
                if p.get('scheduler', None) is not None:
                    scheduler.step()
                else:
                    scheduler.step(ssim_v)
                # 8.4) Check Early Stopping and update best checkpoint
                if ssim_v > best_ssim:
                    # If validation SSIM improves, update best_ssim, reset counter, and save checkpoint
                    best_ssim = ssim_v
                    no_improve_count = 0

                    checkpoint_dict = {
                        'epoch': epoch + 1,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict() if scheduler else None,
                        'best_ssim': best_ssim,
                        'no_improve_count': no_improve_count,
                        'fold_idx': fold_idx
                    }
                    torch.save(checkpoint_dict, fold_config['checkpoint'])
                    cprint(f"[Fold {fold_idx}][Epoch {epoch+1}] Validation SSIM improved to {best_ssim:.4f}, saved best checkpoint → {fold_config['checkpoint']}", 'blue')
                else:
                    no_improve_count += 1
                    cprint(f"[Fold {fold_idx}][Epoch {epoch+1}] Validation SSIM did not improve (current {ssim_v:.4f}, best {best_ssim:.4f}), no-improve count: {no_improve_count}/{es_patience}", 'yellow')

                # If no improvement for es_patience consecutive epochs, trigger Early Stopping
                if no_improve_count >= es_patience:
                    cprint(f"[Fold {fold_idx}] No validation SSIM improvement for {es_patience} consecutive epochs, triggering Early Stopping.", 'red')
                    break

                # 8.5) Save metric plots (optional: save only every N epochs if desired)
                fig, axs = plt.subplots(3, 2, figsize=(10, 12))
                fig.subplots_adjust(hspace=0.4, wspace=0.3)

                axs[0, 0].plot(loss_list, label="train_loss")
                axs[0, 0].set_title("Training Loss", loc="right")
                axs[1, 0].plot(ssim_list, label="train_ssim")
                axs[1, 0].set_ylim(0.7, 1.0)
                axs[1, 0].set_title("Training SSIM", loc="right")
                axs[2, 0].plot(psnr_list, label="train_psnr")
                axs[2, 0].set_ylim(25, 45)
                axs[2, 0].set_title("Training PSNR", loc="right")

                axs[0, 1].plot(ssim_v_list, label="val_ssim")
                axs[0, 1].set_ylim(0.7, 1.0)
                axs[0, 1].set_title("Validation SSIM", loc="right")
                axs[1, 1].plot(psnr_v_list, label="val_psnr")
                axs[1, 1].set_ylim(25, 45)
                axs[1, 1].set_title("Validation PSNR", loc="right")
                axs[2, 1].plot(rmse_v_list, label="val_rmse")
                axs[2, 1].set_ylim(0, 0.1)
                axs[2, 1].set_title("Validation RMSE", loc="right")

                # Leave axs[2,1] blank or use it for an aggregated metric

                plot_filename = f"metrics_fold{fold_idx}.png"
                plot_path = os.path.join(figures_dir, plot_filename)
                fig.suptitle(f"Fold {fold_idx} Metrics", fontsize=16)
                fig.savefig(plot_path, dpi=200)
                plt.close(fig)
                cprint(f"[Fold {fold_idx}][Epoch {epoch+1}] Saved metric plot → {plot_path}", 'blue')

            # 9) End of this fold's loop (either all epochs done or Early Stopped)
            if last_epoch is None:
                actual_end_epoch = start_epoch
                cprint(
                    f"[Fold {fold_idx}] No training epochs run "
                    f"(start_epoch={start_epoch} >= epochs={fold_config['epochs']}); "
                    f"checkpoint already at or past target. Running validation for fold metrics.",
                    "yellow",
                )
                val_epoch = max(start_epoch - 1, 0)
                ssim_v, psnr_v, rmse_v = unrolling_val(
                    val_dataloader,
                    model,
                    criterion,
                    optimizer,
                    val_epoch,
                    device,
                    fold_config,
                    pbeam=projection,
                )
            elif no_improve_count < es_patience:
                actual_end_epoch = last_epoch + 1
            else:
                actual_end_epoch = last_epoch - no_improve_count + 1

            with open(fold_log_file, 'a') as f:
                f.write("\n")
                if last_epoch is None:
                    f.write(
                        f"Fold {fold_idx}: no new epochs "
                        f"(resume start {start_epoch} >= {fold_config['epochs']}).\n"
                    )
                f.write(f"Fold {fold_idx} training finished, actual epochs completed: {actual_end_epoch}/{fold_config['epochs']}\n")
                f.write(f"Best validation SSIM: {best_ssim:.4f}\n")
                f.write(f"Best checkpoint path: {fold_config['checkpoint']}\n")
                f.write(f"Final model path: {fold_config['model']}\n")
                f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")

            # 10) Save this fold's final model weights (whether Early Stopped or completed all epochs)
            torch.save(model.state_dict(), fold_config['model'])
            cprint(f"[Fold {fold_idx}] Saved final model → {fold_config['model']}", 'green')

            SSIM_by_fold.append(best_ssim)
            PSNR_by_fold.append(psnr_v)
            RMSE_by_fold.append(rmse_v)

        # 11) Calculate and print average metrics across all folds
        avg_ssim = sum(SSIM_by_fold) / len(SSIM_by_fold)
        avg_psnr = sum(PSNR_by_fold) / len(PSNR_by_fold)
        avg_rmse = sum(RMSE_by_fold) / len(RMSE_by_fold)
        ssim_std = np.std(SSIM_by_fold)
        psnr_std = np.std(PSNR_by_fold)
        rmse_std = np.std(RMSE_by_fold)

        cprint("Average Metrics Across All Folds:", 'magenta')
        cprint(f"Average SSIM: {avg_ssim:.4f}", 'magenta')
        cprint(f"Average PSNR: {avg_psnr:.4f}", 'magenta')
        cprint(f"Average RMSE: {avg_rmse:.4f}", 'magenta')
        cprint("="*80, 'magenta')
        with open(os.path.join(p['output_dir'], 'Final_LOOCV_metrics.txt'), 'w') as f:
            f.write("Average Metrics Across All Folds:\n")
            f.write(f"Average SSIM: {avg_ssim:.4f} ± {ssim_std:.4f}\n")
            f.write(f"Average PSNR: {avg_psnr:.4f} ± {psnr_std:.4f}\n")
            f.write(f"Average RMSE: {avg_rmse:.4f} ± {rmse_std:.4f}\n")
            f.write("="*80 + "\n")
        # All folds complete
        cprint("All LOOCV folds have been completed.", 'magenta')
    elif 'LOOCV' not in p or not p['LOOCV']:
        log_file = os.path.join(p['output_dir'], 'log.out')
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("Experiment Configuration:\n")
                f.write(json.dumps(p, indent=4) + "\n")
                f.write("="*80 + "\n")
                f.write(f"Experiment Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")
        else:
            with open(log_file, 'a') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"Resuming training at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")
        es_patience = p.get('ES_PATIENCE', 30)

        # Copy YAML config files to the output directory
        if 'output_dir' in p and p['output_dir']:
            output_experiment_dir = p['output_dir']
            try:
                os.makedirs(output_experiment_dir, exist_ok=True)

                if args.config_exp and os.path.exists(args.config_exp):
                    shutil.copy2(args.config_exp, os.path.join(output_experiment_dir, os.path.basename(args.config_exp)))
                    cprint(f"Copied experiment config to: {output_experiment_dir}", 'green')
            except Exception as e:
                cprint(f"Warning: Failed to copy config files: {e}", 'yellow')
        else:
            cprint("Warning: 'output_dir' path not found in config 'p'. Config files not copied.", 'yellow')

        # Create metrics log file
        metrics_log_file = os.path.join(p['output_dir'], 'metrics.txt')
        if not os.path.exists(metrics_log_file):
            with open(metrics_log_file, 'w') as f:
                f.write("epoch\tloss\tssim\tpsnr\trmse\tssim_val\tpsnr_val\trmse_val\n")

        # Load historical metrics if exists
        loss_list = []
        ssim_list = []
        psnr_list = []
        rmse_list = []
        ssim_v_list = []
        psnr_v_list = []
        rmse_v_list = []
        
        if os.path.exists(metrics_log_file):
            with open(metrics_log_file, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    epoch, loss, ssim, psnr, rmse, ssim_v, psnr_v, rmse_v = map(float, line.strip().split('\t'))
                    loss_list.append(loss)
                    ssim_list.append(ssim)
                    psnr_list.append(psnr)
                    rmse_list.append(rmse)
                    ssim_v_list.append(ssim_v)
                    psnr_v_list.append(psnr_v)
                    rmse_v_list.append(rmse_v)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Model
        cprint("Loading model...", 'cyan')
        model = get_model(p)
        model_params = sum(param.numel() for param in model.parameters()) / 1e6
        cprint(f"Model: {model.__class__.__name__}, Parameters: {model_params:.2f}M", 'cyan')
        with open(log_file, 'a') as f:
            f.write("\nModel Information:\n")
            f.write(f"  Model Class: {model.__class__.__name__}\n")
            f.write(f"  Parameter Count: {model_params:.2f}M\n")
            f.write(f"  Device: {device}\n")
            f.write("="*80 + "\n")
        model = model.to(device)

        # Projection Simulation
        cprint("Building projection simulation...", 'magenta')
        projection = get_projection_simulation(p)
        
        # Data
        cprint("Getting data augmentation transforms...", 'magenta')
        train_transforms = get_train_transformations(p)
        val_transforms = get_val_transformations(p)

        cprint("Building datasets and dataloaders...", 'magenta')
        train_dataset = get_train_dataset(p, projection, train_transforms)
        val_dataset = get_val_dataset(p, projection, val_transforms)

        train_dataloader = get_train_dataloader(p, train_dataset)
        val_dataloader = get_val_dataloader(p, val_dataset)
        with open(log_file, 'a') as f:
            f.write("\nDataset Information:\n")
            f.write(f"  Training Set Size:   {len(train_dataset)}\n")
            f.write(f"  Validation Set Size: {len(val_dataset)}\n")
            f.write("="*80 + "\n")

        # Criterion
        cprint("Retrieving criterion...", 'cyan')
        criterion = get_criterion(p)
        cprint(f"Criterion: {criterion.__class__.__name__}", 'cyan')
        criterion = criterion.to(device)

        # Optimizer and scheduler
        cprint("Retrieving optimizer...", 'cyan')
        optimizer = get_optimizer(p, model)
        cprint(f"Optimizer: {optimizer.__class__.__name__}", 'cyan')

        if p.get('scheduler', None) is not None:
            scheduler = get_scheduler(p, optimizer)
            cprint(f"Learning Rate Scheduler: {scheduler.__class__.__name__}", 'cyan')
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',       # Monitor SSIM (higher is better), so use 'max'
                factor=0.5,       # Multiply LR by 0.5
                patience=5,       # If no improvement for 5 epochs, reduce LR
                verbose=True,
                min_lr=1e-8       # Lower bound for LR (won't go below this)
            )
            cprint("Learning Rate Scheduler: ReduceLROnPlateau (monitor SSIM)", 'cyan')

        with open(log_file, 'a') as f:
            f.write("\nOptimizer Information:\n")
            f.write(f"  Criterion: {criterion.__class__.__name__}\n")
            f.write(f"  Optimizer: {str(optimizer)}\n")
            f.write("="*80 + "\n")

            # 5) Check if a checkpoint exists to resume training
            last_checkpoint_path = p['checkpoint_last']
            best_checkpoint_path = p['checkpoint']
            
            if os.path.exists(last_checkpoint_path):
                checkpoint = torch.load(last_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                scheduler.load_state_dict(checkpoint['scheduler_state']) if scheduler and checkpoint.get('scheduler_state') else None
                best_ssim = checkpoint.get('best_ssim', 0.0)
                no_improve_count = checkpoint.get('no_improve_count', 0)
                start_epoch = checkpoint.get('epoch', 0)
                cprint(f"Found last checkpoint, loading...", 'blue')

            elif os.path.exists(best_checkpoint_path):
                checkpoint = torch.load(best_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                scheduler.load_state_dict(checkpoint['scheduler_state']) if scheduler and checkpoint.get('scheduler_state') else None
                best_ssim = checkpoint.get('best_ssim', 0.0)
                no_improve_count = checkpoint.get('no_improve_count', 0)
                start_epoch = checkpoint.get('epoch', 0)
                cprint(f"Found best checkpoint, loading...", 'blue')

            else:
                cprint(f"No checkpoint found, initializing from scratch...", 'blue')
                best_ssim = 0.0
                no_improve_count = 0
                start_epoch = 0

        # Training
        cprint(f"Starting training from epoch {start_epoch}/{p['epochs']}...", 'green')

        with open(log_file, 'a') as f:
            f.write("\nTraining Process:\n")

        last_epoch = None
        for epoch in range(start_epoch, p['epochs']):
            last_epoch = epoch
            epoch_start_time = time.time()
            cprint(f"Epoch {epoch+1}/{p['epochs']}", 'yellow')
            cprint("-" * 40, 'yellow')

            # Train
            cprint(f"[Epoch {epoch+1}] Training...", 'white', 'on_blue')
            train_start = time.time()
            loss, ssim, psnr, rmse = unrolling_train(
                train_dataloader, model, criterion, optimizer, epoch, device, p, pbeam=projection
            )
            train_duration = time.time() - train_start

            loss_list.append(loss)
            ssim_list.append(ssim)
            psnr_list.append(psnr)
            rmse_list.append(rmse)

            # Val
            cprint(f"[Epoch {epoch+1}] Validation...", 'white', 'on_blue')
            val_start = time.time()
            ssim_v, psnr_v, rmse_v = unrolling_val(
                val_dataloader, model, criterion, optimizer, epoch, device, p, pbeam=projection
            )
            if (epoch + 1) % p.get('save_every_epoch', 5) == 0:
                checkpoint_path = p['checkpoint_last']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict() if scheduler else None,
                    'best_ssim': best_ssim,
                    'no_improve_count': no_improve_count
                }, checkpoint_path)
                cprint(f"[Epoch {epoch+1}] Saved periodic checkpoint → {checkpoint_path}", 'magenta')
            val_duration = time.time() - val_start

            ssim_v_list.append(ssim_v)
            psnr_v_list.append(psnr_v)
            rmse_v_list.append(rmse_v)
            epoch_duration = time.time() - epoch_start_time

            with open(log_file, 'a') as f:
                f.write(f"\n[Epoch {epoch+1}/{p['epochs']}]\n")
                f.write(f"  Training Time:   {train_duration:.2f} s\n")
                f.write(f"  Validation Time: {val_duration:.2f} s\n")
                f.write(f"  Total Time:      {epoch_duration:.2f} s\n")
                f.write(f"  Training Metrics -> Loss: {loss:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}, RMSE: {rmse:.4f}\n")
                f.write(f"  Validation Metrics -> SSIM_val: {ssim_v:.4f}, PSNR_val: {psnr_v:.4f}, RMSE_val: {rmse_v:.4f}\n")
                f.write("-" * 80 + "\n")

            # Save metrics to file
            with open(metrics_log_file, 'a') as f:
                f.write(f"{epoch+1}\t{loss:.6f}\t{ssim:.6f}\t{psnr:.6f}\t{rmse:.6f}\t{ssim_v:.6f}\t{psnr_v:.6f}\t{rmse_v:.6f}\n")

            # Learning rate scheduling based on validation SSIM
            if p.get('scheduler', None) is not None:
                scheduler.step()
            else:
                scheduler.step(ssim_v)

            # Check Early Stopping and update best checkpoint
            if ssim_v > best_ssim:
                best_ssim = ssim_v
                no_improve_count = 0

                checkpoint_dict = {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_ssim': best_ssim,
                    'no_improve_count': no_improve_count
                }
                torch.save(checkpoint_dict, p['checkpoint'])
                cprint(f"[Epoch {epoch+1}] Validation SSIM improved to {best_ssim:.4f}, saved best checkpoint → {p['checkpoint']}", 'blue')
            else:
                no_improve_count += 1
                cprint(f"[Epoch {epoch+1}] Validation SSIM did not improve (current {ssim_v:.4f}, best {best_ssim:.4f}), no-improve count: {no_improve_count}/{es_patience}", 'yellow')

            # If no improvement for es_patience consecutive epochs, trigger Early Stopping
            if no_improve_count >= es_patience:
                cprint(f"No validation SSIM improvement for {es_patience} consecutive epochs, triggering Early Stopping.", 'red')
                break

            # Save metric plots
            fig, axs = plt.subplots(3, 2, figsize=(10, 12))
            fig.subplots_adjust(hspace=0.4, wspace=0.3)

            axs[0, 0].plot(loss_list, label="train_loss")
            axs[0, 0].set_title("Training Loss", loc="right")
            axs[1, 0].plot(ssim_list, label="train_ssim")
            axs[1, 0].set_ylim(0.7, 1.0)
            axs[1, 0].set_title("Training SSIM", loc="right")
            axs[2, 0].plot(psnr_list, label="train_psnr")
            axs[2, 0].set_ylim(25, 45)
            axs[2, 0].set_title("Training PSNR", loc="right")

            axs[0, 1].plot(ssim_v_list, label="val_ssim")
            axs[0, 1].set_ylim(0.7, 1.0)
            axs[0, 1].set_title("Validation SSIM", loc="right")
            axs[1, 1].plot(psnr_v_list, label="val_psnr")
            axs[1, 1].set_ylim(25, 45)
            axs[1, 1].set_title("Validation PSNR", loc="right")
            axs[2, 1].plot(rmse_v_list, label="val_rmse")
            axs[2, 1].set_ylim(0, 0.1)
            axs[2, 1].set_title("Validation RMSE", loc="right")

            plot_filename = "metrics.png"
            plot_path = os.path.join(p['figures_base'], plot_filename)
            fig.suptitle("Training Metrics", fontsize=16)
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            cprint(f"[Epoch {epoch+1}] Saved metric plot → {plot_path}", 'blue')

        # End of training
        if last_epoch is None:
            actual_end_epoch = start_epoch
            cprint(
                f"No training epochs run (start_epoch={start_epoch} >= epochs={p['epochs']}); "
                f"checkpoint already at or past target.",
                "yellow",
            )
        elif no_improve_count < es_patience:
            actual_end_epoch = last_epoch + 1
        else:
            actual_end_epoch = last_epoch - no_improve_count + 1

        with open(log_file, 'a') as f:
            f.write("\n")
            if last_epoch is None:
                f.write(
                    f"No new epochs (start_epoch={start_epoch} >= epochs={p['epochs']}).\n"
                )
            f.write(f"Training finished, actual epochs completed: {actual_end_epoch}/{p['epochs']}\n")
            f.write(f"Best validation SSIM: {best_ssim:.4f}\n")
            f.write(f"Best checkpoint path: {p['checkpoint']}\n")
            f.write(f"Final model path: {p['model']}\n")
            f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")

        torch.save(model.state_dict(), p['model'])
        cprint(f"Saved final model → {p['model']}", 'green')


if __name__ == "__main__":
    main()
