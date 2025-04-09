import time
import os
import sys
import json
from datetime import datetime
import random 

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiplicativeLR

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

sys.path.append('../../Fall2024/DARPA/non_local_means/')
from data import get_dataloader
from utils import img_utils
from utils.logger_config import get_logger
from utils.img_utils_v2 import save_side_by_side

from piq import psnr
import json

from PIL import Image
import wandb

from shift_varying_model import ShiftVaryingSystemCNN, ShiftVaryingBlur
from fista import FISTA
from helpers import log_weights

def get_max_iters(max_iters, val=False):
    if isinstance(max_iters, float) or isinstance(max_iters, int):
        return int(max_iters)
    
    elif isinstance(max_iters, list) and len(max_iters) == 2:
        if val:
            return int(max_iters[0])
        return random.randint(max_iters[0], max_iters[1])
    
    raise ValueError("max_iters must be either a float or a list [min, max]")

def print_backward_hook(name):
    def hook(grad):
        print(f"Backward called for {name}, gradient is None: {grad is None}")
        return grad  # Optionally modify the gradient here
    return hook

def val(args, model, measurement_operator, fista, epoch = 1, epoch_path = '', logger = None):
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(epoch_path, exist_ok=True)
    
    dataset = args.dataset
    val_data_dir = args.val_data_dir
    batch_size = args.batch_size
    num_val_imgs = args.num_val_imgs
    
    psnr_recons = []
    psnr_recon_with_trues = []
    psnr_measuremensts = []
    measurement_errors = []
              
    
    data_loader = get_dataloader(args.dataset, val_data_dir, batch_size = batch_size, num_imgs = num_val_imgs, shuffle=False, mode='val', train=False)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if dataset == "div2k":
                img, label = batch_data
                img = img.to(device)
            
            else:
                raise ValueError("Unknown dataset")
            
            # print("val image is ", img.shape)
            measurement = measurement_operator(img) 
            
            # if True or args.normalize_meas:
            #     measurement -= torch.min(measurement, dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
            #     measurement /= torch.max(measurement, dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            
            if args.one_step_meas_val:
                reconstruction = fista.solve(measurement, operator=model, step_size = args.fista_step_size, x_true = img, one_step_meas = True)
                
                measurement_error = torch.mean((reconstruction - measurement) ** 2, dim=(1, 2, 3)) 
                
                
                
                reconstruction -= reconstruction.min()
                reconstruction /= reconstruction.max()
                
                
                measurement -= measurement.min()
                measurement /= measurement.max()
                
                save_side_by_side(reconstruction, measurement, f'./{args.results_root}/{args.exp_idx}/train_imgs/z_val_meas_2{epoch}_{batch_idx}')
                
                measurement_errors.append(measurement_error.mean().item())
                
            else:    
                reconstruction, _, _ = fista.solve(measurement, operator=model, step_size = args.fista_step_size, x0 = model.adjoint(measurement), max_iter = args.max_fista_iters_val)
                reconstruction_with_true, _, _  = fista.solve(measurement, operator=measurement_operator, step_size = args.fista_step_size, x0 = measurement_operator.adjoint(measurement), max_iter = args.max_fista_iters_val)

                psnr_recon = psnr(reconstruction.clamp(0,1), img)
                psnr_recon_with_true = psnr(reconstruction_with_true.clamp(0,1), img)  
                psnr_measurement = psnr(measurement.clamp(0,1), img)  

                logger.info(f"Iter: {batch_idx}/{len(data_loader)}: "
                    f"PSNR - Recon: {psnr_recon:.4f}, "
                    f"Recon with True: {psnr_recon_with_true:.4f}, "
                    f"Measurement: {psnr_measurement:.4f}"
                )
        
                psnr_recons.append(psnr_recon.item())
                psnr_recon_with_trues.append(psnr_recon_with_true.item())
                psnr_measuremensts.append(psnr_measurement.item())
                
                for i in range(len(img)):
                    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 6), sharex=True, sharey=True)

                    if measurement[i].max() > 1: 
                        measurement[i] -= measurement[i].min()
                        measurement[i] /= measurement[i].max()
                    
                    ax[0].imshow(img_utils.clamp(measurement[i].cpu().permute(1,2,0)))
                    ax[0].axis('off')
                    ax[0].set_title('measurement')

                    ax[1].imshow(img_utils.clamp(reconstruction_with_true[i].cpu().permute(1,2,0)))
                    ax[1].axis('off')
                    ax[1].set_title('Recon - True F')

                    ax[2].imshow(img_utils.clamp(reconstruction[i].cpu().permute(1,2,0)))
                    ax[2].axis('off')
                    ax[2].set_title('Recon - Learned F')
                    
                    ax[3].imshow(img_utils.clamp(img[i].cpu().permute(1,2,0)))
                    ax[3].axis('off')
                    ax[3].set_title('Ground Truth')

                    fig.tight_layout()
                    img_path = f"{epoch_path}/out_{batch_idx}_{i}.png"
                    plt.savefig(img_path)
                    plt.close()

    model.train()
    
    if args.one_step_meas_val:
        return {
            'val_meas_err_mean': np.array(measurement_errors).mean(),
            'val_meas_err_std': np.array(measurement_errors).std(),
        }
    else:
        return {
            "psnr_recons" : np.array(psnr_recons).mean(),
            "psnr_recon_with_trues" : np.array(psnr_recon_with_trues).mean(),
            "psnr_measuremenst": np.array(psnr_measuremensts).mean()
        }

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = args.learning_rate
    learning_rate_weights = args.learning_rate_weights
    print_freq = args.print_freq
    save_freq = args.save_freq
    exp_idx = args.exp_idx
    results_root = args.results_root
    num_imgs = args.num_imgs
    num_epochs = args.num_epochs
    
    os.makedirs(f'./{results_root}/{exp_idx}/', exist_ok=True)
    os.makedirs(f'./{results_root}/{exp_idx}/outputs', exist_ok=True)
    os.makedirs(f'./{results_root}/{exp_idx}/ckpts', exist_ok=True)
    os.makedirs(f'./{results_root}/{exp_idx}/train_imgs', exist_ok=True)

    args_file_path = f"{results_root}/{exp_idx}/args.json"
    logs_file_path = f"{results_root}/{exp_idx}/logs.txt"
    
    with open(args_file_path, 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    
    logger = get_logger("shift_varying", log_file = logs_file_path)
    
    data_loader = get_dataloader(args.dataset, args.data_dir, batch_size = args.batch_size, num_imgs = num_imgs, num_works = 2)
    
    kernel_size = args.kernel_size
    model = ShiftVaryingSystemCNN(device=device, kernel_size=kernel_size, num_psfs=args.num_psfs,
                                   learn_w=True, weight_low_rank_factors = args.weight_low_rank_factors, 
                                   img_height = 256, img_width = 256, non_blind = args.non_blind, conv_fft = args.conv_fft, 
                                   weight_scale_factor = args.weight_scale_factor, weight_init_func = args.weight_init_func, kernel_init_func = args.kernel_init_func,
                                   banded_matrix_width = args.banded_matrix_width)
    
    measurement_operator = ShiftVaryingBlur(kernel_size=kernel_size, device = device, kernel_type = args.kernel_type, miniscope_psf_num = args.miniscope_psf_num)
    fista = FISTA(device = device, max_iter=get_max_iters(args.max_fista_iters), smooth_loss_count=args.smooth_loss_count)
    
    # if non_blind, fix and set true filters and only learn weights
    if args.non_blind:
        model.true_filters = measurement_operator.get_true_filters()
        print("model.weights shape ", model.weights.data.shape,  measurement_operator.get_weights_filters().shape)
        model.weights.data = measurement_operator.get_weights_filters()
        print("model.weights shape after ", model.weights.data.shape,  measurement_operator.get_weights_filters().shape)

    criterion = torch.nn.MSELoss(reduction="mean")
    # Define different learning rates for filters and weights
    lr_filters = learning_rate
    lr_weights = learning_rate_weights

    # Create parameter groups for the optimizer
    param_groups = []
    
    if model.filters is not None:
       param_groups.append( {"params": [model.filters], "lr": lr_filters})
    
    if model.weights is not None:
        if isinstance(model.weights, nn.ParameterList):
            param_groups.append({"params": model.weights, "lr": lr_weights})
        else:  # Single nn.Parameter case
            param_groups.append({"params": [model.weights], "lr": lr_weights})

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(param_groups)

    def lr_lambda(epoch, min_lr=1e-6):
        if epoch % 10000 == 9999:
            return 0.9
        return 1
        
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    
    if args.model_path:
        checkpoint = torch.load(args.model_path)
        try:
            # Attempt to load the state_dict directly
            res = model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model {args.model_path}. Res: {res}")
        except RuntimeError as e:
            logger.error(f"Error loading model {e}")
            pass
            
    # if torch.cuda.device_count() > 1:  # Check if multiple GPUs are available
    #     model = nn.DataParallel(model)
        
    model = model.to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Learning Weights ({name}): {param.shape}")

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {total_trainable_params}")
    logger.info(f"Total number of trainable parameters: {total_trainable_params}")

    logger.info("-------- Model: ")
    logger.info(model)
    logger.info("-----------------")    

    running_losses = []
    running_psnrs = []
    iterations = []
    best_psnr_recons = float('-inf')
    best_error_meas = float('inf')
    # Dumps args to file as json 
    
    # model.weights.register_hook(print_backward_hook("weights"))
    # model.filters.register_hook(print_backward_hook("filters"))

    # Training Loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_psnr = 0.0
        running_psnr_noisy = 0.0


        for batch_idx, batch_data in enumerate(data_loader):
            # if batch_idx >= 0: break
            if args.dataset == 'div2k':
                (img, label) = batch_data
                
                img = img.to(device)
                measurement = measurement_operator(img)
                
                # if True or args.normalize_meas:
                #     measurement -= torch.min(measurement, dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
                #     measurement /= torch.max(measurement, dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            else:
                raise ValueError("Unknown dataset")
            
            s = time.time()
            optimizer.zero_grad()
            
            if args.debug:
                print("1 measurement norm ", measurement.norm(), measurement.max())
            
            if args.one_step_meas:
                assert args.meas_loss_alpha > 0, "If one_step_means is enabled, meas_loss_alpha must be > 0"
                
                reconstruction = fista.solve(measurement, operator=model, step_size = args.fista_step_size, x_true = img, one_step_meas = True)
                est_meas_loss = criterion(reconstruction, measurement) 
                
                loss_recon_meas = est_meas_loss
                loss_recon = criterion(model.adjoint(measurement), measurement_operator.adjoint(measurement)) 
                
                reconstruction_meas = reconstruction
                # clamp for post-prosessing
                outputs_img = reconstruction.clamp(0, 1)
                # print("outputs img min max: ", outputs_img.min(), outputs_img.max())
                psnr_cur = psnr(outputs_img, measurement.clamp(0, 1)) 
                psnr_meas = psnr(reconstruction_meas.clamp(0,1), measurement.clamp(0, 1))
                smooth_loss = 0 
                if epoch % 50 == 49:
                    save_side_by_side(reconstruction, measurement, f'./{results_root}/{exp_idx}/train_imgs/pred_meas_{epoch}')
            else:
                x_true = img if args.debug else None
                # print("----------------------------------------- recon")
                reconstruction, pred_y, x_outs = fista.solve(measurement, operator=model, step_size = args.fista_step_size, 
                                                             debug = args.debug, x0 = model.adjoint(measurement), x_true = x_true,
                                                             max_iter=get_max_iters(args.max_fista_iters))
                # print("----------------------------------------- true")
                reconstruction_meas, _, _ = fista.solve(measurement, operator=measurement_operator, step_size = args.fista_step_size, 
                                                      debug = args.debug, x0 = measurement_operator.adjoint(measurement), x_true = x_true,
                                                      max_iter=get_max_iters(args.max_fista_iters))
            
                if args.debug :
                    print("recon ", reconstruction.min().item(), reconstruction.max().item())
                    print("GT    ", img.min().item(), img.max().item())
                # exit()
                
                est_meas_loss = criterion(pred_y, measurement) 
                loss_recon = criterion(reconstruction, img)
                loss_recon_meas = criterion(reconstruction_meas, img)

                smooth_loss = 0            
                if len(x_outs) > 0 and args.smooth_loss_alpha > 0:
                    # smooth_loss = torch.sum(torch.tensor([ criterion(x, img) for x in x_outs]))    
                    for i in range(len(x_outs) - 1):
                        smooth_loss += torch.sum((x_outs[i] - x_outs[i+1])**2)  # Penalize large differences between consecutive outputs        
                    smooth_loss /= len(x_outs)

                # clamp for post-prosessing
                outputs_img = reconstruction.clamp(0, 1)
                
                psnr_cur = psnr(outputs_img, img) 
                psnr_meas = psnr(reconstruction_meas.clamp(0,1), img) 
        
            filter_l1_loss = model.filter_norm()
            
            loss = loss_recon + args.l1_alpha * filter_l1_loss + args.meas_loss_alpha * est_meas_loss + args.smooth_loss_alpha * smooth_loss

            if args.weight_l1_alpha > 0:
                weight_l1_loss = model.weight_norm()
                # print("Adding weight l1 alpha ", weight_l1_loss.item(),filter_l1_loss.item() )
                loss += args.weight_l1_alpha * weight_l1_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            e = time.time()

            running_loss += loss.item()
            running_psnr += psnr_cur.item()
            
            if args.debug:
                if isinstance(model.weights, nn.ParameterList):
                    filters_grad = model.filters.grad.norm().item() if model.filters is not None and model.filters.grad is not None else "None"
                    weights_grad_0 = model.weights[0].grad.norm().item() if model.weights[0].grad is not None else "None"
                    weights_grad_1 = model.weights[1].grad.norm().item() if model.weights[1].grad is not None else "None"
                    print("Grads ", filters_grad, weights_grad_0, weights_grad_1)
                else:
                    filters_grad = model.filters.grad.norm().item() if model.filters is not None and model.filters.grad is not None else "None"
                    weights_grad = model.weights.grad.norm().item() if model.weights is not None and model.weights.grad is not None else "None"
                    print("Grads ", filters_grad, weights_grad)
                    print("weights ", model.weights.norm(dim=(-1,-2)))

            # print("Kernel err: ", , f" time: {e-s}")

            if batch_idx % print_freq == print_freq - 1:
                avg_loss = running_loss / print_freq
                avg_psnr = running_psnr / print_freq
                avg_psnr_noisy = running_psnr_noisy / print_freq
                last_lr = scheduler.get_last_lr()       
                
                logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(data_loader)}], Loss: {avg_loss:.5f} ({est_meas_loss:.5f},{filter_l1_loss:.5f}), True loss {loss_recon_meas.item():.2f}, PSNR: {avg_psnr:.2f} vs PSNR TRUE: {psnr_meas.item():2f}. LR:{last_lr}')
                
                running_losses.append(avg_loss)
                running_psnrs.append(avg_psnr)
                iterations.append(epoch * len(data_loader) + batch_idx)  # 
                
                if args.use_wandb:     
                    wandb.log({
                        'avg_loss': avg_loss,
                        'avg_psnr': avg_psnr,
                    })
                    
                
                running_loss = 0.0
                running_psnr = 0.0
                running_psnr_noisy = 0.0

        scheduler.step()

        if epoch % save_freq == save_freq - 1 or epoch + 1 == num_epochs:
            logger.info("="*20 + f" Begin validation @ Epoch {epoch}" +  '='*20)
            
            epoch_path = f"{results_root}/{exp_idx}/outputs/epoch_{epoch}"
            # args, model, measurement_operator, fista = None, measurement = None, x_true = None
            val_result = val(args, model=model, measurement_operator=measurement_operator, fista = fista, epoch=epoch, epoch_path=epoch_path, logger = logger)
            
            if args.one_step_meas_val:
                logger.info(f"Validation Result - Error: Mean = {val_result['val_meas_err_mean']:.4f}, Std = {val_result['val_meas_err_std']:.4f}")
                # exit()
            else:    
                psnr_recons = val_result["psnr_recons"].item()
                psnr_recon_with_trues = val_result["psnr_recon_with_trues"].item()
                psnr_measuremenst = val_result["psnr_measuremenst"].item()
                
                # Todo log weights, log filters , log images 
                
                logger.info(
                    f"Validation Results for Epoch {epoch}: "
                    f"  - VAL (PSNR): {psnr_recons:.4f} "
                    f"  - VAL (PSNR with true filters): {psnr_recon_with_trues:.4f} "
                    f"  - VAL (PSNR Measurement): {psnr_measuremenst:.4f} "
                )
            
            if args.use_wandb:                     
                if args.one_step_meas_val:
                    logs = {
                        "val_meas_err_mean":  val_result['val_meas_err_mean'],
                        "val_meas_err_std": val_result['val_meas_err_std']    
                    }
                else:
                    logs = {  
                            "psnr_learned_F": psnr_recons,
                            "psnr_true_F": psnr_recon_with_trues,
                            "psnr_measurement": psnr_measuremenst
                    }    
                if model.weights is not None: 
                    weights = model.weight_enforcer(model.weights).squeeze()
                    
                    weight_imgs = [ wandb.Image(weights[i].detach().cpu(), caption=f"Learned Weight {i}", mode='L') for i, img in enumerate(range(len(weights))) ]
                    # wandb.log({})
                    logs["Weights"] = weight_imgs
                    log_weights(weights, log_dir = epoch_path, file_name = 'weight')
                 
                if model.filters is not None: 
                    filter_imgs = [ wandb.Image(model.filters[i].squeeze().detach().cpu(), caption=f"Learned filter {i}", mode='L') for i, img in enumerate(range(len(model.filters))) ]   
                    logs["Filters"] = filter_imgs
                    log_weights(model.filters, log_dir = epoch_path, file_name = 'filter')
                elif model.true_filters is not None:   
                    filter_imgs = [ wandb.Image(model.true_filters[i].squeeze().detach().cpu(), caption=f"Learned filter {i}", mode='L') for i, img in enumerate(range(len(model.true_filters))) ]   
                    logs["Filters"] = filter_imgs 
                    log_weights(model.true_filters, log_dir = epoch_path, file_name = 'filter')
                
                image_files = [f for f in os.listdir(epoch_path) if "out" in f and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                image_paths_to_log = [os.path.join(epoch_path, image_file) for image_file in image_files[:10]]

                recond_imgs = [wandb.Image(img, caption=f"Denoised Image {i}") for i, img in enumerate(image_paths_to_log)]
                
                if len(recond_imgs) > 0:
                    logs['Recon'] = recond_imgs
                
                wandb.log(logs)

            if not args.one_step_meas_val:  
                if psnr_recons > best_psnr_recons:
                    logger.info(f"\u2705 Better val psnr found {psnr_recons} >{best_psnr_recons} @ epoch: {epoch}. Saving updated model ...  ")
                    best_psnr_recons = psnr_recons
                    
                    # Saving the model state and optimizer state
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val': {
                            "val_psnr": psnr_recons, 
                        }
                    }, f"./{results_root}/{exp_idx}/ckpts/epoch_{epoch:05d}_val_PSNR_{psnr_recons:3f}.pkl")
                else:
                    logger.info(f"\u274C No PSNR improvement. {psnr_recons} < {best_psnr_recons} ")
            else:
                val_err = val_result['val_meas_err_mean']
                
                if val_err < best_error_meas:
                    logger.info(f"\u2705 Better val psnr found {val_err} < {best_error_meas} @ epoch: {epoch}. Saving updated model ...  ")
                    best_error_meas = val_err

                    # Saving the model state and optimizer state
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val': {
                            "val_meas_err_mean": val_err
                        }
                    }, f"./{results_root}/{exp_idx}/ckpts/epoch_{epoch:05d}_val_PSNR_{val_err:4f}.pkl")
                else:
                    logger.info(f"\u274C No error improvement. { val_err } > {best_error_meas} ")
                
                
                