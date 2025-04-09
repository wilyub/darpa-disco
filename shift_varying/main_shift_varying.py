import sys

import argparse
import random

import wandb
import train
# import val

def parse_list(value):
    # Try parsing as a float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Try parsing as a list of two floats
    try:
        values = [float(x) for x in value.strip('[]').split(',')]
        if len(values) == 2:
            return values
        else:
            raise argparse.ArgumentTypeError("Must provide either a single float or a list [min, max]")
    except ValueError:
        raise argparse.ArgumentTypeError("Must provide either a single float or a list [min, max]")

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Define the arguments and their default values
    parser.add_argument('--num_channels', type=int, default=3, help='Number of channels')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--learning_rate_weights', type=float, default=5e-2, help='Learning rate for weights')
    parser.add_argument('--print_freq', type=int, default=1, help='Frequency of print statements during training')
    parser.add_argument('--save_freq', type=int, default=1, help='Frequency of print statements during training')
    parser.add_argument('--exp_idx', type=str, default="train_6", help='Experiment index')
    parser.add_argument('--results_root', type=str, default='results', help='Results directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batchsize during training')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the saved model')
    parser.add_argument('--num_imgs', type=int, default=-1, help='Number of training examples during training')
    parser.add_argument("--mode", choices=["train", "val", "learn_grouping"], default='train', help="Program mode. Train/Val/Test. ")
    parser.add_argument('--data_dir', type=str, default = "../../Fall2024/DARPA/datasets/DIV2K_train_HR/", help='Path to the saved model')
    parser.add_argument('--val_data_dir', type=str, default = "../../Fall2024/DARPA/datasets/DIV2K_valid_HR/", help='Path to the saved model')
    parser.add_argument('--use_wandb', action='store_true', help="Use Wandb")
    parser.add_argument('--dataset', choices=["div2k", "sidd"], default='div2k', help="Dataset type to be used in training")
    parser.add_argument('--l1_alpha', type=float, default=0, help='L1 Alpha for the filters')
    parser.add_argument('--weight_l1_alpha', type=float, default=0, help='L1 Alpha for the weights')
    parser.add_argument('--meas_loss_alpha', type=float, default=0, help='L1 Alpha ')    
    
    parser.add_argument('--smooth_loss_alpha', type=float, default=0, help='Smoothing loss Alpha ')
    parser.add_argument('--smooth_loss_count', type=int, default=5, help='Smoothing loss count ')
    
    parser.add_argument('--kernel_size', type=int, default=13, help='kernel_size ')
    parser.add_argument('--num_val_imgs', type=int, default=4, help='Batchsize during training')

    parser.add_argument('--num_psfs', type=int, default=2, help='Num of PSFs to learn ')
    parser.add_argument('--max_fista_iters', type=parse_list, default=50, help='Number of max FISTA iterations')
    parser.add_argument('--max_fista_iters_val', type=int, default=100, help='Number of max FISTA iterations')
    parser.add_argument('--kernel_type', choices=['oriented', 'blended', 'random', 'motion', 'gaussian', "MLA"], default='blended', help="Type of measurement blur")
    parser.add_argument('--weight_low_rank_factors', type=int, default=-1, help='Rank for low-rank matrix')
    parser.add_argument('--fista_step_size', type=float, default=0.1, help='FISTA gradient descent step size ')
    parser.add_argument('--non_blind', action='store_true', help='Enable blind deconvolution')
    parser.add_argument('--conv_fft', action='store_true', help='Use FFT based convolution')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--weight_scale_factor', type=int, default=1, help='Scale factor for spatial weights')
    parser.add_argument('--weight_init_func', choices=['random', 'uniform', 'constant'], default='constant', help="Weight init function")
    parser.add_argument('--kernel_init_func', choices=['random', 'delta'], default='delta', help="Kernel init function")
    parser.add_argument('--miniscope_psf_num', type=int, default=32, help='Number of PSFS to use in miniscope model')
    parser.add_argument('--one_step_meas', action='store_true', help='Single step measurement loss')
    parser.add_argument('--one_step_meas_val', action='store_true', help='Single step measurement loss during validation')
    parser.add_argument('--banded_matrix_width', type=int, default=-1, help='Banded matrix')

    # Parse the arguments
    args = parser.parse_args()
    
    if args.use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="shift_varying",
            name=args.exp_idx,

            # track hyperparameters and run metadata
            config=args
        )
    
    if args.mode == 'train':
        train.train(args)
    elif args.mode == 'val':
        train.val(args, model=None)
        pass
    else:
        raise ValueError("Unknowm model ", args.mode)
if __name__ == "__main__":
    main()