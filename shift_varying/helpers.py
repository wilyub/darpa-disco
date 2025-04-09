import torch
import wandb
import matplotlib.pyplot as plt
import os


def log_weights(weights, log_dir="logs", file_name="weight"):
    os.makedirs(log_dir, exist_ok=True)

    # Create a horizontal subplot
    num_weights = len(weights)
    fig, axs = plt.subplots(1, num_weights, figsize=(num_weights * 3, 3))  # Adjust size for better visibility
    if num_weights == 1:  # Handle case where there's only one image
        axs = [axs]
    
    for i, ax in enumerate(axs):
        img = weights[i].detach().cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray', aspect='auto')
        ax.axis('off')
        ax.set_title(f"{file_name} {i}")

    # Save locally
    save_path = os.path.join(log_dir, f"{file_name}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)