import torch
import matplotlib.pyplot as plt
import os

def save_side_by_side(reconstruction, measurement, results_root):
    # Ensure tensors are on CPU and converted to numpy
    rec_np = reconstruction.detach().cpu().numpy()
    meas_np = measurement.detach().cpu().numpy()

    batch_size = rec_np.shape[0]

    save_path = results_root
    
    for i in range(batch_size):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Handle single-channel and multi-channel cases
        if rec_np.shape[1] == 1:  # Single-channel (grayscale)
            axes[0].imshow(rec_np[i, 0], cmap='gray')
            axes[1].imshow(meas_np[i, 0], cmap='gray')
        else:  # Multi-channel (assume RGB)
            axes[0].imshow(rec_np[i].transpose(1, 2, 0))
            axes[1].imshow(meas_np[i].transpose(1, 2, 0))
        
        axes[0].set_title('Learned Measurement')
        axes[1].set_title('Actual Measurement')

        for ax in axes:
            ax.axis('off')

        save_file = f'{save_path}_{i}.jpg'
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)