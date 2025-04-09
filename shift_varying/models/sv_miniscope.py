import mat73
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import zoom
from scipy.signal import convolve2d, fftconvolve
from scipy.linalg import svd
from skimage import data

import torch
import torch.nn.functional as F

from utils import gen_patterns

class SVMiniscope:
    def __init__(self, pad_size = 7, merge_lenses = False, num_psfs = 32,
                 b_lists_path="data/b_lists.npy", m_lists_path="data/m_lists.npy", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_size = pad_size
        self.merge_lenses = merge_lenses
        self.num_psfs = num_psfs
        
        self.b_lists = torch.tensor(np.load(b_lists_path), dtype=torch.float32, device=self.device)[:,:num_psfs]
        self.m_lists = torch.tensor(np.load(m_lists_path), dtype=torch.float32, device=self.device)[:,:num_psfs]
        
        assert self.b_lists.shape[:2] == self.m_lists.shape[:2], "Mismatch in B and M list shapes"
    
    def forward(self, img): 
        return self.__call__(img)
    
    def get_true_weights(self):
        M = self.m_lists.shape[0]
        H, W = 256,256
        masks =  self.m_lists[M // 2] if not self.merge_lenses else self.m_lists
        # print("Masks shape is ", masks.shape)
        pad_size = self.pad_size
        mask_interp_size = (H + 2 * pad_size) * 4, (W + 2 * pad_size) * 4
        masks_resized = F.interpolate(masks.unsqueeze(0), size=mask_interp_size, mode='bicubic', align_corners = False)
        _, _, H_orig, W_orig = masks_resized.shape
        
        start_h = (H_orig - H) // 2
        start_w = (W_orig - W) // 2
        
        masks_resized = masks_resized[0, :, start_h:start_h + H, start_w:start_w + W ]
            
        return masks_resized
        
    def get_true_filters(self):
        M = self.b_lists.shape[0]
        normalized_psfs = self.b_lists # / self.b_lists.sum(dim=(-2, -1), keepdim=True)  # Normalize across P, Q

        return normalized_psfs[M // 2] if not self.merge_lenses else normalized_psfs
        
    def __call__(self, img):
        M, K, P, Q = self.b_lists.shape
        M, K, X, Y = self.m_lists.shape
        B, C, H, W = img.shape

        pad_size = self.pad_size
        
        output = torch.zeros_like(img).to(self.device)

        m_iter = [M//2] if not self.merge_lenses else range(M)
        for m in m_iter:
        # for m in range(M):
            masks = self.m_lists[m]
            psfs = self.b_lists[m]
        
            # Upscale M_list using bicubic interpolation
            mask_interp_size = (H + 2 * pad_size) * 4, (W + 2 * pad_size) * 4
            masks_resized = F.interpolate(masks.unsqueeze(0), size=mask_interp_size, mode='bicubic', align_corners = False)
            _, _, H_orig, W_orig = masks_resized.shape
            H_new = H + 2 * pad_size
            W_new = W + 2 * pad_size

            start_h = (H_orig - H_new) // 2
            start_w = (W_orig - W_new) // 2
            
            masks_resized = masks_resized[0, :, start_h:start_h + H_new, start_w:start_w + W_new ].unsqueeze(1).unsqueeze(1)
            
            padded_image =  F.pad(img, (pad_size, pad_size, pad_size, pad_size))
            # Compute weighted object maps
            weighted_imgs = masks_resized * padded_image.unsqueeze(0)
            filter_pad_size = (H + 2 * pad_size - P) // 2
            psfs_padded = F.pad(psfs, (filter_pad_size, filter_pad_size, filter_pad_size, filter_pad_size))
            
            weighted_imgs_fft = torch.fft.fftn(weighted_imgs, dim = (-2,-1))
            psfs_fft = torch.fft.fftn(torch.fft.ifftshift(psfs_padded, dim = (-2,-1)), dim = (-2,-1)).unsqueeze(1).unsqueeze(1)

            conv_result = torch.fft.ifftn(weighted_imgs_fft * psfs_fft, dim=(-2, -1)).real.sum(axis=0)
            
            output += conv_result[:,:, pad_size : - pad_size, pad_size : - pad_size]
            
        # print(torch.max(output, dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0])
        # output -= torch.min(output, dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        # output /= torch.max(output, dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        
        # Divide measurements
        # output /= 1e6
        return output
    
def max_psf_width(psfs, lens_idx):
    """
    Computes the maximum width of the nonzero PSF region from the center.

    Args:
        psfs (numpy.ndarray): Cropped PSF array of shape (H, W, P, P, L).
        lens_idx (int): Index of the lens to analyze.

    Returns:
        float: Maximum nonzero width of the PSF.
    """

    # Extract the PSF slice for the given lens
    H, W, P, _, L = psfs.shape
    psf_2d = np.mean(psfs[:, :, :, :, lens_idx], axis=(2, 3))  # Aggregate over (15, 15)

    # Find the center
    center_x, center_y = H // 2, W // 2

    # Get indices of nonzero values
    # nonzero_indices = np.argwhere(psf_2d > 0)

    # if nonzero_indices.size == 0:
    #     return 0  # If all zeros, width is 0

    # # Compute absolute distances separately (no square)
    # abs_dist_x = np.abs(nonzero_indices[:, 0] - center_x)
    # abs_dist_y = np.abs(nonzero_indices[:, 1] - center_y)

    # # Maximum absolute distance in either x or y direction
    # max_distance = np.max([abs_dist_x.max(), abs_dist_y.max()])

    # # Return full width (diameter)
    # return 2 * max_distance
    x = np.arange(H).reshape(-1, 1)
    y = np.arange(W).reshape(1, -1)
    
    # Compute intensity-weighted means (center of mass)
    total_intensity = np.sum(psf_2d)
    mean_x = np.sum(x * np.sum(psf_2d, axis=1, keepdims=True)) / total_intensity
    mean_y = np.sum(y * np.sum(psf_2d, axis=0, keepdims=True)) / total_intensity

    # Compute intensity-weighted standard deviations
    std_x = np.sqrt(np.sum(((x - mean_x) ** 2) * np.sum(psf_2d, axis=1, keepdims=True)) / total_intensity)
    std_y = np.sqrt(np.sum(((y - mean_y) ** 2) * np.sum(psf_2d, axis=0, keepdims=True)) / total_intensity)

    # Effective width as 4σ (covers ~95% of the PSF spread)
    effective_width = 4 * max(std_x, std_y)

    return effective_width
    
import numpy as np
import matplotlib.pyplot as plt

def truncated_svd_approx(psfs, lens_idx, max_rank=50, save_path="psfs/cropped_psf_svd_approx.png"):
    # Extract PSF for the given lens index
    psf = psfs[..., lens_idx]  # Shape: (H, W, P, P)

    # Reshape to 2D: (H*W, P*P) for SVD
    H, W, P, _ = psf.shape
    psf_reshaped = psf.reshape(H * W, P * P)

    # Compute SVD
    U, S, Vt = np.linalg.svd(psf_reshaped, full_matrices=False)

    errors = []
    norm_H = np.linalg.norm(psf_reshaped, 'fro') ** 2  # Frobenius norm squared
    
    for k in range(1, max_rank + 1):
        S_trunc = np.zeros_like(S)
        S_trunc[:k] = S[:k]
        psf_approx = (U * S_trunc) @ Vt
        error = np.linalg.norm(psf_reshaped - psf_approx, 'fro') ** 2 / norm_H  # NMSE
        errors.append(error)

    # Create subplots for approximation error and singular values
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot approximation error
    axs[0].plot(range(1, max_rank + 1), errors, marker='o')
    axs[0].set_xlabel("Truncation Rank")
    axs[0].set_ylabel("NMSE")
    axs[0].set_title(f"PSF SVD Approximation Error (Lens {lens_idx})")
    axs[0].grid()

    # Plot singular values
    axs[1].semilogy(range(1, len(S) + 1), S, marker='o')  # Log scale for better visualization
    axs[1].set_xlabel("Singular Value Index")
    axs[1].set_ylabel("Singular Value ")
    axs[1].set_title(f"Singular Values (Lens {lens_idx})")
    axs[1].grid()

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    
def reconstruct_image(psfs, object_image, K=32):
    """
    Perform SVD decomposition on cropped PSFs, reconstruct image using
    coefficient maps and basis PSFs.

    Parameters:
    - psfs: NumPy array of shape (240, 240, 15, 15) -> Cropped PSFs
    - object_image: NumPy array of shape (240, 240) -> Object fluorescence distribution
    - K: Number of singular values to retain

    Returns:
    - g_total: Reconstructed image of shape (240, 240)
    """

    # Step 1: Reshape PSFs into 2D matrix (57,600, 225)
    p, q, x, y = psfs.shape  # (240, 240, 15, 15)
    H,W = object_image.shape[:2]
    psfs_reshaped = psfs.reshape(p * q, x * y)  # Shape: (57,600, 225)

    # Step 2: Compute Singular Value Decomposition (SVD)
    U, S, Vt = svd(psfs_reshaped, full_matrices=False)

    # Step 3: Truncate SVD components to retain top K singular values
    U_K = U[:, :K]  # (57,600, K)
    S_K = np.diag(S[:K])  # (K, K)
    Vt_K = Vt[:K, :]  # (K, 225)

    # Step 4: Extract Basis PSFs B_k and reshape to (240, 240)
    B_list = [U_K[:, k].reshape(p, q) for k in range(K)]  # List of 32 PSFs (240, 240)

    # Step 5: Extract Coefficient Maps M_k and reshape to (15, 15)
    M_list = [(S_K[k, k] * Vt_K[k, :]).reshape(x, y) for k in range(K)]  # (15, 15)
    
    # Step 6: Upscale M_k from (15, 15) → (240, 240) using bicubic interpolation
    M_list_resized = [zoom(M_k, H/15, order=3) for M_k in M_list]  # Bicubic upscale

    # Step 7: Compute element-wise multiplication with object image
    weighted_objects = [M_resized[:,:,None] * object_image for M_resized in M_list_resized]

    # Step 8: Perform Convolution with Basis PSFs
    convolved_results = [
        fftconvolve(weighted_obj, B_k[:,:,None], mode="same", axes=(0, 1))  # or specify the correct axes based on your data
        for weighted_obj, B_k in zip(weighted_objects, B_list)
    ]

    # Step 9: Sum over all K components to get final reconstructed image
    g_total = np.sum(convolved_results, axis=0)
    
    g_total -= g_total.min()
    g_total /= g_total.max()

    fig, axs = plt.subplots(2, 10, figsize=(15, 5))

    # Plot the top 3 images from B_list (PSFs)
    for i in range(10):
        axs[0, i].imshow(B_list[i] / B_list[i].max(), cmap='gray')
        axs[0, i].set_title(f'B_list[{i}]')
        axs[0, i].axis('off')  # Hide axes for better visual appeal

    # Plot the top 3 images from M_list (Coefficient Maps)
    for i in range(10):
        axs[1, i].imshow(M_list[i] / M_list[i].max(), cmap='viridis')
        axs[1, i].set_title(f'M_list[{i}]')
        axs[1, i].axis('off')  # Hide axes for better visual appeal

    # Adjust layout for better spacing
    plt.tight_layout()
    
    plt.savefig(f"psfs/b_mlist_{psf_idx}.jpg")
    # plt.close()
    return g_total, (np.array(B_list), np.array(M_list))

def visualize_psfs(cropped_psfs, lens_idx=4, k=5):
    """
    Visualizes a kxk grid of 240x240 PSFs sampled from the cropped PSF array.

    Parameters:
        cropped_psfs (numpy.ndarray): The full PSF array of shape (240, 240, 15, 15, 9).
        lens_idx (int): Index of the microlens to visualize.
        k (int): Grid size (k x k).
    """
    # Select kxk evenly spaced scan positions within the 15x15 grid
    scan_indices = np.linspace(0, 14, k, dtype=int)
    selected_positions = [(i, j) for i in scan_indices for j in scan_indices]

    fig, axes = plt.subplots(k, k, figsize=(10, 10))

    for ax, (i, j) in zip(axes.ravel(), selected_positions):
        psf = cropped_psfs[:, :, i, j, lens_idx]
        ax.imshow(psf, cmap='gray')
        ax.set_title(f"Scan ({i}, {j})")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig('./psfs/cropped_psf_samples.jpg')

# Example Usage:
if __name__ == "__main__":
    
    mat_data = mat73.loadmat('data/cropped_psfs_2d.mat')
    cropped_psfs = mat_data['psfs'] 
    
    # visualize_psfs(cropped_psfs, lens_idx=4, k=5)
    truncated_svd_approx(cropped_psfs, lens_idx = 4, max_rank = 225)
    max_psf_width = max_psf_width(cropped_psfs, lens_idx = 4)
    print("Max psf width for lens 4 ", max_psf_width)
    
    sums = np.sum(cropped_psfs, axis=(0, 1))
    cropped_psfs = np.where(sums != 0, cropped_psfs / sums, 0)
    print("Any NaNs?", np.isnan(cropped_psfs).any())
    print("Any Infs?", np.isinf(cropped_psfs).any())
    # cropped_psfs = cropped_psfs / np.sum(cropped_psfs, axis=(0, 1))  

    # # psf_h, psf_w = cropped_psfs.shape[:2]

    # # all_psfs = cropped_psfs.reshape(psf_h, psf_w, -1)
    cameraman = data.camera()
    astronaut = data.astronaut()
    pad_width = 32

    # # # Pad the image
    object_image = np.pad(astronaut, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
    print("object image shape ", object_image.shape)
    reconstructed_image = 0
    b_lists, m_lists = [] , []
    # Perform reconstruction
    for psf_idx in range(9):
    # for psf_idx in [0, 8]:
        cur_recon, (b_list, m_list) =  reconstruct_image(cropped_psfs[:,:,:,:,psf_idx], object_image, K = 128)

        b_lists.append(b_list)
        m_lists.append(m_list)
        print("Cur recon shape ", cur_recon.shape)
        plt.imshow(cur_recon)
        plt.savefig(f"psfs/recon_img_{psf_idx}.png")
        plt.close()
        
        
        reconstructed_image += cur_recon
    
    print("shapes to save: ", np.array(b_lists).shape , np.array(m_lists).shape)
    
    # np.save("./data/b_lists.npy", np.array(b_lists))
    # np.save("./data/m_lists.npy", np.array(m_lists))
    
    # reconstructed_image /= reconstructed_image.max()
    # # # Display result
    # plt.imshow(reconstructed_image[pad_width:-pad_width, pad_width:-pad_width], cmap="gray")
    # plt.title("Reconstructed Image")
    # plt.colorbar()
    # plt.savefig("psfs/recon_img.png")

    # miniscope = SVMiniscope(merge_lenses=False)
    # # astronaut = astronaut.astype(np.float32) / 255.
    # astronaut = gen_patterns.generate_dot_grid()[:,:,None]
    # astronaut = astronaut.astype(np.float32) / 255.

    # x = torch.from_numpy(astronaut).permute(2,0,1).unsqueeze(0).cuda()
    # x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

    # y = miniscope(x)

    # print("Y is ", y.shape)

    # # plt.imshow(y.squeeze().cpu().permute(1,2,0))
    # plt.imshow(y.squeeze().cpu(), cmap='gray')
    # plt.title("Reconstructed Image")
    # # plt.colorbar()
    # plt.savefig("psfs/recon_img_r4.png")
