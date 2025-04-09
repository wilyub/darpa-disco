import time 

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import matplotlib.pyplot as plt
from models.sv_miniscope import SVMiniscope
from utils import gen_patterns
from fista import FISTA

def generate_gaussian_kernel(kernel_size, sigma_x, sigma_y, theta):
    """Generate a 2D Gaussian kernel with an arbitrary covariance matrix."""
    k = kernel_size // 2
    x, y = np.meshgrid(np.linspace(-k, k, kernel_size), np.linspace(-k, k, kernel_size))

    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Covariance matrix and apply rotation matrix
    Sigma = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
    Cov = R @ Sigma @ R.T  # Apply rotation

    # Invert covariance matrix
    Cov_inv = np.linalg.inv(Cov)
    det_Cov = np.linalg.det(Cov)

    # Gaussian formula
    exp_term = -0.5 * (x * Cov_inv[0, 0] * x + x * Cov_inv[0, 1] * y + y * Cov_inv[1, 0] * x + y * Cov_inv[1, 1] * y)
    kernel = np.exp(exp_term) / (2 * np.pi * np.sqrt(det_Cov))
    kernel /= kernel.sum()  # Normalize

    return torch.tensor(kernel, dtype=torch.float32)

def get_blending_mask(h, w):
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device), 
        torch.linspace(-1, 1, w, device=x.device),
        indexing="ij"
    )
    radius = torch.sqrt(x_coords**2 + y_coords**2)  # Radial distance from center

    # Define the three regions
    r1, r2 = 1/3, 2/3  # Boundaries for the regions
    pure_blur_mask = (radius <= r1).float()
    no_blur_mask = (radius > r2).float()
    mix_mask = ((radius > r1) & (radius <= r2)).float()
    
    # Smooth interpolation in the mix region
    mix_mask_weight = (radius - r1) / (r2 - r1)
    mix_mask_weight = (1-mix_mask_weight) * mix_mask  # Zero outside the mix region

    # Combine masks to create the final blending mask
    return pure_blur_mask + mix_mask_weight
    
class ShiftVaryingBlur(nn.Module):
    def __init__(self, kernel_size = 15, device = 'cuda', kernel_type='blended', alpha = 0.5, miniscope_psf_num = 32):
        super(ShiftVaryingBlur, self).__init__()
        self.kernel_size = kernel_size
        self.device = device
        
        if kernel_type not in ['oriented', 'blended', 'random', 'motion', 'split_motion', 'gaussian', "MLA"]:
            raise ValueError(f"Invalid kernel_type: {kernel_type}. Use 'oriented' or 'blended'.")
        
        self.kernel_type = kernel_type
        
        if self.kernel_type == 'blended':
            self.kernel = torch.stack([self.get_kernel(self.kernel_size, alpha = 0), self.get_kernel(self.kernel_size, alpha = 1)], dim = 0).to(device)
        elif self.kernel_type == 'gaussian':
            self.kernel = self.generate_gaussian_kernel()
        elif self.kernel_type == 'MLA':
            self.kernel = None
            self.miniscope = SVMiniscope(merge_lenses=False, num_psfs = miniscope_psf_num)
        else:
            self.kernel = self.get_kernel(self.kernel_size, alpha = alpha).to(device)
            print("Kernel is ", self.kernel)
    
    def adjoint_check(self):
        
        x = torch.rand(8,3,256,256).to(self.device)

        Ax = self.forward(x)
        y = torch.rand_like(Ax)
        
        AT_y = self.adjoint(y)
        
        v1 = torch.sum(Ax * y)
        v2 = torch.sum(x * AT_y)
        error = torch.abs(v1 - v2) / torch.max(torch.abs(v1), torch.abs(v2))
        
        assert error < 1e-6, f'"A.T" is not the adjoint of "A". Check the definitions of these operators. Error: {error}'

        print(f"Adjoint check passed: Error: {error}")
        
    def get_kernel(self, kernel_size, alpha):
        """
        Get the appropriate kernel based on the instance's kernel_type.
        """
        if self.kernel_type == 'oriented':
            return self.create_oriented_kernel(kernel_size, alpha)
        elif self.kernel_type == 'blended':
            return self.create_blended_kernel(kernel_size, alpha)
        elif self.kernel_type == 'random':
            return self.create_random_kernel(kernel_size, alpha)
        elif self.kernel_type == 'gaussian':
            return self.create_random_kernel(kernel_size, alpha)
        elif self.kernel_type == 'motion':
            return self.create_motion_blur_kernel(kernel_size, alpha)
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}.")
        
    def create_oriented_kernel(self, kernel_size, alpha):
        """
        Create a kernel with a line centered and oriented smoothly based on alpha.

        :param kernel_size: Size of the square kernel.
        :param alpha: Blending factor (0 for horizontal, 1 for vertical).
        :return: A kernel with a line at the center oriented by alpha.
        """
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        angle = alpha * 90
        radians = np.deg2rad(angle)

        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                x = j - center
                y = i - center
                if abs(y * np.cos(radians) - x * np.sin(radians)) < 0.5:
                    kernel[i, j] = 1

        kernel /= kernel.sum()  # Normalize the kernel
        
        return torch.tensor(kernel, dtype=torch.float32)

    def create_random_kernel(self, kernel_size, alpha):
        torch.manual_seed((alpha*1024))
        kernel = torch.rand(kernel_size, kernel_size)
        # print("kernel ", kernel[kernel_size//2, kernel_size // 2 - 2: kernel_size//2 + 2])
        return kernel / kernel.sum()
        
    def create_blended_kernel(self, kernel_size, alpha):
        """
        Create a blended kernel that transitions smoothly between horizontal and vertical blur.

        :param kernel_size: Size of the square kernel.
        :param alpha: Blending factor (0 for horizontal blur, 1 for vertical blur).
        :return: A blended kernel with a mix of horizontal and vertical blur.
        """
        # Create horizontal blur kernel
        horizontal_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        horizontal_kernel[kernel_size // 2, :] = 1  # Middle row has all ones
        horizontal_kernel /= horizontal_kernel.sum()  # Normalize the kernel

        # Create vertical blur kernel
        vertical_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        vertical_kernel[:, kernel_size // 2] = 1  # Middle column has all ones
        vertical_kernel /= vertical_kernel.sum()  # Normalize the kernel

        # Blend the two kernels
        blended_kernel = alpha * horizontal_kernel + (1 - alpha) * vertical_kernel
        blended_kernel /= blended_kernel.sum()  # Normalize the final kernel

        return torch.tensor(blended_kernel, dtype=torch.float32)
    
    def create_motion_blur_kernel(self, kernel_size, alpha):
        """
        Creates a motion blur kernel of a given size and orientation.

        :param kernel_size: Size of the square kernel (must be odd).
        :param angle: Angle of motion blur in degrees (0 = horizontal, 90 = vertical).
        :return: A 2D motion blur kernel as a PyTorch tensor.
        """
        angle = alpha * 90

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        
        radians = np.deg2rad(angle)
        sin_angle = np.sin(radians)
        cos_angle = np.cos(radians)
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = j - center
                y = i - center
                # Check if the point lies on the motion line within tolerance
                if abs(y * cos_angle - x * sin_angle) < 0.5:
                    kernel[i, j] = 1

        # Normalize the kernel
        kernel /= kernel.sum()
        
        return torch.tensor(kernel, dtype=torch.float32)
    
    def create_gaussian_blur_kernels(self, kernel_size, alpha):
        pass 

    def forward(self, x):
        """
        Apply spatially varying blur with smoothly transitioning orientation.
        
        :param x: Input image tensor of shape (batch_size, channels, height, width).
        :return: Blurred image tensor of the same shape.
        """
        if self.kernel_type == 'motion':
            return self.motion_blur_fwd(x)
        
        if self.kernel_type == 'split_motion':
            return self.motion_blur_fwd(x)
        
        if self.kernel_type == 'gaussian':
            return self.gaussian_blur_fwd(x)
        
        if self.kernel_type == 'MLA':
            return self.miniscope(x)

        batch_size, c, h, w = x.shape
        blurred_img = torch.zeros_like(x)
        
        pad_size = self.kernel_size // 2
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    

        for i in range(h):
            i_pad = i + self.kernel_size // 2
            alpha = i / h  # Transition factor for blur orientation

            # Create the oriented kernel for the current row
            kernel = self.get_kernel(self.kernel_size, alpha)

            kernel = kernel.unsqueeze(0).unsqueeze(0).to(x.device)  # Add batch and channel dimensions
            kernel = kernel.repeat(c, 1, 1, 1)  # Replicate in dim 0 'c' times
    
            image_slice = x_padded[:, :, i_pad - self.kernel_size // 2 : i_pad + self.kernel_size // 2 + 1, :]
            blurred_img[:, :, i, :] = F.conv2d(image_slice, kernel, padding="same", groups = c)[:,:,self.kernel_size // 2, pad_size:-pad_size]

        return blurred_img

    def motion_blur_fwd(self, x):
        batch_size, c, h, w = x.shape
        kernel = self.kernel
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(x.device)  # Add batch and channel dimensions
        kernel = kernel.repeat(c, 1, 1, 1)  # Replicate in dim 0 'c' times

        pad_size = self.kernel_size // 2
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

        blurred_image = F.conv2d(x_padded, kernel, groups = c)
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device), 
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij"
        )
        radius = torch.sqrt(x_coords**2 + y_coords**2)  # Radial distance from center
    
        # Define the three regions
        r1, r2 = 1/3, 2/3  # Boundaries for the regions
        pure_blur_mask = (radius <= r1).float()
        no_blur_mask = (radius > r2).float()
        mix_mask = ((radius > r1) & (radius <= r2)).float()
        
        # Smooth interpolation in the mix region
        mix_mask_weight = (radius - r1) / (r2 - r1)
        mix_mask_weight = (1-mix_mask_weight) * mix_mask  # Zero outside the mix region

        # Combine masks to create the final blending mask
        blending_mask = pure_blur_mask + mix_mask_weight
        # plt.imshow(blending_mask.cpu(), cmap='gray')
        # plt.savefig("z_blend_mask")
        # Blend blurred and original images
        output_image = blending_mask * blurred_image + (1 - blending_mask) * x

        return output_image

    def get_true_filters(self):
        if self.kernel_type == "MLA": 
            return self.miniscope.get_true_filters()
        
        
        raise ValueError(f"get_true_filters not implemented for kernel_type: {self.kernel_type}.")

    def get_weights_filters(self):
        if self.kernel_type == "MLA": 
            return self.miniscope.get_true_weights()
        
        
        raise ValueError(f"get_true_filters not implemented for kernel_type: {self.kernel_type}.")

    def gaussian_blur_fwd(self, x):
        batch_size, c, h, w = x.shape
        kernel_size = self.kernel_size
        
        output_image = torch.zeros_like(x)
        
        # Process each 128x128 patch
        for i in range(0, h, 128):
            for j in range(0, w, 128):
                # Random covariance parameters for each patch
                sigma_x = np.random.uniform(1.0, 5.0)
                sigma_y = np.random.uniform(1.0, 5.0)
                theta = np.random.uniform(0, np.pi)  # Random rotation

                # Generate Gaussian kernel
                kernel = self.generate_gaussian_kernel(kernel_size, sigma_x, sigma_y, theta).to(x.device)
                kernel = kernel.view(1, 1, kernel_size, kernel_size)

                # Apply convolution for each channel
                for b in range(batch_size):
                    for ch in range(c):
                        patch = x[b, ch, i:i+128, j:j+128].unsqueeze(0).unsqueeze(0)
                        blurred_patch = F.conv2d(patch, kernel, padding=kernel_size//2)
                     

        return output_image
    
    def adjoint(self, y):
        """
        Apply the adjoint of the spatially varying blur with smoothly transitioning orientation.

        :param y: Blurred image tensor of shape (batch_size, channels, height, width).
        :return: Reconstructed image tensor of the same shape.
        """
        with torch.enable_grad():         
            # Easier to use VJP 
            # Note for self: Let f(x) = y^TAx -> grad f wrt x =  A^T y
            x = torch.ones_like(y).requires_grad_()
            f = torch.sum(y * self.forward(x))

            return torch.autograd.grad(f, x, create_graph=True)[0]
        '''
            Manual way to compute adjoints. Do not use
        '''
        # batch_size, c, h, w = y.shape
        # adjoint_img = torch.zeros_like(y)

        # pad_size = self.kernel_size // 2
        # y_padded = F.pad(y, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)

        # for i in range(h):
        #     i_pad = i + self.kernel_size // 2
        #     alpha = i / h  # Transition factor for blur orientation

        #     # Create the oriented kernel for the current row
        #     kernel = self.get_kernel(self.kernel_size, alpha)

        #     # Flip the kernel (transposed convolution equivalent in this case)
        #     kernel = torch.flip(kernel, dims=(0,1)).unsqueeze(0).unsqueeze(0).to(y.device)
        #     # kernel = kernel.unsqueeze(0).unsqueeze(0).to(y.device)
        #     kernel = kernel.repeat(c, 1, 1, 1)  # Replicate in dim 0 'c' times

        #     image_slice = y_padded[:, :, i_pad - self.kernel_size // 2 : i_pad + self.kernel_size // 2 + 1, :]
        #     adjoint_img[:, :, i, :] = F.conv2d( image_slice, kernel, padding="same", groups=c)[:, :, self.kernel_size // 2, pad_size:-pad_size]

        # return adjoint_img

if __name__ == "__main__":
    # blur = ShiftVaryingBlur(15, 'cuda', 'blended')
    
    # # Generate stripe pattern
    # h, w = 256, 256
    # stripe_width = 5
    # x = torch.arange(h, device="cuda").unsqueeze(0) // stripe_width % 2  # Binary stripes
    # x = x.float().unsqueeze(0).unsqueeze(0).expand(1, 3, h, w)  # Expand to (1, 3, 256, 256)
    
    # # x = (torch.arange(h, device="cuda").unsqueeze(1) + torch.arange(w, device="cuda").unsqueeze(0)) // stripe_width % 2
    # # x = x.float().unsqueeze(0).unsqueeze(0).expand(1, 3, h, w)  # Expand to (1, 3, 256, 256)
    
    # y = blur(x)
    # print("Y ", y.shape)
    
    # # Visualize and save
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # # Display original stripe pattern
    # axes[0].imshow(x[0].permute(1, 2, 0).cpu().numpy())
    # axes[0].set_title("Stripe Pattern")
    # axes[0].axis("off")
    
    # # Display blurred result
    # axes[1].imshow(y[0].permute(1, 2, 0).cpu().detach().numpy())
    # axes[1].set_title("Blurred Image")
    # axes[1].axis("off")
    
    # # Save the figure
    # plt.tight_layout()
    # plt.savefig("z_measurement_motion.png")
    # plt.close()

    # plt.imshow(blur.get_kernel(15, alpha=0).cpu(), cmap='gray')
    # plt.axis("off")
    # plt.savefig("zz_kernel_alpha_0")
    # plt.imshow(blur.get_kernel(15, alpha=0.25).cpu(), cmap='gray')
    # plt.savefig("zz_kernel_alpha_1")
    # plt.imshow(blur.get_kernel(15, alpha=0.75).cpu(), cmap='gray')
    # plt.savefig("zz_kernel_alpha_2")
    # plt.imshow(blur.get_kernel(15, alpha=1).cpu(), cmap='gray')
    # plt.savefig("zz_kernel_alpha_3")
    
    
    # fig, axes = plt.subplots(2, 3, figsize=(10, 6))  # Adjust grid and size for clarity

    # # Define alpha values
    # alphas = [0, 0.25, 0.5, 0.6, 0.75, 1]

    # # Plot each kernel in the grid
    # for i, (ax, alpha) in enumerate(zip(axes.ravel(), alphas)):
    #     kernel = blur.get_kernel(15, alpha=alpha).cpu()  # Generate the kernel
    #     ax.imshow(kernel, cmap='gray')
    #     ax.set_title(rf"$k_{i}$", fontsize=18)  # Dynamic title for each filter
    #     ax.axis("off")

    # # Adjust layout and save the figure
    # plt.tight_layout()
    # plt.savefig("kernel_grid_6.png", dpi=300)
    # plt.show()

    # # Adjust layout and save the figure
    # plt.tight_layout()
    # plt.savefig("kernel_grid.png", dpi=300)
    # plt.show()
        
    # mask = get_blending_mask(256,256)
    # print("blending mask ", mask[128])


    miniscope = ShiftVaryingBlur(15, 'cuda', 'MLA', miniscope_psf_num = 2)
    x = gen_patterns.generate_dot_grid()[:,:,None]
    x = x.astype(np.float32) / 255.

    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).cuda()

    x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
    y = miniscope(x)
    Aty = miniscope.adjoint(y)

    # Aty -= Aty.min()
    # Aty /= Aty.max()
    x_np = x.squeeze().detach().cpu().numpy()
    y_np = y.squeeze().detach().cpu().numpy()
    Aty_np = Aty.squeeze().detach().cpu().numpy()

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    # Plot each tensor
    titles = ['x', 'y', 'Aty']
    for ax, img, title in zip(axes, [x_np, y_np, Aty_np], titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    miniscope.adjoint_check()

    max_iter = int(200) 
    step_size = 0.5
    

    # FISTA check
    fista = FISTA(max_iter = max_iter)
    x_hat, _ = fista.solve(y, operator=miniscope, step_size = step_size, debug=True, x_true = x)

    # print(torch.sum( (x_hat - x)**2).item())
    # print(x_hat)step_size
    # print(x)
    
    print(x.shape, y.shape, Aty.shape, Aty_np.min(), Aty_np.max(), max_iter, step_size, torch.sum( (x_hat - x)**2).item())
    print(x_hat.min(), x_hat.max())
    
    x_hatnp = x_hat.squeeze().detach().cpu().clamp(0, 1).numpy()
    # x_hatnp -= x_hatnp.min()
    # x_hatnp /= x_hatnp.max()
    
    axes[-1].imshow(x_hatnp, cmap='gray')
    axes[-1].set_title("xhat")
    axes[-1].axis('off')
    # Save to file
    plt.savefig('zz_output2.png', bbox_inches='tight', dpi=300)
    plt.show()