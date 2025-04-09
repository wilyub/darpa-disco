# Shift-Varying Systems Using CNNs

This repository implements an efficient model for **shift-varying systems** based on convolutional neural networks (CNNs) and an unrolled optimization method. The approach models a shift-varying operator as a combination of spatially varying weight maps and convolution kernels (PSFs).

## Model Overview

The forward model is approximated as:

$$
H x \approx \sum_{p=1}^P \text{conv}(c_p^\star) \cdot \text{diag}(w_p^\star) x,
$$

Where:
- \( \text{conv}(c_p^\star) \) is the convolution with the \( p \)-th point spread function (PSF),
- \( w_p^\star \) is the spatially varying weight map associated with the \( p \)-th kernel.

The adjoint of this operator is expressed as:

$$
H^\top x = \sum_{p=1}^P \text{diag}(w_p^\star) \cdot \text{conv}(\bar{c}_p^\star) x,
$$

Where \( \bar{c}_p^\star \) represents the flipped version of the kernel \( c_p^\star \).

## CNN-Based Implementation

### Steps to Implement:

1. **Weight Maps**  
   - Store the spatially varying weights \( w_p^\star \) in a tensor of shape `[P, H, W]`, representing the pixel-wise contribution of each kernel.
   - These weight maps are typically sparse, as the PSFs are localized.

2. **PSF Kernels**  
   - Each \( c_p^\star \) is treated as a separate convolution kernel. Define a convolutional layer with \( P \) filters, each corresponding to one PSF.

3. **Windowed Convolution**  
   - The model computes the operator by modulating the input with the weight maps and performing convolution:
   
   $$
   H x \approx \sum_{p=1}^P (x \odot w_p^\star) \otimes c_p^\star,
   $$

   Alternatively, we use the following formulation:

   $$
   H x \approx \sum_{i=1}^R (x \odot w_i) \otimes k_i.
   $$

## Learning Setup

The model uses **unrolled optimization** with the FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) method for training.

### Main Steps:

1. **Initialization**  
   - Initialize the convolutional filters \( \{ c_p^\star \} \) and weights \( \{ w_p^\star \} \).

2. **FISTA Update**  
   - At each iteration \( k \), update the current estimate \( x^{(k)} \) using:

   $$
   x^{(k+1)} = \text{prox}_{\lambda R}(x^{(k)} - \eta \nabla f(x^{(k)})),
   $$

   where \( R \) is the regularization term and \( f \) is the data fidelity term.

3. **Reconstruction Loss**  
   - Calculate the reconstruction loss as:

   $$
   \mathcal{L}_\text{recon} = \| x_\text{recon} - x_\text{true} \|_2^2.
   $$

4. **Backpropagation**  
   - Use the reconstruction loss to update both the convolutional filters and spatial weight maps.

## Usage

To train the model, follow these steps:

1. Clone the repository:

   ```bash
   git clone 
   cd shift-varying-cnn
   ```
