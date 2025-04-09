import torch

class FISTA:
    def __init__(self, lam=0, max_iter=100, tol=1e-7, smooth_loss_count = 0, device = 'cuda'):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.smooth_loss_count = smooth_loss_count

    def soft_thresholding(self, x, threshold):
        return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))

    def solve(self, y, operator, x_true = None, x0=None, step_size=1.0, debug = False, max_iter = None, one_step_meas = False):
        x = torch.zeros_like(y).to(self.device) if x0 is None else x0.clone()
        z = x.clone()
        t = torch.tensor(1.)

        x_outs = []

        if one_step_meas:
            return operator.forward(x_true)

        max_iter = max_iter if max_iter is not None else self.max_iter
        for i in range(max_iter):
            pred_y = operator.forward(z)
            # Gradient step
            gradient = operator.adjoint(pred_y - y)

            
            x_new = self.soft_thresholding(z - step_size * gradient, self.lam * step_size)

            # Update momentum term
            t_new = (1 + torch.sqrt(1 + 4 * t**2)) / 2
            z = x_new + ((t - 1) / t_new) * (x_new - x)

            # Check for convergence
            if torch.norm(x_new - x) < self.tol:
                break

            x, t = x_new, t_new

            if self.smooth_loss_count > 0 and (max_iter - i) <= self.smooth_loss_count:
                x_outs.append(x)
            # 
            mse = 0
            
            if x_true is not None:
                mse = torch.mean((x_true - x)**2)
            
            if debug:
                print(f"{i}: ", f"Recon MSE: {mse.item()}" if mse > 0 else "", 
                    f"Max gradient: {gradient.max().item()}")

        return x, pred_y, x_outs

