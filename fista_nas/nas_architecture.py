import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DARTSModel(nn.Module): 

    def __init__(self, input_size, hidden_size, frozen_weight, num_iterations):
        super(DARTSModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.register_buffer('frozen_weight', frozen_weight)
        self.lipschitz = torch.linalg.norm(self.frozen_weight[0], ord=2)**2
        self.eval_flag = False
        self._ops = nn.ModuleList([
            nn.Softshrink(lambd=0.001/self.lipschitz),
            nn.ReLU(),
            nn.Identity(),
            nn.GELU(),
            nn.ELU(),
            nn.Hardtanh(),
            nn.Hardswish(),
            nn.SELU(),
        ])
        self.alpha = nn.Parameter(torch.randn(num_iterations, len(self._ops)))
        self.alpha.data.fill_(1)
    def forward(self, x):
        current_z = nn.Parameter(torch.zeros(x.shape[0], self.hidden_size, 1), requires_grad=False).to(device)
        z_prev = current_z.clone()
        for i in range(self.num_iterations):
            if self.eval_flag:
                alpha_weights = torch.zeros_like(self.alpha[0])
                alpha_weights[self.alpha[0].argmax(dim=0)] = 1
            else:
                alpha_weights = F.softmax(self.alpha[i], dim=0)
            z_aux = self.gradient_operation_first_order(x, current_z, z_prev, i)
            
            z_op = sum(w * op(z_aux) for w, op in zip(alpha_weights, self._ops))
            z_prev = current_z.clone()
            current_z = z_op
        return current_z
    def gradient_operation(self, x, z, i):
        grad = self.frozen_weight.permute(0, 2, 1).matmul(self.frozen_weight.matmul(z)-x)
        update = z - (1/self.lipschitz)*grad
        return update
    def gradient_operation_first_order(self, x, z, z_prev, i):
        z_aux = z + (i / (i + 3)) * (z - z_prev)
        grad = self.frozen_weight.permute(0, 2, 1).matmul(self.frozen_weight.matmul(z_aux)-x)
        update = z_aux - (1/self.lipschitz)*grad
        return update

class DARTSModelSmall(nn.Module):

    def __init__(self, input_size, hidden_size, frozen_weight, num_iterations, ops_version):
        super(DARTSModelSmall, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.register_buffer('frozen_weight', frozen_weight)
        self.lipschitz = torch.linalg.norm(self.frozen_weight[0], ord=2)**2
        self.eval_flag = False
        if ops_version == 1:
            self._ops = nn.ModuleList([
                nn.ReLU(),
                nn.GELU(),
                nn.Sigmoid(),
                nn.SiLU(),
                nn.Hardtanh(),
            ])
        elif ops_version == 2:
            self._ops = nn.ModuleList([
                nn.ELU(),
                nn.GELU(),
                nn.Sigmoid(),
                nn.SiLU(),
                nn.Hardtanh(),
            ])
        elif ops_version == 3:
            self._ops = nn.ModuleList([
                nn.Softshrink(lambd=0.001/self.lipschitz),
                nn.ReLU(),
                nn.GELU(),
                nn.Sigmoid(),
                nn.SiLU(),
                nn.Hardtanh(),
            ])
        self.alpha = nn.Parameter(torch.randn(1, len(self._ops)))
        self.alpha.data.fill_(1)
    def forward(self, x):
        current_z = nn.Parameter(torch.zeros(x.shape[0], self.hidden_size, 1), requires_grad=False).to(device)
        z_prev = current_z.clone()
        for i in range(self.num_iterations):
            if self.eval_flag:
                alpha_weights = torch.zeros_like(self.alpha[0])
                alpha_weights[self.alpha[0].argmax(dim=0)] = 1
            else:
                alpha_weights = F.softmax(self.alpha[0], dim=0)
            z_aux = self.gradient_operation_first_order(x, current_z, z_prev, i)
            z_op = sum(w * op(z_aux) for w, op in zip(alpha_weights, self._ops))
            z_prev = current_z.clone()
            current_z = z_op
        return current_z
    def gradient_operation(self, x, z, i):
        grad = self.frozen_weight.permute(0, 2, 1).matmul(self.frozen_weight.matmul(z)-x)
        update = z - (1/self.lipschitz)*grad
        return update
    def gradient_operation_first_order(self, x, z, z_prev, i):
        z_aux = z + (i / (i + 3)) * (z - z_prev)
        grad = self.frozen_weight.permute(0, 2, 1).matmul(self.frozen_weight.matmul(z_aux)-x)
        update = z_aux - (1/self.lipschitz)*grad
        return update

class DARTSModelLayers(nn.Module): #FISTA Variant
    def __init__(self, input_size, hidden_size, frozen_weight, num_iterations):
        super(DARTSModelLayers, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.register_buffer('frozen_weight', frozen_weight)
        self.lipschitz = torch.linalg.norm(self.frozen_weight[0], ord=2)**2
        self.eval_flag = False
        self._ops = nn.ModuleList([
            nn.Softshrink(lambd=0.001/self.lipschitz),
            nn.ReLU(),
            nn.Identity(),
            nn.GELU(),
            nn.ELU(),
            nn.Hardshrink(lambd=0.001/self.lipschitz),
            nn.Hardtanh(),
            nn.Hardswish(),
            nn.SELU(),
            nn.CELU(),
            nn.LeakyReLU(),
            nn.LogSigmoid(),
            nn.Tanhshrink(),
            nn.Softsign(),
            nn.Softplus(),
            nn.Tanh(),
            nn.Sigmoid(),
            nn.Hardsigmoid(),
            nn.SiLU(),
            nn.Mish()
        ])
        self.alpha = nn.Parameter(torch.randn(num_iterations, len(self._ops)))
        self.alpha.data.fill_(1)
        self.layer_beta = nn.Parameter(torch.ones(num_iterations, 2)) #[skip, calculate] layer options
        
    def forward(self, x):
        current_z = nn.Parameter(torch.zeros(x.shape[0], self.hidden_size, 1), requires_grad=False).to(device)
        z_prev = current_z.clone()
        for i in range(self.num_iterations):
            if self.eval_flag:
                alpha_weights = torch.zeros_like(self.alpha[i])
                alpha_weights[self.alpha[i].argmax(dim=0)] = 1
                beta_weights = torch.zeros_like(self.layer_beta[i])
                beta_weights[self.layer_beta[i].argmax(dim=0)] = 1
            else:
                beta_weights = F.softmax(self.layer_beta[i], dim=0)
                alpha_weights = F.softmax(self.alpha[i], dim=0)

            z_aux = self.gradient_operation_first_order(x, current_z, z_prev, i)
            z_op = sum(w * op(z_aux) for w, op in zip(alpha_weights, self._ops))
            current_z = current_z*beta_weights[0] + z_op * beta_weights[1]
            z_prev = current_z.clone()
            current_z = z_op
        return current_z
    def gradient_operation(self, x, z, i):
        grad = self.frozen_weight.permute(0, 2, 1).matmul(self.frozen_weight.matmul(z)-x)
        update = z - (1/self.lipschitz)*grad
        return update
    def gradient_operation_first_order(self, x, z, z_prev, i):
        z_aux = z + (i / (i + 3)) * (z - z_prev)
        grad = self.frozen_weight.permute(0, 2, 1).matmul(self.frozen_weight.matmul(z_aux)-x)
        update = z_aux - (1/self.lipschitz)*grad
        return update

def create_model(input_size, hidden_size, frozen_weight, num_iterations):
    return DARTSModel(input_size, hidden_size, frozen_weight, num_iterations)
