import torch
import torch.nn as nn
from utils import orthog_layer_init

## kinematics input models
class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.num_outputs = num_outputs
        self.affine1 = orthog_layer_init(nn.Linear(num_inputs, 64))
        self.affine2 = orthog_layer_init(nn.Linear(64, 64))
        self.linear3 = orthog_layer_init(nn.Linear(64, num_outputs * 2), std=0.01)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        mu = self.linear3(x)[:, : self.num_outputs]

        # bias of final layer not added for KFAC/EKFAC
        log_std = self.linear3.bias[self.num_outputs :].unsqueeze(0).expand_as(mu)
        std = torch.exp(log_std)

        return mu, log_std, std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = orthog_layer_init(nn.Linear(num_inputs, 64))
        self.affine2 = orthog_layer_init(nn.Linear(64, 64))
        self.value_head = orthog_layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        x = self.value_head(x)
        return x
