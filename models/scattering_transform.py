import torch
from kymatio.torch import Scattering1D


class ScatteringTransform(torch.nn.Module):
    def __init__(self, J, T, Q):
        super(ScatteringTransform, self).__init__()
        self.scattering  = Scattering1D(J, T, Q)
    
    def forward(self, x):
        x = self.scattering(x)
        x = x[:, 1:, :]
        x = torch.log(torch.abs(x) + 1e-6).mean(dim=-1)
        return x

