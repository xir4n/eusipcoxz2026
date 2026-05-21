import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from murenn import UDTCWT
from murenn.dtcwt.nn import ModulusStable
import murenn
import time
from speechbrain.nnet.normalization import PCEN

class PowerStable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, beta):
        x = x.clamp(min=-beta)
        output = (x + beta) ** alpha 
        ctx.save_for_backward(x, alpha, beta, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta, output = ctx.saved_tensors
        dx, dalpha, dbeta = None, None, None
        if ctx.needs_input_grad[0]:
            dx = grad_output * alpha * ((x+beta) ** (alpha - 1))
            dx.masked_fill_(output == 0, 0)
        if ctx.needs_input_grad[1]:
            dalpha = grad_output * output * torch.log((x+beta))
            dalpha.masked_fill_(output == 0, 0)
        if ctx.needs_input_grad[2]:
            dbeta = grad_output * alpha * ((x+beta) ** (alpha - 1))
            dbeta.masked_fill_(output == 0, 0)
        return dx, dalpha, dbeta


class Downsampling(torch.nn.Module):
    """
    Downsample the input signal by a factor of 2**J_phi.
    --------------------
    Args:
        J_phi (int): Number of levels of downsampling.
    """
    def __init__(self, J_phi):
        super().__init__()
        self.J_phi = J_phi
        # We are using a 13-tap low-pass filter
        self.phi = murenn.DTCWT(
            J=1,
            level1="near_sym_b",
            skip_hps=True,
        )
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        for j in range(self.J_phi):
            x, _ = self.phi(x)
            x = x[:,:,::2]
        return x

class PsiModDn(torch.nn.Module):
    """
    Convolve with psi, then apply modulus, and downsample.
    inputs: (batch, in_channels, time)
    outputs: (batch, in_channels * sum(Q), time / 2**J_phi
    """
    def __init__(self, *, J, Q, T, in_channels, J_phi, use_conv1d=True, use_power=True,):
        super().__init__()
        if isinstance(Q, int):
            self.Q = [Q for j in range(J)]
        elif isinstance(Q, list):
            assert len(Q) == J
            self.Q = Q
        else:
            raise TypeError(f"Q must to be int or list, got {type(Q)}")
        self.T = T
        self.in_channels = in_channels
        self.dtcwt = UDTCWT(
            J=J,
        )
        conv1d = []
        for j in range(J):
            conv1d_j = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.Q[j] * in_channels,
                kernel_size=self.T,
                bias=False,
                padding="same",
                dilation=2**(j -1) if j > 0 else 1,
                groups=in_channels,
            )
            conv1d.append(conv1d_j)
            self.conv1d = torch.nn.ParameterList(conv1d)

        self.down = Downsampling(J_phi)

    def forward(self, x):
        assert self.in_channels == x.shape[1]
        lp, bps = self.dtcwt(x)

        u_psi_x = []
        for j in range(self.dtcwt.J):
            xj = bps[j]
            Wx_j_r = self.conv1d[j](xj.real) / math.sqrt(2) ** j
            Wx_j_i = self.conv1d[j](xj.imag) / math.sqrt(2) ** j
            u_psi_x_j = ModulusStable.apply(Wx_j_r, Wx_j_i)
            u_psi_x_j = self.down(u_psi_x_j)

            u_psi_x.append(u_psi_x_j)
        u_psi_x = torch.cat(u_psi_x, dim=1)
        return u_psi_x


class TwoLayerMurenn(nn.Module):
    def __init__(self, J, Q, T, J_phi, use_power=True):
        super().__init__()
        depth = len(J)
        self.stages = nn.ModuleList([])
        self.phis = nn.ModuleList([])
        self.roots = nn.ParameterList([])
        self.betas = nn.ParameterList([])
        self.mixing = nn.ModuleList([])
        self.kqvs = nn.ModuleList([])
        out_channel_total = 0
        for level in range(depth):
            out_channels = math.prod(Q[:level+1]) * math.prod(J[:level+1])
            if level == 0:
                in_channels = 1
                self.kqvs.append(
                    torch.nn.Linear(
                        in_features=out_channels,
                        out_features=out_channels * 3,
                        bias=False,
                    ))
            else:
                in_channels = math.prod(Q[:level]) * math.prod(J[:level])
            
            self.stages.append(
                PsiModDn(
                    J=J[level],
                    Q=Q[level],
                    T=T[level],
                    in_channels=in_channels,
                    J_phi=J_phi[level],
                )
            )

            out_channel_total += out_channels
            self.roots.append(nn.Parameter(torch.ones(1, out_channels, 1)))
            self.betas.append(nn.Parameter(torch.zeros(1)))

            self.phis.append(
                Downsampling(sum(J_phi[level:]))
            )
            self.sigmoid = nn.Sigmoid()
            self.use_power = use_power
            self.excitation = nn.Sequential(
                nn.Linear(out_channel_total, out_channel_total),
                nn.Sigmoid(),
            )


    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        s_x = [] 
        for level in range(len(self.stages)):
            x = self.stages[level](x)
            if level == 0:
                # Self-attention : input x: (B, C, T)
                x_ = x.permute(0, 2, 1)  # (B, T, C)
                qkv = self.kqvs[level](x_).unsqueeze(1)  # (B, 1, T, 3*C)
                Q, K, V = torch.chunk(qkv, 3, dim=-1)  # Each is (B, 1, T, C)
                x = V.squeeze(1).permute(0, 2, 1) + x
                x_ = nn.functional.scaled_dot_product_attention(Q, K, V).squeeze(1)  # (B, T, C)
                s_x.append(x_.mean(dim=1))
            else:
                if self.use_power:
                    x_alpha = PowerStable.apply(x, self.sigmoid(self.roots[level]), self.betas[level])
                else:
                    x_alpha = x
                s_x.append(x_alpha.mean(dim=-1))
        s_x = torch.cat(s_x, dim=1)
        weight = self.excitation(s_x)
        s_x = s_x * weight
        return s_x.squeeze(1)



if __name__ == '__main__':
    x = torch.zeros(2, 160000)
    J = [8, 6]
    Q = [8, 1]
    T = [16, 1]
    J_phi = [8, 6]
    model = TwoLayerMurenn(J=J, Q=Q, T=T, J_phi=J_phi)
    output = model(x)
    print(output.shape)
    # cout the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')
