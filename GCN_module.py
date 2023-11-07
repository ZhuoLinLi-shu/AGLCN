from torch import nn
import torch
import math
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # X = 32,16,137,181, A = 137, 137
        # x = 32, 16, 137, 181
        x = torch.einsum('ncvl,vw->ncwl', [x, A])
        return x.contiguous()


class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        if len(A.size()) == 2:
            A = A.unsqueeze(0).repeat(x.shape[0], 1, 1)
        # x = torch.einsum('nvw, ncvl->ncwl', [A, x])
        x = torch.einsum('nvw, ncwl->ncvl', [A, x])
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        # x = 32,16,137,181
        x = torch.einsum('ncvl,nvd->ncdl', [x, A])
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)