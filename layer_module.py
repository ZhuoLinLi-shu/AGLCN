import torch
from torch import nn
import torch.nn.functional as F


class gatedFusion_1(nn.Module):
    def __init__(self, dim, device):
        super(gatedFusion_1, self).__init__()
        self.device = device
        self.dim = dim
        self.w = nn.Linear(in_features=dim, out_features=dim)
        self.t = nn.Parameter(torch.zeros(size=(self.dim, self.dim)))
        nn.init.xavier_uniform_(self.t.data, gain=1.414)
        self.norm = nn.LayerNorm(dim)
        self.re_norm = nn.LayerNorm(dim)

        self.w_r = nn.Linear(in_features=dim, out_features=dim)
        self.u_r = nn.Linear(in_features=dim, out_features=dim)

        self.w_h = nn.Linear(in_features=dim, out_features=dim)
        self.w_u = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, batch_size, nodevec, time_node):
        if batch_size == 1 and len(time_node.shape) < 3:
            time_node = time_node.unsqueeze(0)
        nodevec = self.norm(nodevec)
        node_res = self.w(nodevec) + nodevec
        node_res = node_res.unsqueeze(0).repeat(batch_size, 1, 1)
        time_res = time_node + torch.einsum('bnd, dd->bnd', [time_node, self.t])

        z = torch.sigmoid(node_res + time_res)
        r = torch.sigmoid(self.w_r(time_node) + self.u_r(nodevec).unsqueeze(0).repeat(batch_size, 1, 1))
        h = torch.tanh(self.w_h(time_node) + r * (self.w_u(nodevec).unsqueeze(0).repeat(batch_size, 1, 1)))
        res = torch.add(z * nodevec, torch.mul(torch.ones(z.size()).to(self.device) - z, h))

        return res