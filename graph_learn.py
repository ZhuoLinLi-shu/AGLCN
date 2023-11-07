import torch
from torch import nn
from .layer_module import *
from torch.nn.utils import weight_norm


class calc_adj(nn.Module):
    def __init__(self, node_dim, heads, head_dim, nodes=207, eta=1,
                 gamma=0.001, dropout=0.5, n_clusters=5):
        super(calc_adj, self).__init__()

        self.D = heads * head_dim  # node_dim #
        self.heads = heads
        self.dropout = dropout
        self.eta = eta
        self.gamma = gamma

        self.head_dim = head_dim
        self.node_dim = node_dim
        self.nodes = nodes
        self.bn = nn.LayerNorm(node_dim)

    def forward(self, nodevec1, batch_size=64):
        nodevec1 = self.bn(nodevec1)
        adp = self.graph_learn(nodevec1)
        if len(adp.shape) < 3:
            adp = adp.unsqueeze(0).repeat(batch_size, 1, 1)
        gl_loss = None
        return adp, gl_loss

    def graph_learn(self, nodevec):
        resolution_static = torch.einsum('bnd, bdm -> bnm', nodevec, nodevec.transpose(1, 2))
        resolution_static = F.softmax(F.relu(resolution_static), dim=-1)
        return resolution_static


class graph_constructor(nn.Module):
    def __init__(self, nodes, dim, device, time_step, cout=16, heads=4, head_dim=8,
                 eta=1, gamma=0.0001, dropout=0.5, m=0.9, batch_size=64, in_dim=2, is_add1=True):
        super(graph_constructor, self).__init__()
        self.embed1 = nn.Embedding(nodes, dim)

        self.m = m
        self.embed2 = nn.Embedding(nodes, dim)
        for param in self.embed2.parameters():
            param.requires_grad = False
        for para_static, para_w in zip(self.embed2.parameters(), self.embed1.parameters()):
            para_static.data = para_w.data
        self.heads = heads
        self.head_dim = head_dim

        self.out_channel = cout
        self.device = device
        # 这个是词嵌入向量的维度大小
        self.dim = dim
        self.nodes = nodes
        self.time_step = time_step
        if is_add1:
            time_length = time_step + 1
        else:
            time_length = time_step

        self.trans_Merge_line = nn.Conv2d(in_dim, dim, kernel_size=(1, time_length), bias=True) # cout
        self.gate_Fusion_1 = gatedFusion_1(self.dim, device)

        self.calc_adj = calc_adj(node_dim=dim, heads=heads, head_dim=head_dim, nodes=nodes,
                                 eta=eta, gamma=gamma, dropout=dropout)

        self.dim_to_channels = nn.Parameter(torch.zeros(size=(heads * head_dim, cout * time_step)))
        nn.init.xavier_uniform_(self.dim_to_channels.data, gain=1.414)
        self.skip_norm = nn.LayerNorm(time_step)
        self.time_norm = nn.LayerNorm(dim)

    def forward(self, input):
        batch_size, nodes, time_step = input.shape[0], self.nodes, self.time_step

        time_node = input
        # time_node_2 = b, n, dim
        time_node = self.time_norm(self.trans_Merge_line(time_node).squeeze(-1).transpose(1, 2))
        idx = torch.arange(self.nodes).to(self.device)
        nodevec1 = self.embed1(idx)

        nodevec1 = self.gate_Fusion_1(batch_size, nodevec1, time_node) + nodevec1
        adj = self.calc_adj(nodevec1, batch_size)
        return adj


