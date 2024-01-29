from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.functional import one_hot
import dgl.nn as dglnn
import dgl

# from torch_geometric.nn import GATConv, GATv2Conv
# from torch_geometric.nn.dense.linear import Linear


class SimpleGCNEncoder(nn.Module):
    def __init__(self, in_size, hid_size, out_size) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        # two layers GCN
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, feat):
        g = dgl.add_self_loop(g)
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class GCN(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        n_layers=5,
        dropout=0.5,
        input_dropout=0.0,
        norm=None,
        n_linear=1,
        spmm_type=0,
        act="F.relu",
        input_layer=False,
        output_layer=False,
        weight_initializer=None,
        bias_initializer=None,
        bias=True,
    ):
        super(GCN, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.n_linear = n_linear
        if norm is None:
            norm = {"flag": False, "norm_type": "LayerNorm"}
        self.norm_flag = norm["flag"]
        self.norm_type = eval("nn." + norm["norm_type"])
        self.act = eval(act)
        if input_layer:
            self.input_linear = nn.Linear(in_features=nfeat, out_features=nhid)
            self.input_drop = nn.Dropout(input_dropout)
        if output_layer:
            self.output_linear = nn.Linear(in_features=nhid, out_features=nclass)
            self.output_normalization = self.norm_type(nhid)
        self.convs = nn.ModuleList()
        if self.norm_flag:
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            if i == 0 and not self.input_layer:
                in_hidden = nfeat
            else:
                in_hidden = nhid
            if i == n_layers - 1 and not self.output_layer:
                out_hidden = nclass
            else:
                out_hidden = nhid

            self.convs.append(
                GraphConvolution(
                    in_hidden,
                    out_hidden,
                    dropout,
                    n_linear,
                    spmm_type=spmm_type,
                    act=act,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer,
                    bias=bias,
                )
            )
            if self.norm_flag:
                self.norms.append(self.norm_type(in_hidden))
        self.convs[-1].last_layer = True

    def forward(self, input):
        x = input[0]
        adj = input[1]
        only_z = input[2] if len(input) > 2 else True
        if self.input_layer:
            x = self.input_linear(x)
            x = self.input_drop(x)
            x = self.act(x)

        for i, layer in enumerate(self.convs):
            if self.norm_flag:
                x_res = self.norms[i](x)
                x_res = layer(x_res, adj)
                x = x + x_res
            else:
                x = layer(x, adj)
            if i != self.n_layers - 1:
                mid = x

        if self.output_layer:
            x = self.output_normalization(x)
            x = self.output_linear(x).squeeze(1)
        if only_z:
            return x.squeeze(1)
        else:
            return mid, x.squeeze(1)
