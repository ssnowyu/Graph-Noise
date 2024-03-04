from torch.nn import Module, ModuleList, BatchNorm1d, ReLU, Linear
import torch
import dgl
from dgl.nn import GraphConv, GATConv, SAGEConv
import torch.nn.functional as F

# import dgl.nn as dglnn


class GCNEncoder(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layer must be greater than 2."

        self.dropout = dropout

        self.conv_in = GraphConv(dim_in, dim_hidden)
        self.conv_out = GraphConv(dim_hidden, dim_out)
        # self.ln_out = LayerNorm(dim_out)
        self.convs = ModuleList()
        self.relu = ReLU()

        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(dim_hidden, dim_hidden))
            # self.bns.append(LayerNorm(dim_hidden))

    def forward(self, g, feat):
        g = dgl.add_self_loop(g)

        h = self.conv_in(g, feat)
        h = self.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        for conv in self.convs:
            h = conv(g, h)
            h = self.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv_out(g, h)

        return h


class GATEncoder(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layer must be greater than 2."

        self.conv_in = GATConv(dim_in, dim_hidden, num_heads=num_heads)

        self.bn_in = BatchNorm1d(dim_hidden * num_heads)
        self.conv_out = GATConv(dim_hidden * num_heads, dim_out, 1)
        self.bn_out = BatchNorm1d(dim_out)
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.relu = ReLU()

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(dim_hidden * num_heads, dim_hidden, num_heads=num_heads)
            )
            self.bns.append(BatchNorm1d(dim_hidden * num_heads))

    def forward(self, g, feat):
        g = dgl.add_self_loop(g)

        h = self.conv_in(g, feat).flatten(-2)
        h = self.bn_in(h)
        h = self.relu(h)

        for conv, bn in zip(self.convs, self.bns):
            h = conv(g, h).flatten(-2)
            h = bn(h)
            h = self.relu(h)

        h = self.conv_out(g, h).flatten(-2)
        h = self.bn_out(h)
        h = self.relu(h)

        return h


class SAGEEncoder(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layer must be greater than 2."

        self.conv_in = SAGEConv(dim_in, dim_hidden, aggregator_type="pool")
        self.bn_in = BatchNorm1d(dim_hidden)
        self.conv_out = SAGEConv(dim_hidden, dim_out, aggregator_type="pool")
        self.bn_out = BatchNorm1d(dim_out)
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.relu = ReLU()

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(dim_hidden, dim_hidden, aggregator_type="pool"))
            self.bns.append(BatchNorm1d(dim_hidden))

    def forward(self, g, feat):
        g = dgl.add_self_loop(g)

        h = self.conv_in(g, feat)
        h = self.bn_in(h)
        h = self.relu(h)

        for conv, bn in zip(self.convs, self.bns):
            h = conv(g, h)
            h = bn(h)
            h = self.relu(h)

        h = self.conv_out(g, h)
        h = self.bn_out(h)
        h = self.relu(h)

        return h


def u_cat_v(edges):
    return {
        "edge_feat": torch.cat((edges.src["src_feat"], edges.src["dst_feat"]), dim=-1)
    }


class MLPLayer(Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int) -> None:
        super().__init__()

        self.linear1 = Linear(dim_in, dim_hidden)
        self.linear2 = Linear(dim_hidden, dim_out)
        self.relu = ReLU()
        self.bn1 = BatchNorm1d(dim_hidden)
        # self.bn2 = BatchNorm1d(dim_out)

    def forward(self, x):
        h = self.linear1(x)
        h = self.bn1(h)
        h = self.relu(h)
        return self.linear2(h)
