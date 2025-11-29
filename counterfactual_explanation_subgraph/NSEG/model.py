import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv, GINConv

class GCN(nn.Module):
    def __init__(self, dim_input, dim_hidden, num_classes, dropout=0.5, num_layers=2, mode='node'):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.mode = mode
        self.dropout = dropout

        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * (num_layers - 1)
        self.dim_hidden = dim_hidden

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(dim_input, dim_hidden[0], norm='both', allow_zero_in_degree=True, bias=True))
        for i in range(1, num_layers-1):
            self.convs.append(GraphConv(dim_hidden[i-1], dim_hidden[i], norm='both', allow_zero_in_degree=True, bias=True))

        self.classify = nn.Linear(dim_hidden[-1], num_classes)

    def forward(self, graph, feat, eweight=None):
        h = feat
        for i, conv in enumerate(self.convs):
            h = conv(graph, h, edge_weight=eweight)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.mode == 'node':
            h = self.classify(h)
        elif self.mode == 'graph':
            with graph.local_scope():
                graph.ndata['h'] = h
                hg = dgl.sum_nodes(graph, 'h')
                h = self.classify(hg)
        return h

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

    def accuracy(self, pred, label):
        pred_class = pred.argmax(dim=1)
        correct = pred_class.eq(label).sum().item()
        return correct / len(label)



class GIN(nn.Module):
    def __init__(self, dim_input, dim_hidden, num_classes, dropout=0, num_layers=3, mode='node'):
        super(GIN, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers
        self.mode = mode

        convlayers = []
        mlp_input = nn.Linear(dim_input, dim_hidden[0])
        conv_input = GINConv(mlp_input)
        convlayers.append(conv_input)

        i = -1
        for i in range(num_layers-1):
            mlp = nn.Linear(dim_hidden[i], dim_hidden[i+1])
            conv = GINConv(mlp)
            convlayers.append(conv)

        self.convs = nn.ModuleList(convlayers)
        self.classify = nn.Linear(dim_hidden[i+1], num_classes)


    def forward(self, graph, feat, eweight=None):
        h = self.convs[0](graph, feat, edge_weight=eweight)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        for i in range(self.num_layers-1):
            h = self.convs[i+1](graph, h, edge_weight=eweight)
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(h)

        if self.mode == 'node':
            h = self.classify(h)

        if self.mode == 'graph':
            with graph.local_scope():
                graph.ndata['f'] = h
                hg = dgl.readout_nodes(graph, 'f', op='sum')
                h = self.classify(hg)

        return h

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

    def accuracy(self, pred, label):
        pred = pred.max(1)[1].type_as(label)
        correct = pred.eq(label).double()
        correct = correct.sum()
        return correct / len(label)

