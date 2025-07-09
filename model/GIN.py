import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from copy import deepcopy
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv, GINConv


class GIN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(GIN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.gc1 = GINConv(
            Sequential(Linear(nfeat, nhid), ReLU(),
                       Linear(nhid, nhid), ReLU()))
        self.gc2 = GINConv(
            Sequential(Linear(nhid, nhid), ReLU(),
                       Linear(nhid, nhid), ReLU()))
        self.gc3 = GINConv(
            Sequential(Linear(nhid, nhid), ReLU(),
                       Linear(nhid, nhid), ReLU()))
        self.lin1 = Linear(nhid*3, nhid*3)
        self.lin2 = Linear(nhid*3, nclass)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h1 = self.gc1(x, edge_index)
        h2 = self.gc2(h1, edge_index)
        h3 = self.gc3(h2, edge_index)
        h = torch.cat((h1, h2, h3), dim=1)
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=self.dropout ,training=self.training)
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)

    def initialize(self):
        """Initialize parameters of GIN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.gc3.reset_parameters()


    def fit(self, pyg_data, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
        """Train the GIN model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """


        if initialize:
            self.initialize()

        self.data = pyg_data[0].to(self.device)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)


    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training GIN model ===')

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)
    def test(self):
        """Evaluate GIN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GIN
        """

        self.eval()
        return self.forward(self.data)
