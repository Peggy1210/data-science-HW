import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GCNConv
from UTILS.utils import from_dgl


class GCNModel(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
    
class simpleGCN(nn.Module):
    """
    Here is a simple implementation of GCN
    """
    def __init__(self, in_size, hid_size, out_size, dropout, num_layers=2):
        super(simpleGCN, self).__init__()
        # self.conv1 = GCNConv(in_size, hid_size)
        # self.conv2 = GCNConv(hid_size, out_size)
        # self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # input layer
        self.convs.append(GCNConv(in_size, hid_size))
        self.bns.append(nn.BatchNorm1d(hid_size))
        # hidden convs
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hid_size, hid_size))
            self.bns.append(nn.BatchNorm1d(hid_size))
        # output layer
        self.convs.append(GCNConv(hid_size, out_size))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(out_size, out_size)
    
    def reset_parameters(self):
        # self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        # data = from_dgl(g)
        # x, edge_index = features, data.edge_index
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)

        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
    
class GCN:
    def __init__(self, data, mask, in_size, out_size, \
            num_layers=3,
            hidden=128,
            epochs=100,  
            lr=0.01,
            weight_decay=0.0005, 
            dropout=0.5, 
            early_stopping=0,  
            normalize_features=True,
            log=False,
            seed=None):
        self.model = simpleGCN(in_size, hidden, out_size, dropout, num_layers)
        self.data = data
        self.mask = mask
        self.epochs = epochs
        # define train/val samples, loss function and optimizer
        self.loss_fcn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.es_iters = early_stopping
        self.print = log

        if seed is not None:
            torch.manual_seed(seed)

        if normalize_features:
            transform = T.NormalizeFeatures(['x', 'edge_attr'])
            self.data = transform(self.data)

    def fit(self):
        # Prepare mask
        train_mask = self.mask['train_mask']
        val_mask = self.mask['val_mask']

        # If early stopping criteria, initialize relevant parameters
        if self.es_iters:
            print("Early stopping monitoring on")
            loss_min = 1e8
            es_i = 0

        cnt = self.epochs / 50

        # training loop
        for epoch in range(self.epochs):
            self.model.train()
            logits = self.model(self.data)
            train_labels = self.data.y[train_mask]
            loss = self.loss_fcn(logits[train_mask], train_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = self.evaluate(val_mask)
            if self.print:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                        epoch, loss.item(), acc
                    )
                )   
            
            val_labels = self.data.y[val_mask]
            val_loss = self.loss_fcn(logits[val_mask], val_labels).item()
            if self.es_iters:
                if val_loss < loss_min:
                    loss_min = val_loss
                    es_i = 0
                else:
                    es_i += 1

                if es_i >= self.es_iters:
                    print(f"\nEarly stopping at epoch={epoch+1}")
                    break
            
            if not self.print:
                if cnt <= 0:
                    cnt = self.epochs / 10
                    print("#", end='')
                else:
                    cnt -= 1
        
        if not self.print:
            print("#")

    def predict(self): 
        train_mask = self.mask['train_mask']
        val_mask = self.mask['val_mask']
        test_mask = self.mask['test_mask']
        self.model.eval()

        train_acc = self.evaluate(train_mask)
        val_acc = self.evaluate(val_mask)

        print("Train Accuracy: {:.4f}\nValidation Accuracy: {:.4f}"
              .format(train_acc, val_acc))
        
        with torch.no_grad():
            logits = self.model(self.data)
            logits = logits[test_mask]
            _, indices = torch.max(logits, dim=1)
            return indices

    def evaluate(self, mask):
        """Evaluate model accuracy"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data)
            logits = logits[mask]
            labels = self.data.y[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)