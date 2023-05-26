###########################################
# 
# Source: 
# Author: @russellizadi
#
###########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_adj

from layers import GCNConv
import psgd as psgd

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

class Net_orig(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout):
        super(Net_orig, self).__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)
        self.p = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, mask=None):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class CRD(nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True) 
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class CLS(nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

class SSPModel(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout):
        super(SSPModel, self).__init__()
        self.crd = CRD(in_size, hid_size, dropout)
        # self.crd2 = CRD(hid_size, hid_size, dropout)
        self.cls = CLS(hid_size, out_size)

    def reset_parameters(self):
        self.crd.reset_parameters()
        # self.crd2.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data, mask):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, mask)
        x = self.cls(x, edge_index, mask)
        return x
    
class SSP:
    def __init__(self, data, in_size, out_size, mask, \
            epochs=100,  
            lr=0.01,
            hidden=64,
            dropout=0.5, 
            normalize_features=True,
            weight_decay=0.0005, 
            str_optimizer='Adam', 
            str_preconditioner=None, 
            momentum=0.9,
            early_stopping=0,  
            eps=0.01,
            update_freq=50,
            gamma=None,
            alpha=None,
            log=False,
            seed=None):
        
        self.model = SSPModel(in_size, hidden, out_size, dropout)
        self.data = data
        self.mask = mask
        self.in_size, self.hid_size, self.out_size = in_size, hidden, out_size
        self.epochs=epochs
        self.seed = seed
        self.es_iters = early_stopping
        self.nonormalize_features = normalize_features,
        self.gamma, self.alpha = gamma, alpha
        self.print = log

        if seed is not None:
            torch.manual_seed(seed)

        if normalize_features:
            transform = T.NormalizeFeatures(['x', 'edge_attr'])
            self.data = transform(self.data)

        self.preconditioner = None
        if str_preconditioner == 'KFAC':
            self.preconditioner = psgd.KFAC(
            self.model.parameters(), 
            eps, 
            sua=False, 
            pi=False, 
            update_freq=update_freq,
            alpha=alpha if alpha is not None else 1.,
            constraint_norm=False
        )

        self.optimizer = None
        if str_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif str_optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=momentum,
            )
    
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
        for epoch in range(1, self.epochs+1):
            lam = (float(epoch)/float(self.epochs))**self.gamma if self.gamma is not None else 0.
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(self.data, train_mask)
            
            label = logits.max(1)[1]
            label[train_mask] = self.data.y[train_mask]
            label.requires_grad = False

            loss = F.nll_loss(logits[train_mask], label[train_mask])
            loss += lam * F.nll_loss(logits[~train_mask], label[~train_mask])
            
            loss.backward(retain_graph=True)

            if self.preconditioner:
                self.preconditioner.step(lam=lam)
            self.optimizer.step()

            acc = self.evaluate(val_mask)
            if self.print:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                        epoch, loss.item(), acc
                    )
                )
            
            val_loss = F.nll_loss(logits[val_mask], label[val_mask]).item()
            if self.es_iters:
                if val_loss < loss_min:
                    loss_min = val_loss
                    es_i = 0
                else:
                    es_i += 1

                if es_i >= self.es_iters:
                    print(f"Early stopping at epoch={epoch+1}")
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
            logits = self.model(self.data, test_mask)
            logits = logits[test_mask]
            _, indices = torch.max(logits, dim=1)
            return indices

    def evaluate(self, mask):
        """Evaluate model accuracy"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data, mask)
            logits = logits[mask]
            labels = self.data.y[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)