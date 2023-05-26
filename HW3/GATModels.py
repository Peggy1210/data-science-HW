import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout=0.6):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads[0], dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads[0], hidden_channels * heads[0], heads[1], dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads[0] * heads[1], out_channels, heads[2],
                             concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = x.flatten(1)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = x.flatten(1)
        x = self.conv3(x, edge_index)
        return x
    
class GAT:
    def __init__(self,
                data,
                mask,
                in_size, 
                out_size,
                epochs=200,
                lr=0.01,
                hid_size=16, 
                dropout=0.6,
                weight_decay=0.0005, 
                normalize_features=True,
                early_stopping=0,
                log=False,
                seed=None):
        self.model = GATModel(in_size, hid_size, out_size, [4, 4, 1], dropout)
        self.data = data
        self.mask = mask
        self.epochs = epochs

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