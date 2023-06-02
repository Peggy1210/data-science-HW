# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj

# from dgl.nn import GraphConv

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels,
                 base_model=GCNConv, k=2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = F.relu(self.conv[i](x, edge_index))
        return x


class GRACEModel(nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, tau=0.5):
        super(GRACEModel, self).__init__()
        self.encoder = encoder
        self.tau = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1, z2, batch_size):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1, z2, mean=True, batch_size=0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

# def drop_feature(x, drop_prob):
#     drop_mask = (
#         th.empty((x.size(1),), dtype=th.float32, device=x.device).uniform_(0, 1)
#         < drop_prob
#     )
#     x = x.clone()
#     x[:, drop_mask] = 0

#     return x

# def mask_edge(graph, mask_prob):
#     E = graph.num_edges()

#     mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
#     masks = th.bernoulli(1 - mask_rates)
#     mask_idx = masks.nonzero().squeeze(1)
#     return mask_idx

# def aug(graph, x, feat_drop_rate, edge_mask_rate):
#     n_node = graph.num_nodes()

#     edge_mask = mask_edge(graph, edge_mask_rate)
#     feat = drop_feature(x, feat_drop_rate)

#     src = graph.edges()[0]
#     dst = graph.edges()[1]

#     nsrc = src[edge_mask]
#     ndst = dst[edge_mask]

#     ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
#     ng = ng.add_self_loop()

#     return ng, feat

# # Multi-layer Graph Convolutional Networks
# class GCN(nn.Module):
#     def __init__(self, in_dim, out_dim, act_fn, num_layers=2):
#         super(GCN, self).__init__()

#         assert num_layers >= 2
#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()

#         self.convs.append(GraphConv(in_dim, out_dim * 2))
#         for _ in range(self.num_layers - 2):
#             self.convs.append(GraphConv(out_dim * 2, out_dim * 2))

#         self.convs.append(GraphConv(out_dim * 2, out_dim))
#         self.act_fn = act_fn

#     def forward(self, graph, feat):
#         for i in range(self.num_layers):
#             feat = self.act_fn(self.convs[i](graph, feat))

#         return feat


# # Multi-layer(2-layer) Perceptron
# class MLP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(in_dim, out_dim)
#         self.fc2 = nn.Linear(out_dim, in_dim)

#     def forward(self, x):
#         z = F.elu(self.fc1(x))
#         return self.fc2(z)


# class GRACEModel(nn.Module):
#     r"""
#         GRACE model
#     Parameters
#     -----------
#     in_dim: int
#         Input feature size.
#     hid_dim: int
#         Hidden feature size.
#     out_dim: int
#         Output feature size.
#     num_layers: int
#         Number of the GNN encoder layers.
#     act_fn: nn.Module
#         Activation function.
#     temp: float
#         Temperature constant.
#     """

#     def __init__(self, in_dim, hid_dim, out_dim, dropout, num_layers = 2, act_fn=nn.ReLU(), temp=1.0):
#         super(GRACEModel, self).__init__()
#         self.encoder = GCN(in_dim, hid_dim, act_fn, num_layers)
#         self.temp = temp
#         self.proj = MLP(hid_dim, out_dim)

#     def sim(self, z1, z2):
#         # normlize embeddings across feature dimension
#         z1 = F.normalize(z1)
#         z2 = F.normalize(z2)

#         s = th.mm(z1, z2.t())
#         return s

#     def get_loss(self, z1, z2):
#         # calculate SimCLR loss
#         f = lambda x: th.exp(x / self.temp)

#         refl_sim = f(self.sim(z1, z1))  # intra-view pairs
#         between_sim = f(self.sim(z1, z2))  # inter-view pairs

#         # between_sim.diag(): positive pairs
#         x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
#         loss = -th.log(between_sim.diag() / x1)

#         return loss

#     def get_embedding(self, graph, feat):
#         # get embeddings from the model for evaluation
#         h = self.encoder(graph, feat)

#         return h.detach()

#     def forward(self, graph1, graph2, feat1, feat2):
#         # encoding
#         h1 = self.encoder(graph1, feat1)
#         h2 = self.encoder(graph2, feat2)

#         # projection
#         z1 = self.proj(h1)
#         z2 = self.proj(h2)

#         # get loss
#         l1 = self.get_loss(z1, z2)
#         l2 = self.get_loss(z2, z1)

#         ret = (l1 + l2) * 0.5

#         return ret.mean()

class GRACE:
    def __init__(self, data, in_dim, hid_dim, out_dim, dropout, mask, proj_hid=256, tau=0.7, drop_feature=[0.0, 0.2], drop_edge=[0.4, 0.1], epochs=200, num_layers = 2, act_fn=nn.ReLU(), temp=1.0, lr=0.01, weight_decay=0.0005):
        self.base_model = GCNConv()
        self.encoder = Encoder(in_dim, hid_dim, base_model=self.base_model, k=num_layers)
        self.model = GRACEModel(self.encoder, hid_dim, proj_hid, tau)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.mask = mask
        self.data = data
        self.drop_feature  = drop_feature

    def fit(self):
        train_mask = self.mask['train_mask']
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            edge_index_1 = dropout_adj(self.data.edge_index, p=self.drop_edge[0])[0]
            edge_index_2 = dropout_adj(self.data.edge_index, p=self.drop_edge[1])[0]
            x_1 = drop_feature(self.data.x[train_mask], self.drop_feature[0])
            x_2 = drop_feature(self.data.x[train_mask], self.drop_feature[1])
            z1 = self.model(x_1, edge_index_1)
            z2 = self.model(x_2, edge_index_2)

            loss = self.model.loss(z1, z2, batch_size=0)
            loss.backward()
            self.optimizer.step()

            logits = loss.item()

    def evaluate(self):
        pass
