import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from argparse import ArgumentParser

from UTILS.data_loader import load_data
from UTILS.utils import from_dgl
from GCNModels import GCN
from GATModels import GAT
from GRACEModels import GRACE
from SSPModels import SSP, SSPModel

import os
import warnings
warnings.filterwarnings("ignore")

def evaluate(model, data, mask):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        logits = model(data, mask)
        logits = logits[mask]
        labels = data.y[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def voting(result_set):
    vote_result, _ = result_set.mode(dim=0)
    return vote_result

def run(
    tags,
    # epochs, 
    # lr, 
    # hidden,
    # dropout, 
    # normalize_features,
    # weight_decay, 
    # str_optimizer, 
    # str_preconditioner, 
    # momentum,
    # early_stopping, 
    # logger, 
    # eps,
    # update_freq,
    # gamma,
    # alpha,
    # hyperparam,
    log,
    seed
):
    # Load data
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()

    # Perrocess data
    # Convert DGLGraph to Tensor
    data = from_dgl(graph)
    data.x = features
    y = torch.cat([train_labels, val_labels, test_labels, torch.zeros(18157)], dim=0).type(torch.long)
    y[train_mask], y[val_mask], y[test_mask] = train_labels, val_labels, test_labels.type(torch.long)
    mask = {'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}
    data.y = y

    if seed is not None:
        torch.manual_seed(seed)
    
    # Initialize the model
    in_size = data.x.shape[1]
    out_size = num_classes

    result_set = []
    model_cnt = 0

    if tags['ssp'] or tags['all']:
        print("Training SSP...")
        for hidden in [16, 16, 32, 32, 64, 64, 128, 128, 256, 256]:
            ssp = SSP(data, in_size, out_size, mask, log=log, hidden=hidden)
            ssp.fit()
            y_pred_ssp = ssp.predict()
            result_set.append(np.array(y_pred_ssp))
            model_cnt += 1


    if tags['gcn'] or tags['all']:
        print("Training GCN...")
        gcn = GCN(data, mask, in_size, out_size, log=log)
        gcn.fit()
        y_pred_gcn = gcn.predict()
        result_set.append(np.array(y_pred_gcn))
        model_cnt += 1

    if tags['gat'] or tags['all']:
        print("Training GAT...")
        gat = GAT(data, mask, in_size, out_size, log=log)
        gat.fit()
        y_pred_gat = gat.predict()
        result_set.append(np.array(y_pred_gat))
        model_cnt += 1

    if model_cnt > 1:
        result_set = torch.tensor(result_set)
        print("Voting...")
        voting_result = voting(result_set)
    else:
        voting_result = torch.tensor(result_set).reshape(-1)

    # Export predictions as csv file
    print("Export predictions as csv file.")
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(voting_result):
            f.write(f'{idx},{int(pred)}\n')

    

if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    # parser.add_argument('--epochs', type=int, default=300)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--hidden', type=int, default=64)
    # parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--normalize_features', type=bool, default=True)
    # parser.add_argument('--weight_decay', type=float, default=0.0005)
    # parser.add_argument('--logger', type=str, default=None)
    # parser.add_argument('--optimizer', type=str, default='Adam')
    # parser.add_argument('--preconditioner', type=str, default=None)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--eps', type=float, default=0.01)
    # parser.add_argument('--update_freq', type=int, default=50)
    # parser.add_argument('--gamma', type=float, default=None)
    # parser.add_argument('--alpha', type=float, default=None)
    # parser.add_argument('--hyperparam', type=str, default=None)
    # parser.add_argument('--early_stopping', type=int, default=0, help='num of iters to trigger early stopping')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--ssp', action='store_true', default=False)
    parser.add_argument('--gcn', action='store_true', default=False)
    parser.add_argument('--gat', action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    tags = {'all': args.all, 'ssp': args.ssp, 'gcn': args.gcn, 'gat': args.gat}
    kwargs = {
        'tags': tags,
        # 'epochs': args.epochs, 
        # 'lr': args.lr, 
        # 'hidden': args.hidden,
        # 'dropout': args.dropout, 
        # 'normalize_features': args.normalize_features,
        # 'weight_decay': args.weight_decay, 
        # 'str_optimizer': args.optimizer, 
        # 'str_preconditioner': args.preconditioner, 
        # 'momentum': args.momentum,
        # 'early_stopping': args.early_stopping, 
        # 'logger': args.logger, 
        # 'eps': args.eps,
        # 'update_freq': args.update_freq,
        # 'gamma': args.gamma,
        # 'alpha': args.alpha,
        # 'hyperparam': args.hyperparam,
        'log': args.log,
        'seed': args.seed
    }

    run(**kwargs)

    # print("Testing...")
    # model.eval()
    # with torch.no_grad():
    #     logits = model(data, test_mask)
    #     logits = logits[test_mask]
    #     _, indices = torch.max(logits, dim=1)

    # # Export predictions as csv file
    # print("Export predictions as csv file.")
    # with open('output.csv', 'w') as f:
    #     f.write('Id,Predict\n')
    #     for idx, pred in enumerate(indices):
    #         f.write(f'{idx},{int(pred)}\n')
    # # Please remember to upload your output.csv file to Kaggle for scoring