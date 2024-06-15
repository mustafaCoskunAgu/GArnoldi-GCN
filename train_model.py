#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

"""

"""

import argparse
from dataset_utils import DataLoader
from utils import random_planetoid_splits
from GNN_models import *

import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import numpy as np


def RunExp(args, dataset, data, Net, percls_trn, val_lb):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN','GARNOLDI','ARNOLDI']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            if args.net == 'GARNOLDI':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            if args.net == 'ARNOLDI':
                Alpha = 1- args.alpha
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    return test_acc, best_val_acc, Gamma_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lower', type=float, default=0.000001)
    parser.add_argument('--upper', type=float, default=2.0)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Monimal','Null'],
                        default='PPR')
    parser.add_argument('--ArnoldiInit', type=str,
                        choices=['Monimal', 'Chebyshev', 'Legendre', 'Jacobi'],
                        default='Legendre')
    parser.add_argument('--FuncName', type=str,
                        choices=['g_0', 'g_1', 'g_2', 'g_3','g_band_rejection'],
                        default='g_1')
    parser.add_argument('--homophily', type=bool, default=False, help='Cora, Citeseer, Pubmed are homophily')
    parser.add_argument('--Vandermonde', type=bool, default=False, help='Should we obtain coeffs with Vandermonde or Arnoldi?')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--Arnoldippnp', default='GArnoldi_prop',
                        choices=['PPNP', 'GArnoldi_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='texas')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=5)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'ARNOLDI', 'GARNOLDI'],
                        default='ChebNetII')

    args = parser.parse_args()
    
    
    functionnames = ['g_band_rejection', 'g_band_pass', 'g_low_pass', 'g_high_pass']
    #functionnames = ['g_0', 'g_1', 'g_2', 'g_3']
    polynames = ['Monomial', 'Chebyshev','Legendre', 'Jacobi']
    #polynames = ['Chebyshev']
    #functionnames = ['g_low_pass']
    
    methodnames = ['GARNOLDI']
    #methodnames = ['GCN', 'GAT', 'APPNP','ChebNet', 'JKNet','GPRGNN','BernNet']
    LR = [  0.002]
    MYdropout = [0.5]
    #sys.stdout = open('PubmedHyperOPTComplexes-L.txt', 'w')
    print("HYPER PARAMETER TUNING")
    for l in range (len(LR)):
        print("---------------------------------------------- LR = ", LR[l] )
        
        for d in range (len(MYdropout)):
            print("===================================== DROPOUT = ", MYdropout[d])
            for i in range(len(functionnames)):
                for j in range(len(polynames)):
                    for t in range(len(methodnames)):   
                        args.net = methodnames[t]
                        args.FuncName = functionnames[i]
                        args.ArnoldiInit = polynames[j]
                        gnn_name = args.net
                        funcName = args.FuncName
                        PolyName = args.ArnoldiInit
                        args.lr = LR[l]
                        args.dropout = MYdropout[d]
                        if gnn_name == 'GCN':
                            Net = GCN_Net
                        elif gnn_name == 'GAT':
                            Net = GAT_Net
                        elif gnn_name == 'APPNP':
                            Net = APPNP_Net
                        elif gnn_name == 'ChebNet':
                            Net = ChebNet
                        elif gnn_name == 'JKNet':
                            Net = GCN_JKNet
                        elif gnn_name == 'GPRGNN':
                            Net = GPRGNN
                        elif gnn_name == 'ARNOLDI':
                            Net = ARNOLDI_Net
                        elif gnn_name == 'GARNOLDI':
                            Net = GARNOLDI
                        elif gnn_name == 'ChebNetII':
                            Net = ChebNetII
                        elif gnn_name == 'BernNet':
                            Net = BernNet
                        
                    
                        dname = args.dataset
                        dataset, data = DataLoader(dname)
                    
                        RPMAX = args.RPMAX
                        Init = args.Init
                    
                        Gamma_0 = None
                        alpha = args.alpha
                        train_rate = args.train_rate
                        val_rate = args.val_rate
                        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
                        val_lb = int(round(val_rate*len(data.y)))
                        TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
                        print('True Label rate: ', TrueLBrate)
                    
                        args.C = len(data.y.unique())
                        args.Gamma = Gamma_0
                    
                        Results0 = []
                    
                        for RP in tqdm(range(RPMAX)):
                    
                            test_acc, best_val_acc, Gamma_0 = RunExp(
                                args, dataset, data, Net, percls_trn, val_lb)
                            Results0.append([test_acc, best_val_acc, Gamma_0])
                            #print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')
                    
                        test_acc_mean, val_acc_mean, _ = np.mean(np.array(Results0, dtype=object), axis=0) * 100
                        test_acc_std = np.sqrt(np.var(np.array(Results0, dtype=object), axis=0)[0]) * 100
                        
                        
                        if args.net in ['GARNOLDI','ARNOLDI']:
                            print(f'{gnn_name}-{PolyName} ({funcName}) with Vandermonde? {args.Vandermonde} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
                            print(
                                f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
                        else:
                            print(f'{gnn_name}  on dataset {args.dataset}, in {RPMAX} repeated experiment:')
                            print(
                                f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
    
    
    #sys.stdout.close()         
