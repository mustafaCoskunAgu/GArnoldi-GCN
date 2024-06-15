#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP, ARNOLDI
#from torch_geometric.nn import ARNOLDI
from arnoldi import*
from torch_geometric.nn.conv.arnoldi import *


from scipy.special import comb



import math


class ChebnetII_prop(MessagePassing):
    def __init__(self, K, Init=False, bias=True, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.Init=Init
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

        if self.Init:
            for j in range(self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                self.temp.data[j] = x_j**2
        
    def forward(self, x, edge_index,edge_weight=None):
        coe_tmp=F.relu(self.temp)
        coe=coe_tmp.clone()
        
        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)


        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #L_tilde=L-I
        edge_index_tilde, norm_tilde= add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(self.node_dim))

        Tx_0=x
        Tx_1=self.propagate(edge_index_tilde,x=x,norm=norm_tilde,size=None)

        out=coe[0]/2*Tx_0+coe[1]*Tx_1

        for i in range(2,self.K+1):
            Tx_2=self.propagate(edge_index_tilde,x=Tx_1,norm=norm_tilde,size=None)
            Tx_2=2*Tx_2-Tx_0
            out=out+coe[i]*Tx_2
            Tx_0,Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
class ChebNetII(torch.nn.Module):
    def __init__(self,dataset, args):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = ChebnetII_prop(args.K)

        self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

class Bern_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index,edge_weight=None):
        TEMP=F.relu(self.temp)

        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
        #2I-L
        edge_index2, norm2=add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
            x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
            tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
            x=tmp[self.K-i-1]
            x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
            for j in range(i):
                x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

            out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x
        return out
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class BernNet(torch.nn.Module):
    def __init__(self,dataset, args):
        super(BernNet, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.m = torch.nn.BatchNorm1d(dataset.num_classes)
        self.prop1 = Bern_prop(args.K)

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

#=====================================================
#   Generalized Arnoldi
#=====================================================


class GArnoldi_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, nameFunc, homophily, Vandermonde, lower, upper, Gamma=None, bias=True, **kwargs):
        super(GArnoldi_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.homophily = homophily
        self.Vandermonde = Vandermonde
        self.nameFunc = nameFunc
        self.lower = lower
        self.upper = upper
        #self.division =  
        assert Init in ['Monomial', 'Chebyshev', 'Legendre', 'Jacobi', 'PPR','SChebyshev']
        if Init == 'Monomial':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            #x = m_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
            if(nameFunc == 'g_0'):
                self.coeffs =  compare_fit_panelA(g_0, Init, Vandermonde, self.K, self.lower, self.upper) # m_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
            elif(nameFunc == 'g_1'):
                self.coeffs =  compare_fit_panelA(g_1, Init, Vandermonde,self.K, self.lower, self.upper) 
            elif(nameFunc == 'g_2'):
                self.coeffs =  compare_fit_panelA(g_2,Init,Vandermonde, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_3'):
                self.coeffs =  compare_fit_panelA(g_3,Init,Vandermonde, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_4'):
                self.coeffs = compare_fit_panelA(g_4,Init,Vandermonde,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_rejection'):
                self.coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_band_pass'):
                self.coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_low_pass'):
                self.coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_high_pass'):
                self.coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_comb'):
               self.coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,self.K,self.lower, self.upper)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init,Vandermonde,self.K,self.lower, self.upper)
            l = [i for i in range (1, len(self.coeffs)+1) ]
            self.coeffs = filter_jackson(self.coeffs)
            TEMP = self.coeffs
            
            # TEMP = p_polynomial_zeros(self.K)
            # TEMP = j_polynomial_zeros(self.K,0,1)
        elif Init == 'Chebyshev':
            # PPR-like
            if(nameFunc == 'g_0'):
                self.coeffs = compare_fit_panelA(g_0, Init, Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_1'):
                self.coeffs = compare_fit_panelA(g_1, Init, Vandermonde,self.K,self.lower, self.upper) 
            elif(nameFunc == 'g_2'):
                self.coeffs = compare_fit_panelA(g_2,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_3'):
                self.coeffs = compare_fit_panelA(g_3,Init,Vandermonde,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_4'):
                self.coeffs = compare_fit_panelA(g_4,Init,Vandermonde,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_rejection'):
                self.coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_pass'):
                self.coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_low_pass'):
                self.coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_high_pass'):
                self.coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_comb'):
                self.coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,self.K,self.lower, self.upper)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init, Vandermonde,self.K)
            l = [i for i in range (1, len(self.coeffs)+1) ]
            #self.coeffs = np.divide(self.coeffs, l)
            self.coeffs = filter_jackson(self.coeffs)
            #self.coeffs = np.divide(self.coeffs, self.division)
            
            TEMP = self.coeffs
            #TEMP = t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
        elif Init == 'Legendre':
            #TEMP = p_polynomial_zeros(self.K)
            if(nameFunc == 'g_0'):
                self.coeffs = compare_fit_panelA(g_0, Init, Vandermonde, self.K,self.lower, self.upper)#p_polynomial_zeros(self.K) 
            elif(nameFunc == 'g_1'):
                self.coeffs = compare_fit_panelA(g_1, Init, Vandermonde,self.K,self.lower, self.upper) #p_polynomial_zeros(self.K)
            elif(nameFunc == 'g_2'):
                self.coeffs = compare_fit_panelA(g_2,Init,Vandermonde, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_3'):
                self.coeffs = compare_fit_panelA(g_3,Init,Vandermonde, self.K,self.lower, self.upper)#p_polynomial_zeros(self.K)
            elif(nameFunc == 'g_4'):
                self.coeffs = compare_fit_panelA(g_4,Init,Vandermonde,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_rejection'):
                self.coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_pass'):
                self.coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_low_pass'):
                self.coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_high_pass'):
                self.coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_comb'):
                self.coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,self.K,self.lower, self.upper)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init,self.K,self.lower, self.upper)
            l = [i for i in range (1, len(self.coeffs)+1)]
            self.coeffs = filter_jackson(self.coeffs)
            #self.coeffs = np.divide(self.coeffs, l)
            #self.coeffs = np.divide(self.coeffs, self.division)
            
            TEMP = self.coeffs
        elif Init == 'Jacobi':
            if(nameFunc == 'g_0'):
                self.coeffs = compare_fit_panelA(g_0, Init, Vandermonde, self.K,self.lower, self.upper)#p_polynomial_zeros(self.K) 
            elif(nameFunc == 'g_1'):
                self.coeffs = compare_fit_panelA(g_1, Init, Vandermonde,self.K,self.lower, self.upper) #p_polynomial_zeros(self.K)
            elif(nameFunc == 'g_2'):
                self.coeffs = compare_fit_panelA(g_2,Init,Vandermonde, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_3'):
                self.coeffs = compare_fit_panelA(g_3,Init,Vandermonde, self.K,self.lower, self.upper)#p_polynomial_zeros(self.K)
            elif(nameFunc == 'g_4'):
                self.coeffs = compare_fit_panelA(g_4,Init,Vandermonde,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_rejection'):
                self.coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_pass'):
                self.coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_low_pass'):
                self.coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_high_pass'):
                self.coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_comb'):
                self.coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,self.K,self.lower, self.upper)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init,self.K)
            l = [i for i in range (1, len(self.coeffs)+1) ]
            #self.coeffs = np.divide(self.coeffs, l)
            
            #self.coeffs = np.divide(self.coeffs, self.division)
            TEMP = self.coeffs
            #TEMP = j_polynomial_zeros(self.K,0,1)
        elif Init == 'SChebyshev':
            #TEMP = s_polynomial_zeros(self.K)
            if(nameFunc == 'g_0'):
                self.coeffs = compare_fit_panelA(g_0, Init, self.K)
            elif(nameFunc == 'g_1'):
                self.coeffs = compare_fit_panelA(g_1, Init, self.K) 
            elif(nameFunc == 'g_2'):
                self.coeffs = compare_fit_panelA(g_2,Init,self.K)
            elif(nameFunc == 'g_3'):
                self.coeffs = compare_fit_panelA(g_3,Init,self.K)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init,self.K)
            TEMP = self.coeffs
        elif Init == 'PPR':
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if(self.Init == 'Monomial'):
            self.temp.data = m_polynomial_zeros(self.lower, self.upper, self.K)#m_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
        elif (self.Init == 'Chebyshev'):
           self.temp.data = t_polynomial_zeros(self.lower, self.upper, self.K)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
        elif (self.Init == 'Legendre'):
            self.temp.data = p_polynomial_zeros(self.K)
        elif (self.Init == 'Jacobi'):
            self.temp.data = j_polynomial_zeros(self.K,0,1)
        else: 
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
        #edge_index_tilde, norm_tilde= add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(self.node_dim))
        #2I-L
        edge_index2, norm2=add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))
        hidden = self.temp[self.K-1]*x
        #hidden = x*(self.temp[0])
        for k in range(self.K-2,-1,-1):
            if (self.homophily):
                x = self.propagate(edge_index, x=x, norm=norm)             
            else:       
                x = self.propagate(edge_index1, x=x, norm=norm1)
            gamma = self.temp[k]
            
            x = x + gamma*hidden
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GARNOLDI(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GARNOLDI, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        if args.Arnoldippnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.Arnoldippnp == 'GArnoldi_prop':
            self.prop1 = GArnoldi_prop(args.K, args.alpha, args.ArnoldiInit, args.FuncName,args.homophily, args.Vandermonde, args.lower, args.upper,args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.FuncName = args.FuncName

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        
        
class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class ARNOLDI_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ARNOLDI_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = ARNOLDI(args.K, args.alpha, args.lower, args.upper, args.homophily,args.FuncName,args.ArnoldiInit,args.Vandermonde)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
    


class GCN_JKNet(torch.nn.Module):
    def __init__(self, dataset, args):
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = torch.nn.Linear(16, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=16,
                                   num_layers=4
                                   )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)
