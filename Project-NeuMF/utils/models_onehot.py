import torch
import torch.nn as nn
import torch.nn.functional as F


class GMF(nn.Module):
    def __init__(self, n_users, n_items, k) -> None:
        super(GMF, self).__init__()
        
        self.P = nn.Linear(n_users, k, bias=False)
        self.Q = nn.Linear(n_items, k, bias=False)
        self.h = nn.Linear(k, 1, bias=False)
        
        nn.init.normal_(self.P.weight, mean=0, std=0.1)
        nn.init.normal_(self.Q.weight, mean=0, std=0.1)
        nn.init.normal_(self.h.weight, mean=0, std=0.1)
        
    def forward(self, Xu, Xi, alpha=1.):
        pu = self.P(Xu)
        qi = self.Q(Xi)
        
        X = pu * qi
        
        X = self.h(X)
        
        return X
    

class MLP(nn.Module):
    def __init__(self, n_users, n_items, k, layer_sizes) -> None:
        super(MLP, self).__init__()
        
        self.P = nn.Linear(n_users, k, bias=False)
        self.Q = nn.Linear(n_items, k, bias=False)
        
        self.layers = nn.ModuleList()        
        prev_size = 2*k
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            prev_size = size
            
        self.h = nn.Linear(prev_size, 1, bias=False)
        
    def forward(self, Xu, Xi, alpha=1.):
        pu = self.P(Xu)
        qi = self.Q(Xi)
        
        X = torch.concat([pu, qi], dim=1)
        
        for layer in self.layers:
            X = F.relu(layer(X))
            
        X = alpha*self.h(X)
        
        return X
    

class NeuFM(nn.Module):
    def __init__(self, gmf, mlp, alpha) -> None:
        super(NeuFM, self).__init__()
        
        self.gmf = gmf
        self.mlp = mlp
        self.alpha = alpha
        
    def forward(self, Xu, Xi):
        Xg = self.gmf(Xu, Xi, self.alpha)
        Xm = self.mlp(Xu, Xi, 1-self.alpha)
        
        return Xg + Xm
