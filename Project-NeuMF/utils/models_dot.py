import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class GMF(nn.Module):
    def __init__(self, n_users, n_items, k) -> None:
        super(GMF, self).__init__()
        
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        
    def forward(self, user_ids, item_ids):
        pu = self.P(user_ids)
        qi = self.Q(item_ids)
        
        X = torch.sum(pu * qi, dim=1, keepdim=True)
        
        return X
    
    
class MLP(nn.Module):
    def __init__(self, n_users, n_items, k, layer_sizes) -> None:
        super(MLP, self).__init__()
        
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        
        self.layers = nn.ModuleList()        
        prev_size = 2*k
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            prev_size = size
            
        self.h = nn.Linear(prev_size, 1, bias=False)
        
        # Init
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.normal_(self.h.weight, std=0.01)
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.01)
        
        
    def forward(self, user_ids, item_ids):
        pu = self.P(user_ids)
        qi = self.Q(item_ids)
        
        X = torch.concat([pu, qi], dim=1)
        
        for layer in self.layers:
            X = F.relu(layer(X))
            
        X = self.h(X)
        
        return X


class NeuFM(nn.Module):
    def __init__(self, gmf: GMF, mlp: MLP, alpha: float) -> None:
        super(NeuFM, self).__init__()
            
        self.gmf = deepcopy(gmf)
        self.mlp = deepcopy(mlp)
        self.alpha = alpha
        
    def forward(self, user_ids, item_ids):
        Xg = self.gmf(user_ids, item_ids)
        Xm = self.mlp(user_ids, item_ids)
        
        out = self.alpha*Xg + (1-self.alpha)*Xm
        
        return out