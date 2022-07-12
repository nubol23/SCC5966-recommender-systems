import torch
import torch.nn as nn
import torch.sparse
import torch.nn.functional as F
from copy import deepcopy
from typing import List


class GMF(nn.Module):
    def __init__(self, n_users: int, n_item_features: int, k: int) -> None:
        super(GMF, self).__init__()
        
        self.P = nn.Linear(n_users, k, bias=False)
        self.Q = nn.Linear(n_item_features, k, bias=False)
        
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        
    def forward(self, users_ids: torch.Tensor, items_features: torch.Tensor):
        pu = self.P(users_ids)
        qi = self.Q(items_features)
        
        X = torch.sum(pu * qi, dim=1, keepdim=True)
        
        return X
    
    
class MLP(nn.Module):
    def __init__(self, n_users: int, n_item_features: int, k: int, layer_sizes: List[int]) -> None:
        super(MLP, self).__init__()
        
        self.P = nn.Linear(n_users, k, bias=False)
        self.Q = nn.Linear(n_item_features, k, bias=False)
        
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
        
        
    def forward(self, users_ids: torch.Tensor, items_features: torch.Tensor):
        pu = self.P(users_ids)
        qi = self.Q(items_features)
        
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
        
    def forward(self, users_ids: torch.Tensor, items_features: torch.Tensor):
        Xg = self.gmf(users_ids, items_features)
        Xm = self.mlp(users_ids, items_features)
        
        out = self.alpha*Xg + (1-self.alpha)*Xm
        
        return out
