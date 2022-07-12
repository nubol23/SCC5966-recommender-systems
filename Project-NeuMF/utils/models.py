import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class GMF(nn.Module):
    def __init__(self, n_users, n_items, k) -> None:
        super(GMF, self).__init__()
        
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        self.h = nn.Linear(k, 1, bias=False)
        
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.normal_(self.h.weight, std=0.01)
        
    def forward(self, user_ids, item_ids):
        pu = self.P(user_ids)
        qi = self.Q(item_ids)
        
        X = pu * qi
        
        X = self.h(X)
        
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

        with torch.no_grad():
            gmf_h = gmf.h.weight
            mlp_h = mlp.h.weight
            new_h = nn.Linear(gmf_h.shape[1] + mlp_h.shape[1], 1, bias=False)
            new_h.weight.copy_(torch.cat([alpha*gmf_h, (1-alpha)*mlp_h], dim=1))
            
            new_gmf = deepcopy(gmf)
            new_gmf.h = nn.Identity()
            
            new_mlp = deepcopy(mlp)
            new_mlp.h = nn.Identity()
            
        self.h = new_h
        self.gmf = new_gmf
        self.mlp = new_mlp
            
        self.alpha = alpha
        
    def forward(self, user_ids, item_ids):
        Xg = self.gmf(user_ids, item_ids)
        Xm = self.mlp(user_ids, item_ids)
        
        concatenated_out = torch.concat([Xg, Xm], dim=1)
        
        out = self.h(concatenated_out)
        
        return out
