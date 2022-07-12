import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class GMF(nn.Module):
    def __init__(self, n_users, n_items, n_genres, k) -> None:
        super(GMF, self).__init__()
        
        self.P = nn.Linear(n_users, k, bias=False)
        self.Q = nn.Linear(n_items+n_genres, k, bias=False)
        
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        
        self.n_users = n_users
        self.n_items = n_items
        
    def forward(self, user_ids, item_ids, genres_one_hot):
        batch_size = user_ids.shape[0]
        device = user_ids.device
        
        arange = torch.arange(batch_size, device=device)
        ones = torch.ones(batch_size, dtype=torch.float32, device=device)
        
        sparse_user_ids = torch.sparse_coo_tensor(
            torch.vstack([arange, user_ids]),
            ones,
            size=(user_ids.shape[0], self.n_users),
            dtype=torch.float32,
            device=device,
        )
        sparse_item_ids = torch.sparse_coo_tensor(
            torch.vstack([arange, item_ids]),
            ones,
            size=(item_ids.shape[0], self.n_items),
            dtype=torch.float32,
            device=device,
        )
        genre_item_ids = torch.cat([sparse_item_ids, genres_one_hot.to_sparse()], dim=1)
        
        pu = self.P(sparse_user_ids)
        qi = self.Q(genre_item_ids)
        
        X = torch.sum(pu * qi, dim=1, keepdim=True)
        
        return X
    
    
class MLP(nn.Module):
    def __init__(self, n_users, n_items, n_genres, k, layer_sizes) -> None:
        super(MLP, self).__init__()
        
        self.P = nn.Linear(n_users, k, bias=False)
        self.Q = nn.Linear(n_items+n_genres, k, bias=False)
        
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
            
        self.n_users = n_users
        self.n_items = n_items
        
        
    def forward(self, user_ids, item_ids, genres_one_hot):
        batch_size = user_ids.shape[0]
        device = user_ids.device
        
        arange = torch.arange(batch_size, device=device)
        ones = torch.ones(batch_size, dtype=torch.float32, device=device)
        
        sparse_user_ids = torch.sparse_coo_tensor(
            torch.vstack([arange, user_ids]),
            ones,
            size=(user_ids.shape[0], self.n_users),
            dtype=torch.float32,
            device=device,
        )
        sparse_item_ids = torch.sparse_coo_tensor(
            torch.vstack([arange, item_ids]),
            ones,
            size=(item_ids.shape[0], self.n_items),
            dtype=torch.float32,
            device=device,
        )
        genre_item_ids = torch.cat([sparse_item_ids, genres_one_hot.to_sparse()], dim=1)
        
        pu = self.P(sparse_user_ids)
        qi = self.Q(genre_item_ids)
        
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
        
    def forward(self, user_ids, item_ids, genres_one_hot):
        Xg = self.gmf(user_ids, item_ids, genres_one_hot)
        Xm = self.mlp(user_ids, item_ids, genres_one_hot)
        
        out = self.alpha*Xg + (1-self.alpha)*Xm
        
        return out
