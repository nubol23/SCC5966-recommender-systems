import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuMF(torch.nn.Module):
    def __init__(self, n_users, n_items, k_gmf, k_mlp, layers, n_features, features_layers, n_txt, txt_layers):
        super(NeuMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.k_gmf = k_gmf
        self.k_mlp = k_mlp

        # GMF
        self.P_gmf = nn.Embedding(self.n_users, self.k_gmf)
        self.Q_gmf = nn.Embedding(self.n_items, self.k_gmf)

        # MLP
        self.P_mlp = nn.Embedding(self.n_users, self.k_mlp)
        self.Q_mlp = nn.Embedding(self.n_items, self.k_mlp)

        self.fc_layers = nn.ModuleList()
        n_in = self.k_mlp*2
        for n_out in layers:
            self.fc_layers.append(nn.Linear(n_in, n_out))
            n_in = n_out

        # Features
        self.feature_layers = nn.ModuleList()
        n_in = n_features
        for n_out in features_layers:
            self.feature_layers.append(nn.Linear(n_in, n_out))
            n_in = n_out

        # Text Features
        self.txt_layers = nn.ModuleList()
        n_in = n_txt
        for n_out in txt_layers:
            self.txt_layers.append(nn.Linear(n_in, n_out))
            n_in = n_out

        # Merge layer
        self.h = nn.Linear(
            in_features=layers[-1] + k_gmf + features_layers[-1] + txt_layers[-1],
            out_features=64
        )
        self.hn1 = nn.Linear(64, 32)
        self.hn2 = nn.Linear(32, 16)
        self.hn3 = nn.Linear(16, 8)
        self.hn4 = nn.Linear(8, 1)

        nn.init.normal_(self.P_gmf.weight, std=0.01)
        nn.init.normal_(self.Q_gmf.weight, std=0.01)

        nn.init.normal_(self.P_mlp.weight, std=0.01)
        nn.init.normal_(self.Q_mlp.weight, std=0.01)

        nn.init.normal_(self.h.weight, std=0.01)

        for layer in self.fc_layers:
            nn.init.normal_(layer.weight, std=0.01)

        for layer in self.feature_layers:
            nn.init.normal_(layer.weight, std=0.01)

        for layer in self.txt_layers:
            nn.init.normal_(layer.weight, std=0.01)

        nn.init.normal_(self.hn1.weight, std=0.01)
        nn.init.normal_(self.hn2.weight, std=0.01)
        nn.init.normal_(self.hn3.weight, std=0.01)
        nn.init.normal_(self.hn4.weight, std=0.01)

    def forward(self, user_indices, item_indices, features, txt):
        # GMF
        pu_gmf = self.P_gmf(user_indices)
        qi_gmf = self.Q_gmf(item_indices)
        gmf_vector = torch.mul(pu_gmf, qi_gmf)

        # MLP
        pu_mlp = self.P_mlp(user_indices)
        qi_mlp = self.Q_mlp(item_indices)
        mlp_vector = torch.cat([pu_mlp, qi_mlp], dim=1)  # the concat latent vector
        for layer in self.fc_layers:
            mlp_vector = F.relu(layer(mlp_vector))

        # Features
        feat_vector = features
        for layer in self.feature_layers:
            feat_vector = F.relu(layer(feat_vector))

        # Text features
        txt_vector = txt
        for layer in self.txt_layers:
            txt_vector = F.relu(layer(txt_vector))

        # Merge layer
        x = torch.cat([gmf_vector, mlp_vector, feat_vector, txt_vector], dim=1)
        x = self.h(x)

        x = self.hn1(F.relu(x))
        x = self.hn2(F.relu(x))
        x = self.hn3(F.relu(x))
        x = self.hn4(F.relu(x))

        return x
