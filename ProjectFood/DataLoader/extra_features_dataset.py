import torch
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
from torch.utils.data import Dataset
import pandas as pd

import numpy as np


base_path = "/home/nubol23/Desktop/Codes/USP/SCC5966/kaggle/notebooks/ProjectFood/Preprocessing/objects"

preparations_tfidf = sparse.load_npz(f"{base_path}/preparations_tfidf.npz")
tags_matrix = sparse.load_npz(f"{base_path}/tags_matrix.npz")


class ExtraFoodRatingsDataset(Dataset):
    def __init__(
        self,
        interactions_file: str,
        recipes_file: str,
        n_users: int,
        n_items: int,
        has_rating_column=False,
    ):
        self.interactions = pd.read_csv(interactions_file)
        self.recipes = pd.read_csv(recipes_file)

        self.n_users = n_users
        self.n_items = n_items

        self.has_rating_column = has_rating_column

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        recipe_id = row["recipe_id"]

        recipe_data = self.recipes.iloc[recipe_id]

        # Check normalization_constants notebook
        response = (
            row["user_id"],
            recipe_id,
            np.array([
                *np.array(eval(recipe_data["nutrition"]))/434360.2,
                recipe_data["minutes"]/2147483647,
                recipe_data["n_steps"]/145,
                recipe_data["n_ingredients"]/43,
            ], dtype=np.float32),
        )

        return (*response, np.array([row["rating"]], dtype=np.float32)) if self.has_rating_column else response


def csr_to_torch_sparse(scipy_mat: csr_matrix):
    row_idxs, col_idxs = scipy_mat.nonzero()
    indices = np.vstack([row_idxs, col_idxs])

    values = np.ones((len(row_idxs, )), dtype=np.float32)
    shape = scipy_mat.shape

    return torch.sparse_coo_tensor(indices, values, shape)


def merge_text_features(recipe_ids):
    sparse_tags = csr_to_torch_sparse(tags_matrix[recipe_ids])
    sparse_tfidf = csr_to_torch_sparse(preparations_tfidf[recipe_ids])

    return torch.cat([
        sparse_tags,
        sparse_tfidf,
    ], dim=1).float()
