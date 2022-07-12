from typing import List, Tuple, Union
from torch.utils.data import Dataset
import pandas as pd

import numpy as np
from numpy.typing import NDArray
import torch
from scipy.sparse import csr_matrix


class FoodRatingsDataset(Dataset):
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
        response = (
            row["user_id"],
            recipe_id,
            np.array([recipe_data["minutes"]])/2147483647,
            np.array(eval(recipe_data["nutrition"]))/434360.2,
            np.array([recipe_data["n_steps"]])/145,
            np.array([recipe_data["n_ingredients"]])/43,
        )
        
        return (*response, np.array([row["rating"]])) if self.has_rating_column else response
    
    
def one_hot_encode_sparse(
    idxs: torch.Tensor,
    arange: torch.LongTensor,
    ones: torch.FloatTensor,
    size: Tuple[int, int],
    device: str,
):
    if idxs.shape[0] != arange.shape[0]:
        arange = torch.arange(idxs.shape[0], device=device)
        ones = torch.ones(idxs.shape[0], dtype=torch.float32, device=device)

    return torch.sparse_coo_tensor(
        torch.vstack([arange, idxs]),
        ones,
        size,
        dtype=torch.float32,
        device=device,
    )


def csr_to_torch_sparse(scipy_mat: csr_matrix):
    row_idxs, col_idxs = scipy_mat.nonzero()
    indices = np.vstack([row_idxs, col_idxs])
    
    values = np.ones((len(row_idxs,)), dtype=np.float32)
    shape = scipy_mat.shape
    
    return torch.sparse_coo_tensor(indices, values, shape)


def to_sparse_batch(
    batch: List[torch.Tensor],
    tfidf_mat: csr_matrix, 
    tags_mat: csr_matrix, 
    n_users: int,
    n_items: int,
    arange: torch.LongTensor, 
    ones: torch.FloatTensor,
    device="cpu",
    return_ratings=True,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    if return_ratings:
        (
            user_ids,
            recipe_ids,
            minutes,
            nutritions,
            steps,
            ingredients,
            ratings
        ) = batch
    else:
        (
            user_ids,
            recipe_ids,
            minutes,
            nutritions,
            steps,
            ingredients,
        ) = batch
    
    batch_size = user_ids.shape[0]

    sparse_tags = csr_to_torch_sparse(tags_mat[recipe_ids])
    sparse_tfidf = csr_to_torch_sparse(tfidf_mat[recipe_ids])

    user_ids = user_ids.cuda()
    recipe_ids = recipe_ids.cuda()
    sparse_user_ids = one_hot_encode_sparse(user_ids, arange, ones, (batch_size, n_users), device).float()
    sparse_recipe_ids = one_hot_encode_sparse(recipe_ids, arange, ones, (batch_size, n_items), device)

    recipe_feature_vector = torch.cat([
        sparse_recipe_ids,
        minutes.to_sparse().to(device),
        nutritions.to_sparse().to(device),
        steps.to_sparse().to(device),
        ingredients.to_sparse().to(device),
        sparse_tags.to(device),
        sparse_tfidf.to(device),
    ], dim=1).to(device).float()
    # recipe_feature_vector = sparse_recipe_ids

    if return_ratings:
        return sparse_user_ids, recipe_feature_vector, ratings.to(device).float()
    return sparse_user_ids, recipe_feature_vector