from numba import njit, prange
import numpy as np


@njit
def generate_ratings_mat(uir_mat, n_users, n_items):
    out = np.zeros((n_users, n_items), dtype=np.float32)
    
    for u, i, r in uir_mat:
        out[u,i] = r
        
    return out


@njit
def predict_fbc(u, i, sims_mat, ratings_mat, k):
    sim_items_rated_by_u = [
        (sims_mat[i, j], ratings_mat[u, j])
        for j in range(ratings_mat.shape[1]) 
        if ratings_mat[u, j] != 0
    ]
    
    num = 0
    den = 0
    for sim_ij, r_uj in sorted(sim_items_rated_by_u, reverse=True)[:k]:
        num += (sim_ij*r_uj)
        den += sim_ij
        
    return num/(den+1e-15)


@njit(parallel=True)
def predict_batch_fbc(ui_mat, sims_mat, ratings_mat, k):
    predictions = np.zeros(ui_mat.shape[0])
    
    for idx in prange(ui_mat.shape[0]):
        u, i = ui_mat[idx]
        predictions[idx] = predict_fbc(u, i, sims_mat, ratings_mat, k)
        
    return np.clip(predictions, 1, 5)


@njit(parallel=True)
def rmse(pred, expected):
    return np.sqrt(np.sum((pred-expected)**2)/pred.shape[0])