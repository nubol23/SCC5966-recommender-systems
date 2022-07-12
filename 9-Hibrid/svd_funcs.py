from numba import njit, prange
import numpy as np


@njit
def fit_funk_svd(train_data, n_users, n_items, k, α=.01, λ=.01, n_iters=20):
    """
    train_data: array Nx3
    """
    bu = np.zeros(n_users, np.double)
    bi = np.zeros(n_items, np.double)
    
    P = np.random.normal(0, .1, (n_users, k))
    Q = np.random.normal(0, .1, (n_items, k))
    
    μ = np.mean(train_data[:, 2])
    
    for it in range(n_iters):
        for u, i, r in train_data:
            pred = μ + bu[u] + bi[i] + np.dot(P[u], Q[i])
            
            error = r - pred
            
            # Updating
            bu[u] += α * (error - λ*bu[u])
            bi[i] += α * (error - λ*bi[i])
            P[u], Q[i] = P[u] + α*(error*Q[i] - λ*P[u]), Q[i] + α*(error*P[u] - λ*Q[i])
    
    return μ, bu, bi, P, Q


@njit
def predict_svd(u, i, params):
    μ, bu, bi, P, Q = params
    
    return μ + bu[u] + bi[i] + np.dot(P[u], Q[i])


@njit
def predict_batch_svd(ui_mat, params):
    predictions = np.zeros(len(ui_mat))
    for it in prange(ui_mat.shape[0]):
        u, i = ui_mat[it]
        predictions[it] = predict_svd(u, i, params)
        
    return np.clip(predictions, 1., 5.)
