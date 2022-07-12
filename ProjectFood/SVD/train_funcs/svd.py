import pandas as pd
import numpy as np
from numba import njit, prange

from tqdm.notebook import tqdm, trange

import math


@njit
def step(
        train_data,
        n_users,
        n_items,
        k,
        α1,
        α2,
        α3,
        α4,
        λ1,
        λ2,
        μ, bi, bu, P, Q,
):
    loss = 0
    for u, i, r in train_data:
        pred = μ + bu[u] + bi[i] + np.dot(Q[i], P[u])
        error = r - pred

        # Updating
        bu[u] += α1 * (error - λ1 * bu[u])
        bi[i] += α2 * (error - λ1 * bi[i])

        Pu = P[u]
        Qi = Q[i]
        P[u] += α3 * (error * Qi - λ2 * Pu)
        Q[i] += α4 * (error * Pu - λ2 * Qi)

        loss += error ** 2

    return np.sqrt(loss / len(train_data))


# RS HD page 171 (chrome), 84 book
def fit_svd(
        train_data, val_data, n_users, n_items, k, α1=.01, α2=.01, α3=.01, α4=.01, λ1=.01, λ2=.01, n_iters=20
):
    """
    train_data: array Nx3
    """
    val_ui = val_data[:, :2]
    val_exp = val_data[:, -1]
    n_val = len(val_exp)

    bu = np.zeros(n_users, np.double)
    bi = np.zeros(n_items, np.double)

    P = np.random.normal(0, .1, (n_users, k))
    Q = np.random.normal(0, .1, (n_items, k))

    μ = np.mean(train_data[:, 2])

    model_params = None
    prev_val_loss = math.inf

    t = trange(n_iters, leave=True)
    for it in t:
        #     for it in range(n_iters):
        loss = step(train_data, n_users, n_items, k, α1, α2, α3, α4, λ1, λ2, μ, bi, bu, P, Q)

        val_preds = predict_batch_svd(val_ui, (μ, bu, bi, P, Q))
        val_loss = np.sqrt(1 / n_val * np.sum((val_preds - val_exp) ** 2))
        t.set_postfix({"Loss": loss, "Val": val_loss})

        if val_loss < prev_val_loss:
            prev_val_loss = val_loss
            model_params = (μ, bu.copy(), bi.copy(), P.copy(), Q.copy())

    #     return μ, bu, bi, P, Q
    return model_params


@njit
def predict(u, i, params):
    μ, bu, bi, P, Q = params

    pred = μ + bu[u] + bi[i] + np.dot(Q[i], P[u])

    return pred


@njit(parallel=True, nogil=True)
def predict_batch_svd(ui_mat, params):
    predictions = np.zeros(len(ui_mat))
    for it in prange(ui_mat.shape[0]):
        u, i = ui_mat[it]
        predictions[it] = predict(u, i, params)

    return np.clip(predictions, 1., 5.)
