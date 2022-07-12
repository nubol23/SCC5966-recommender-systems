import pandas as pd
import numpy as np
from numba import njit, prange

from tqdm.notebook import tqdm, trange

from numba.typed import List
from numba import types

from numba_progress import ProgressBar

import math


@njit
def get_items_rated_by_users(train_data, n_users):
    res = List([List.empty_list(types.int64) for _ in range(n_users)])

    for u, i, _ in train_data:
        res[u].append(i)

    for u in range(n_users):
        res[u].sort()

    return res


@njit
def step(
        train_data,
        Rus,
        n_users,
        n_items,
        k,
        α1,
        α2,
        α3,
        α4,
        α5,
        λ1,
        λ2,
        μ, bi, bu, P, Q, Y,
):
    loss = 0
    for u, i, r in train_data:
        Ru = Rus[u]
        sqrt_Ru = np.sqrt(len(Ru))

        implicit_feedback = np.zeros(k)
        for j in Ru:
            implicit_feedback += Y[j]
        implicit_feedback /= (sqrt_Ru + 1e-15)

        pred = μ + bu[u] + bi[i] + np.dot(Q[i], P[u] + implicit_feedback)
        error = r - pred

        # Updating
        bu[u] += α1 * (error - λ1 * bu[u])
        bi[i] += α2 * (error - λ1 * bi[i])

        Pu = P[u]
        Qi = Q[i]
        P[u] += α3 * (error * Qi - λ2 * Pu)
        Q[i] += α4 * (error * (Pu + implicit_feedback) - λ2 * Qi)

        term_1 = error * (Qi / (sqrt_Ru + 1e-15))
        for j in Ru:
            Y[j] += α5 * (term_1 - λ1 * Y[j])

        loss += error ** 2

    return np.sqrt(loss / len(train_data))


# RS HD page 171 (chrome), 84 book
def fit_svdpp(train_data, val_data, Rus, n_users, n_items, k, α1=.01, α2=.01, α3=.01, α4=.01, α5=.01, λ1=.01, λ2=.01,
              n_iters=20):
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
    Y = np.random.normal(0, .1, (n_items, k))

    μ = np.mean(train_data[:, 2])

    model_params = None
    best_epoch = 0
    prev_val_loss = math.inf

    t = trange(n_iters, leave=True)
    for it in t:
        loss = step(train_data, Rus, n_users, n_items, k, α1, α2, α3, α4, α5, λ1, λ2, μ, bi, bu, P, Q, Y)

        val_preds = predict_batch_svdpp(val_ui, Rus, (μ, bu, bi, P, Q, Y))
        val_loss = np.sqrt(1 / n_val * np.sum((val_preds - val_exp) ** 2))
        t.set_postfix({"Loss": loss, "Val": val_loss})

        if val_loss < prev_val_loss:
            prev_val_loss = val_loss
            model_params = (μ, bu.copy(), bi.copy(), P.copy(), Q.copy(), Y.copy())
            best_epoch = it

    return model_params


@njit
def predict(u, i, Rus, params):
    μ, bu, bi, P, Q, Y = params
    k = P.shape[1]

    Ru = Rus[u]
    sqrt_Ru = np.sqrt(len(Ru))

    implicit_feedback = np.zeros(k)
    for j in Ru:
        implicit_feedback += Y[j]
    implicit_feedback /= (sqrt_Ru + 1e-15)

    pred = μ + bu[u] + bi[i] + np.dot(Q[i], P[u] + implicit_feedback)

    return pred


@njit(parallel=True, nogil=True)
def predict_batch_inner(ui_mat, Rus, params, progress_hook):
    predictions = np.zeros(len(ui_mat))
    for it in prange(ui_mat.shape[0]):
        u, i = ui_mat[it]
        predictions[it] = predict(u, i, Rus, params)
        if np.isnan(predictions[it]):
            print(u, i)

        progress_hook.update(1)

    return np.clip(predictions, 1., 5.)


def predict_batch_progress_bar(ui_mat, Rus, params):
    with ProgressBar(total=len(ui_mat)) as progress:
        return predict_batch_inner(ui_mat, Rus, params, progress)


@njit(parallel=True, nogil=True)
def predict_batch_svdpp(ui_mat, Rus, params):
    predictions = np.zeros(len(ui_mat))
    for it in prange(ui_mat.shape[0]):
        u, i = ui_mat[it]
        predictions[it] = predict(u, i, Rus, params)

    return np.clip(predictions, 1., 5.)
