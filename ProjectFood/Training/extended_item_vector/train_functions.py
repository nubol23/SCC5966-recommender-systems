import sys

import pandas as pd

sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from models_dot import GMF, MLP, NeuFM
from typing import List, Union
from DataLoader.dataset import to_sparse_batch
import math
from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix
from numpy.typing import NDArray
import numpy as np
import os


def define_model(
    n_users,
    n_item_features,
    k_gmf,
    k_mlp,
    layer_sizes,
    alpha=0.5
):
    return NeuFM(
        GMF(n_users, n_item_features, k_gmf),
        MLP(n_users, n_item_features, k_mlp, layer_sizes),
        alpha=alpha,
    ).cuda()


def train(
    model: nn.Module,
    lr: float,
    epochs: int,
    preparations_tfidf: csr_matrix,
    tags_matrix: csr_matrix,
    n_users: int,
    n_items: int,
    arange: torch.LongTensor,
    ones: torch.FloatTensor,
    train_dataloader: DataLoader,
    val_dataloader: Union[DataLoader, None] = None,
    show_loss=True,
    device="cpu",
):
    criterion = nn.MSELoss().cuda()
    optimizer = opt.Adam(model.parameters(), lr=lr)
    print(optimizer)
    # optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9)

    avg_losses = []

    model_state = None
    best_epoch = 0
    prev_val_loss = math.inf
    for epoch in tqdm(range(epochs)):
        n_batches = len(train_dataloader)
        avg_loss = 0
        val_loss = 0

        # Train step
        for _, batch in enumerate(train_dataloader):
            vus, vis, ratings = to_sparse_batch(
                batch,
                preparations_tfidf,
                tags_matrix,
                n_users,
                n_items,
                arange,
                ones,
                device=device,
                return_ratings=True,
            )

            optimizer.zero_grad()
            y_hat = model(vus, vis)

            loss = criterion(y_hat, ratings)
            loss.backward()
            optimizer.step()

            avg_loss += math.sqrt(float(loss.detach().cpu()))

        avg_loss /= n_batches

        # Val step
        if val_dataloader is not None:
            with torch.no_grad():
                for batch in val_dataloader:
                    val_vus, val_vis, val_rs = to_sparse_batch(
                        batch,
                        preparations_tfidf,
                        tags_matrix,
                        n_users,
                        n_items,
                        arange,
                        ones,
                        device=device,
                        return_ratings=True,
                    )

                    val_pred = model(val_vus, val_vis)
                    val_loss += math.sqrt(float(criterion(val_pred, val_rs).detach().cpu()))

            val_loss /= len(val_dataloader)
            if show_loss:
                print(f"epoch: {epoch + 1}, train_loss: {avg_loss}, val_loss: {val_loss}")

            avg_losses.append([avg_loss, val_loss])
        else:
            val_loss = avg_loss
            if show_loss:
                print(f"epoch: {epoch + 1}, train_loss: {avg_loss}")

            avg_losses.append([avg_loss])

        if val_loss < prev_val_loss:
            prev_val_loss = val_loss
            model_state = model.state_dict()
            best_epoch = epoch

    return best_epoch, prev_val_loss, model_state, avg_losses


def fit(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    n_users: int,
    n_items: int,
    n_item_features: int,
    k_gmf: int,
    k_mlp: int,
    layer_sizes: List[int],
    preparations_tfidf: csr_matrix,
    tags_matrix: csr_matrix,
    arange: torch.LongTensor,
    ones: torch.FloatTensor,
    alpha=0.5,
    lr=0.0005,
    epochs=40,
    weight_path="/home/",
    run_number=1,
    random_state=None,
    show_loss=True,
    device="cpu",
):
    if random_state is not None:
        torch.manual_seed(random_state)

    model = define_model(n_users, n_item_features, k_gmf, k_mlp, layer_sizes, alpha=alpha)

    best_epoch, val_loss, model_state, losses = train(
        model,
        lr,
        epochs,
        preparations_tfidf,
        tags_matrix,
        n_users,
        n_items,
        arange,
        ones,
        train_dataloader,
        val_dataloader,
        show_loss,
        device,
    )

    run_path = f"{weight_path}/run_{run_number}"
    if not os.path.isdir(run_path):
        os.makedirs(run_path)

    if val_dataloader is None:
        torch.save(
            model_state,
            f"{run_path}/final-{best_epoch + 1}-{val_loss}.pt"
        )
    else:
        torch.save(
            model_state,
            f"{run_path}/{best_epoch + 1}-{val_loss}.pt"
        )

    return np.asarray(losses)


def predict(
    test_dataloader: DataLoader,
    weight_path: str,
    n_users: int,
    n_items: int,
    n_item_features: int,
    k_gmf: int,
    k_mlp: int,
    layer_sizes: List[int],
    preparations_tfidf: csr_matrix,
    tags_matrix: csr_matrix,
    arange: torch.LongTensor,
    ones: torch.FloatTensor,
    alpha: float,
    test_csv_path: str,
    out_path: str,
    out_name: str,
    device="cpu",
):
    trained_model = define_model(n_users, n_item_features, k_gmf, k_mlp, layer_sizes, alpha=alpha)

    trained_model.load_state_dict(torch.load(weight_path))
    trained_model.cuda().eval()

    test_predictions = []

    for batch in test_dataloader:
        vus, vis, _ = to_sparse_batch(
            batch,
            preparations_tfidf,
            tags_matrix,
            n_users,
            n_items,
            arange,
            ones,
            device=device,
            return_ratings=True,
        )

        pred = torch.clip(trained_model(vus, vis), 1, 5).cpu().ravel().tolist()
        test_predictions += pred

    test_csv = pd.read_csv(test_csv_path)

    out_df = pd.DataFrame.from_dict(
        {
            "id": list(test_csv["id"]),
            "rating": test_predictions
        }
    )

    out_df.to_csv(f"{out_path}/{out_name}.csv", index=False)
    out_df.head()
