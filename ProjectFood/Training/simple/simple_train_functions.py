import sys
from typing import Union, List

# sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader

from DataLoader.simple_dataset import SimpleFoodRatingsDataset
from .simple_models import GMF, MLP, NeuMF
import math
from tqdm.notebook import tqdm
import numpy as np
import os
from copy import deepcopy


def define_model(
    n_users,
    n_items,
    k_gmf,
    k_mlp,
    layer_sizes,
    alpha=0.5
):
    return NeuMF(
        GMF(n_users, n_items, k_gmf),
        MLP(n_users, n_items, k_mlp, layer_sizes),
        alpha=alpha,
        mlp_out_size=layer_sizes[-1],
        gmf_out_size=k_gmf,
    ).cuda()


def train(
    model: nn.Module,
    lr: float,
    epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: Union[DataLoader, None] = None,
    show_loss=True,
):
    criterion = nn.MSELoss().cuda()
    optimizer = opt.Adam(model.parameters(), lr=lr)

    avg_losses = []

    model_state = None
    best_epoch = 0
    prev_val_loss = math.inf
    for epoch in tqdm(range(epochs)):
        n_batches = len(train_dataloader)
        avg_loss = 0
        val_loss = 0

        # Train step
        model.train()
        for _, (vus, vis, ratings) in enumerate(train_dataloader):
            vus = vus.cuda()
            vis = vis.cuda()
            ratings = ratings.cuda()

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
                for val_vus, val_vis, val_rs in val_dataloader:
                    val_vus = val_vus.cuda()
                    val_vis = val_vis.cuda()
                    val_rs = val_rs.cuda()

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
            model_state = deepcopy(model.state_dict())
            best_epoch = epoch

    return best_epoch, prev_val_loss, model_state, avg_losses


def fit(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    n_users: int,
    n_items: int,
    k_gmf: int,
    k_mlp: int,
    layer_sizes: List[int],
    alpha=0.5,
    lr=0.0005,
    epochs=40,
    weight_path="/home/",
    run_number=1,
    random_state=None,
    show_loss=True,
):
    if random_state is not None:
        torch.manual_seed(random_state)

    model = define_model(n_users, n_items, k_gmf, k_mlp, layer_sizes, alpha=alpha)
    # print(model)

    best_epoch, val_loss, model_state, losses = train(
        model,
        lr,
        epochs,
        train_dataloader,
        val_dataloader,
        show_loss,
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


def eval_test(
    weights_path,
    test_csv_path,
    dataset,
    n_users: int,
    n_items: int,
    k_gmf: int,
    k_mlp: int,
    layer_sizes: List[int],
    alpha=0.5,
):
    trained_model = define_model(n_users, n_items, k_gmf, k_mlp, layer_sizes, alpha=alpha)
    trained_model.load_state_dict(torch.load(weights_path))

    test_dataloader = DataLoader(
        dataset(
            test_csv_path,
            n_users,
            n_items,
            has_rating_column=True,
        ),
        batch_size=4086,
        num_workers=12,
        shuffle=False,
    )

    criterion = nn.MSELoss().cuda()

    trained_model.eval()
    loss = 0
    for vus, vis, rs in test_dataloader:
        vus = vus.cuda()
        vis = vis.cuda()
        rs = rs.cuda()

        pred = trained_model(vus, vis)
        loss += math.sqrt(float(criterion(pred, rs).detach().cpu()))

    loss /= len(test_dataloader)

    return loss


# def predict(
#     test_dataloader: DataLoader,
#     weight_path: str,
#     n_users: int,
#     n_items: int,
#     n_item_features: int,
#     k_gmf: int,
#     k_mlp: int,
#     layer_sizes: List[int],
#     preparations_tfidf: csr_matrix,
#     tags_matrix: csr_matrix,
#     arange: torch.LongTensor,
#     ones: torch.FloatTensor,
#     alpha: float,
#     test_csv_path: str,
#     out_path: str,
#     out_name: str,
#     device="cpu",
# ):
#     trained_model = define_model(n_users, n_item_features, k_gmf, k_mlp, layer_sizes, alpha=alpha)
#
#     trained_model.load_state_dict(torch.load(weight_path))
#     trained_model.cuda().eval()
#
#     test_predictions = []
#
#     for batch in test_dataloader:
#         vus, vis, _ = to_sparse_batch(
#             batch,
#             preparations_tfidf,
#             tags_matrix,
#             n_users,
#             n_items,
#             arange,
#             ones,
#             device=device,
#             return_ratings=True,
#         )
#
#         pred = torch.clip(trained_model(vus, vis), 1, 5).cpu().ravel().tolist()
#         test_predictions += pred
#
#     test_csv = pd.read_csv(test_csv_path)
#
#     out_df = pd.DataFrame.from_dict(
#         {
#             "id": list(test_csv["id"]),
#             "rating": test_predictions
#         }
#     )
#
#     out_df.to_csv(f"{out_path}/{out_name}.csv", index=False)
#     out_df.head()