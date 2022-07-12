import math
import os
from copy import deepcopy
from typing import Union, List, Tuple

import numpy as np
from tqdm.notebook import tqdm

from .models_extra_features import NeuMF
from DataLoader.extra_features_dataset import merge_text_features

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt


def train(
        model: nn.Module,
        lr: float,
        epochs: int,
        n_users: int,
        n_items: int,
        train_dataloader: DataLoader,
        val_dataloader: Union[DataLoader, None] = None,
        show_loss=True,
        device="cpu",
):
    criterion = nn.MSELoss().cuda()
    optimizer = opt.Adam(model.parameters(), lr=lr)
    #     optimizer = opt.RMSprop(model.parameters(), lr=lr)

    avg_losses = []

    model_state = None
    best_epoch = 0
    prev_val_loss = math.inf
    for epoch in tqdm(range(epochs)):
        n_batches = len(train_dataloader)
        avg_loss = 0
        val_loss = 0

        # Train step
        for vus, vis, feats, rs in train_dataloader:
            txt_feats = merge_text_features(vis.ravel())

            optimizer.zero_grad()
            y_hat = model(vus.cuda(), vis.cuda(), feats.cuda(), txt_feats.cuda())

            loss = criterion(y_hat, rs.cuda())
            loss.backward()
            optimizer.step()

            avg_loss += math.sqrt(float(loss.detach().cpu()))

        avg_loss /= n_batches

        # Val step
        if val_dataloader is not None:
            with torch.no_grad():
                for val_vus, val_vis, val_feats, val_rs in val_dataloader:
                    val_txt_feats = merge_text_features(val_vis.ravel())
                    val_pred = model(val_vus.cuda(), val_vis.cuda(), val_feats.cuda(), val_txt_feats.cuda())
                    val_loss += math.sqrt(float(criterion(val_pred, val_rs.cuda()).detach().cpu()))

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
    n_features: int,
    n_txt: int,
    k_gmf: int,
    k_mlp: int,
    layer_sizes: List[int],
    feature_layers: List[int],
    txt_layers: List[int],
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

    model = NeuMF(n_users, n_items, k_gmf, k_mlp, layer_sizes, n_features, feature_layers, n_txt, txt_layers).cuda()
    # print(model)

    best_epoch, val_loss, model_state, losses = train(
        model, lr, epochs, n_users, n_items, train_dataloader, val_dataloader, show_loss, device
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
    n_features: int,
    n_txt: int,
    k_gmf: int,
    k_mlp: int,
    layer_sizes: List[int],
    feature_layers: List[int],
    txt_layers: List[int],
    recipes_file: str,
):
    model = NeuMF(n_users, n_items, k_gmf, k_mlp, layer_sizes, n_features, feature_layers, n_txt, txt_layers).cuda()
    model.load_state_dict(torch.load(weights_path))

    test_dataloader = DataLoader(
        dataset(
            test_csv_path,
            recipes_file,
            n_users,
            n_items,
            has_rating_column=True,
        ),
        batch_size=4086,
        num_workers=12,
        shuffle=False,
    )

    criterion = nn.MSELoss().cuda()

    model.eval()
    loss = 0
    for vus, vis, feats, rs in test_dataloader:
        txt_feats = merge_text_features(vis.ravel()).cuda()
        vus = vus.cuda()
        vis = vis.cuda()
        feats = feats.cuda()
        rs = rs.cuda()

        pred = model(vus, vis, feats, txt_feats)
        loss += math.sqrt(float(criterion(pred, rs).detach().cpu()))

    loss /= len(test_dataloader)

    return loss

