from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SimpleFoodRatingsDataset(Dataset):
    def __init__(
            self,
            interactions_file: str,
            n_users: int,
            n_items: int,
            has_rating_column=False,
    ):
        self.interactions = pd.read_csv(interactions_file)

        self.n_users = n_users
        self.n_items = n_items

        self.has_rating_column = has_rating_column

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]

        response = (
            row["user_id"],
            row["recipe_id"],
        )

        return (*response, np.array([row["rating"]]).astype(np.float32)) if self.has_rating_column else response
