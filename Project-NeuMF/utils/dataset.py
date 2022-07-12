import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class RatingsDatasetOneHot(Dataset):
    def __init__(self, csv_file, user_column, item_column, rating_column):
        df = pd.read_csv(csv_file)
        
        self.n_users = max(df[user_column].tolist())
        self.n_items = max(df[item_column].tolist())
        
        df[user_column] = df[user_column].apply(lambda x: x - 1)
        df[item_column] = df[item_column].apply(lambda x: x - 1)
        
        self.df = pd.DataFrame.from_dict({
            user_column: df[user_column].tolist(),
            item_column: df[item_column].tolist(),
            rating_column: df[rating_column].tolist(),
        })
        
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        user_one_hot = torch.zeros(self.n_users)
        user_one_hot[row[self.user_column]] = 1
        
        item_one_hot = torch.zeros(self.n_items)
        item_one_hot[row[self.item_column]] = 1
        
        return user_one_hot, item_one_hot, torch.tensor([float(row[self.rating_column])])


class RatingsDataset(Dataset):
    def __init__(self, csv_file, user_column, item_column, rating_column=None):
        df = pd.read_csv(csv_file)
        
        self.n_users = max(df[user_column].tolist())
        self.n_items = max(df[item_column].tolist())
        
        df[user_column] = df[user_column].apply(lambda x: x - 1)
        df[item_column] = df[item_column].apply(lambda x: x - 1)
        
        if rating_column is None:
            self.df = pd.DataFrame.from_dict({
                user_column: df[user_column].tolist(),
                item_column: df[item_column].tolist(),
            })
        else:
            self.df = pd.DataFrame.from_dict({
                user_column: df[user_column].tolist(),
                item_column: df[item_column].tolist(),
                rating_column: df[rating_column].tolist(),
            })
        
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.rating_column is None:
            return (
                torch.tensor(row[self.user_column]),
                torch.tensor(row[self.item_column]),
            )
        else:
            return (
                torch.tensor(row[self.user_column]),
                torch.tensor(row[self.item_column]),
                torch.tensor([ float(row[self.rating_column]) ])
            )
            

class RatingsDatasetGenre(Dataset):
    def __init__(
        self, 
        n_users,
        n_items,
        csv_file, 
        genre_csv, 
        user_column, 
        item_column, 
        rating_column=None, 
        genre_column=None,
        genres_map=None,
    ):
        df = pd.read_csv(csv_file)
        g_df = pd.read_csv(genre_csv)
        self.genres_map = genres_map
        
        self.n_users = n_users
        self.n_items = n_items
        
        df[user_column] = df[user_column].apply(lambda x: x - 1)
        df[item_column] = df[item_column].apply(lambda x: x - 1)
        
        if rating_column is None:
            self.df = pd.DataFrame.from_dict({
                user_column: df[user_column].tolist(),
                item_column: df[item_column].tolist(),
            })
        else:
            self.df = pd.DataFrame.from_dict({
                user_column: df[user_column].tolist(),
                item_column: df[item_column].tolist(),
                rating_column: df[rating_column].tolist(),
            })
        
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        
        # Genres
        g_df[item_column] = g_df[item_column].apply(lambda x: x - 1)
        self.genres_encoded = self.generate_genres_encoding(
            g_df[self.item_column].tolist(), g_df[genre_column].tolist()
        )
        
    def generate_genres_encoding(self, ids, genres):
        out = np.zeros((self.n_items, len(self.genres_map)), dtype=np.float32)

        for movie_id, genre_list in zip(ids, genres):
            genre_list = genre_list.split("|")
            for genre in genre_list:
                out[movie_id, self.genres_map[genre]] = 1

        return out
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        user_id = torch.tensor(row[self.user_column])
        item_id = torch.tensor(row[self.item_column])
        genres = torch.tensor(self.genres_encoded[item_id])
        
        if self.rating_column is None:
            return (
                user_id,
                item_id,
                genres,
            )
        else:
            return (
                user_id,
                item_id,
                torch.tensor([ float(row[self.rating_column]) ]),
                genres,
            )


class RatingsDatasetAllData(Dataset):
    def __init__(
        self, 
        n_users,
        n_items,
        csv_file, 
        genre_csv,
        user_csv,
        user_column, 
        item_column, 
        rating_column=None, 
        genre_column=None,
        genres_map=None,
        gender_column=None,
        gender_map=None,
        age_column=None,
        age_map=None,
        occupation_column=None,
        occupation_map=None,
    ):
        df = pd.read_csv(csv_file)
        
        self.n_users = n_users
        self.n_items = n_items
        
        df[user_column] = df[user_column].apply(lambda x: x - 1)
        df[item_column] = df[item_column].apply(lambda x: x - 1)
        
        if rating_column is None:
            self.df = pd.DataFrame.from_dict({
                user_column: df[user_column].tolist(),
                item_column: df[item_column].tolist(),
            })
        else:
            self.df = pd.DataFrame.from_dict({
                user_column: df[user_column].tolist(),
                item_column: df[item_column].tolist(),
                rating_column: df[rating_column].tolist(),
            })
        
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        
        # Genres
        g_df = pd.read_csv(genre_csv)
        self.genres_map = genres_map
        
        g_df[item_column] = g_df[item_column].apply(lambda x: x - 1)
        self.genres_encoded = self.generate_genres_encoding(
            g_df[self.item_column].tolist(), g_df[genre_column].tolist()
        )
        
        # User data
        u_df = pd.read_csv(user_csv)
        self.gender_map = gender_map
        self.age_map = age_map
        self.occupation_map = occupation_map
        
        u_df[user_column] = u_df[user_column].apply(lambda x: x - 1)
        
        (
            self.genders_encoded, 
            self.ages_encoded, 
            self.occupations_encoded
        ) = self.generate_user_encoding(
            u_df[user_column].tolist(),
            u_df[gender_column].tolist(),
            u_df[age_column].tolist(),
            u_df[occupation_column].tolist(),
        )
        
        
    def generate_genres_encoding(self, ids, genres):
        out = np.zeros((self.n_items, len(self.genres_map)), dtype=np.float32)

        for movie_id, genre_list in zip(ids, genres):
            genre_list = genre_list.split("|")
            for genre in genre_list:
                out[movie_id, self.genres_map[genre]] = 1

        return out
    
    def generate_user_encoding(self, ids, genders, ages, occupations):
        out_sex = np.zeros((self.n_users, len(self.gender_map)), dtype=np.float32)
        out_age = np.zeros((self.n_users, len(self.age_map)), dtype=np.float32)
        out_occupation = np.zeros((self.n_users, len(self.occupation_map)), dtype=np.float32)

        for user_id, gender, age, occupation in zip(ids, genders, ages, occupations):
            out_sex[user_id, self.gender_map[gender]] = 1
            out_age[user_id, self.age_map[age]] = 1
            out_occupation[user_id, self.occupation_map[occupation]] = 1

        return out_sex, out_age, out_occupation
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        user_id = torch.tensor(row[self.user_column])
        item_id = torch.tensor(row[self.item_column])
        genres = torch.tensor(self.genres_encoded[item_id])
        genders = torch.tensor(self.genders_encoded[user_id])
        ages = torch.tensor(self.ages_encoded[user_id])
        occupations = torch.tensor(self.occupations_encoded[user_id])
        
        if self.rating_column is None:
            return (
                user_id,
                item_id,
                genres,
                genders,
                ages,
                occupations,
            )
        else:
            return (
                user_id,
                item_id,
                torch.tensor([ float(row[self.rating_column]) ]),
                genres,
                genders,
                ages,
                occupations,
            )
