import pandas as pd
import numpy as np

MOVIES = 'movies.dat'
RATINGS = 'ratings.dat'

movies = []
with open(MOVIES, 'r') as f:
    for line in f.readlines():
        movies.append(line.strip().split('::'))

ratings = []
with open(RATINGS, 'r') as f:
    for line in f.readlines():
        ratings.append(line.strip().split('::'))

tags = []
with open(RATINGS, 'r') as f:
    for line in f.readlines():
        tags.append(line.strip().split('::'))

df_movies = pd.DataFrame(movies)
df_movies.columns = ['movie_id', 'title', 'genres']
df_ratings = pd.DataFrame(ratings)
df_ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df_tags = pd.DataFrame(tags)
df_tags.columns = ['user_id', 'movie_id', 'tag', 'timestamp']

unique_users = df_ratings.user_id.unique()
users_id2code = {i:code for i,code in
                 enumerate(unique_users)},
users_code2id =  {code:i for i,code in
                  enumerate(unique_users)}

unique_movies = df_ratings.movie_id.unique()
movies_id2code = {i:code for i,code in
                  enumerate(unique_movies)},
movies_code2id =  {code:i for i,code in
                  enumerate(unique_movies)}

nrow = df_ratings.shape[0]

y = df_ratings.rating
y_ind = np.arange(nrow)

row = np.concatenate((y_ind, y_ind))

users = df_ratings['user_id'].apply(lambda code: users_code2id[code]).values
movies = df_ratings['movie_id'].apply(lambda code: movies_code2id[code]).values + \
         unique_uesrs.shape[0]

col = np.concatenate((
    users, movies
))
