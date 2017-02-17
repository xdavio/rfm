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

