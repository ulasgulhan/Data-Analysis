
import pandas as pd
import numpy as np
from math import sqrt
from pprint import pprint

movie_df = pd.read_csv('Data/movies.csv')
rating_df = pd.read_csv('Data/ratings.csv')


movie_df['year'] = movie_df['title'].str.extract(r'\((\d{4})\)', expand=False)
movie_df['title'] = movie_df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

movie_df.drop(
    labels=['genres', 'year'],
    axis=1,
    inplace=True
)

rating_df.drop('timestamp', axis=1, inplace=True)

user_input_df = pd.DataFrame([
    {'title': 'Toy Story', 'rating': 5},
    {'title': 'Jumanji', 'rating': 5},
    {'title': 'Space Jam', 'rating': 5},
    {'title': 'Robots', 'rating': 4},
    {'title': 'Back to the Future', 'rating': 4},
])

input_movies = movie_df.merge(
    right=user_input_df,
    how='right',
    left_on='title',
    right_on='title'
)

# Identify users who have rated the same movies as the new user

user_subset_df = pd.merge(
    left=rating_df,
    right=input_movies,
    left_on='movieId',
    right_on='movieId',
    how='inner'
)

user_subset_df.drop(
    labels=['title', 'rating_y'],
    axis=1,
    inplace=True
)

user_subset_df.rename(
    columns={
        'rating_x': 'rating'
    },
    inplace=True
)

# Group the data by userId

user_subset_groups = user_subset_df.groupby(['userId'])

# Sort the groups based on the number of shared movies with the new user

sorted_user_usesubset_group = sorted(
    user_subset_groups,
    key=lambda x: len(x[1]),
    reverse=True
)

# We will determine the correlation between the newly arrived user and the users who have already rated the same movies as this user. Here, we will examine the relationship between users using correlation, which is extensively used in statistical science, and there are different types of correlations. We will use Pearson Correlation here.

# Calculate Pearson correlation between the new user and existing users who have rated similar movies

pearson_corellation_dict = {}

for userId, group in sorted_user_usesubset_group:

    group.sort_values(by='movieId', inplace=True)
    input_movies.sort_values(by='movieId', inplace=True)

    n_rating = len(group)


    temp_df = input_movies[input_movies['movieId'].isin(group['movieId'].tolist())]


    temp_rating_list = temp_df['rating'].tolist()
    temp_group_list = group['rating'].tolist()

    Sxx = sum([i ** 2 for i in temp_rating_list]) - pow(sum(temp_rating_list), 2) / float(n_rating)
    Syy = sum([i ** 2 for i in temp_group_list]) - pow(sum(temp_group_list), 2) / float(n_rating)

    Sxy = sum(i * j for i, j in zip(temp_rating_list, temp_group_list)) - sum(temp_rating_list) * sum(temp_group_list) / float(n_rating)

    if Sxx != 0 and Syy != 0:
        pearson_corellation_dict[userId] = Sxy / sqrt(Sxx * Syy)
    else:
        pearson_corellation_dict[userId] = 0


pearson_df = pd.DataFrame.from_dict(pearson_corellation_dict, orient='index')
pearson_df.columns = ['similary index']
pearson_df['userId'] = pearson_df.index
pearson_df.index = range(len(pearson_df))
pearson_df['userId'] = pearson_df.userId.apply(lambda x: x[0] if isinstance(x, tuple) and len(x) == 1 else None)
sorted_pearson_df = pearson_df.sort_values(by='similary index', ascending=False)


top_user_rating = sorted_pearson_df.merge(
    rating_df,
    on='userId',
    how='inner'
)

top_user_rating['weighted_rating'] = top_user_rating['similary index'] * top_user_rating['rating']
temp_user_rating = top_user_rating.groupby('movieId').sum()[['similary index', 'weighted_rating']]

temp_user_rating['recomandation score'] = temp_user_rating['weighted_rating'] / temp_user_rating['similary index']

temp_user_rating = temp_user_rating.sort_values(by='recomandation score', ascending=False)

recomandation_df = pd.merge(
    left=temp_user_rating,
    right=movie_df,
    right_on='movieId',
    left_on='movieId',
    how='inner'
)

print(recomandation_df.head(20).to_string())

