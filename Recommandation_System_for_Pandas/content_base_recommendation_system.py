# Content Base Recommendation System

import pandas as pd
import numpy as np


movie_df = pd.read_csv('Data/movies.csv')


# Add the year information from the 'title' column to a new column named 'year'

movie_df['year'] = movie_df['title'].str.extract(r'\((\d{4})\)', expand=False)

# Remove the year information from the 'title' column
movie_df['title'] = movie_df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()


# Let's assign 1 if each movie in the dataset has each genre, and 0 otherwise. We won't use one hot encoding. We'll hard code it.


genre_list = []

for index, column in movie_df.iterrows():
    for genre in column['genres'].split('|'):
        if genre not in genre_list and genre != '(no genres listed)':
            genre_list.append(genre)


movies_genre_df = movie_df.copy()
for genre in genre_list:
    movies_genre_df[genre] = np.NAN


for index, column in movies_genre_df.iterrows():
    for genre in column['genres'].split('|'):
        if genre in genre_list:
            movies_genre_df.loc[index, genre] = 1


movies_genre_df.fillna(0, inplace=True)


# New member's data

user_input_df = pd.DataFrame([
    {'title': 'Toy Story', 'rating': 5},
    {'title': 'Jumanji', 'rating': 5},
    {'title': 'Space Jam', 'rating': 5},
    {'title': 'Heat', 'rating': 4},
    {'title': 'Back to the Future', 'rating': 4},
])


# The user's data came to us only with the title and rating information. Let's determine which movies in the incoming data correspond to the movies in our dataset.

merged_input = movie_df.merge(user_input_df, how='inner', on='title')

# Multiple versions of the movie 'Heat' from different years are included in the main movie dataset. Let's remove these movies based on their years.

merged_input.drop(index=[3, 4], axis=0, inplace=True)

# Let's get rid of the Genres and Year columns

merged_input.drop(labels=['genres', 'year'], axis=1, inplace=True)

# As a result of the above operations, the index structure of the relevant dataset was disrupted. Let's reset them.

input_movies_df = merged_input.reset_index(drop=True)

# Let's determine the genres of the movies rated by the new user.

user_movies_df = movies_genre_df.merge(input_movies_df, how='inner', on='movieId')

# As a result of the merge operation, many columns were created. Let's get rid of unnecessary columns.

user_genre_df = user_movies_df.drop(labels=['movieId', 'title_x', 'genres', 'year', 'title_y', 'rating'], axis=1)

# As a result of the above operations, we determined the genres of the movies rated by the new user. In this step, we will obtain the 'user_profile' by multiplying these genres with the rating points given by the new user.

user_profile = user_genre_df.transpose().dot(input_movies_df['rating'])

# Let's create another matrix needed for implementing the Content-Based system, called the movie matrix. We already have the movies_genre_df dataset containing the genres of the movies. We will manipulate this dataset to convert it into the desired dataset.

# Step 1: Set the movieId column as the index of the relevant dataset.

movie_matrix = movies_genre_df.set_index(movies_genre_df['movieId'])

# Step 2: Let's remove the unnecessary columns.

movie_matrix.drop(
    labels=['movieId', 'title', 'genres', 'year'],
    axis=1,
    inplace=True)

# We have prepared the user profile and movie matrix datasets. Now we can find the weighted movie matrix.

weighted_movie_matrix = (user_profile * movie_matrix).sum(axis=1)

# Let's create the recommendation matrix.

recomandation_movie_matrix = weighted_movie_matrix / user_profile.sum()
recomandation_movie_matrix.sort_values(ascending=False, inplace=True)

# The recomandation_movie_matrix contains only the movieId and recommendation weight. Let's convert this matrix to a DataFrame so that we can easily manipulate and get the required fields. For example; title, genres, etc.

recomandation_df = pd.DataFrame(recomandation_movie_matrix)

# We have converted it to a DataFrame, but it needs attention. For example, the column name is '0', etc.

recomandation_df.columns = ['Weight of Recomandation']

# Let's merge with the movie_df to bring additional information in the recommendation dataset.

result = movie_df.merge(
    recomandation_df,
    how='right',
    left_on='movieId',
    right_on='movieId'
)

print(result.head(10).to_string())
