import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.model_selection  import train_test_split
# import matplotlib.pyplot as plt

# Reading ratings file
# Ignore the timestamp column
ratings = pd.read_csv('data/ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])

# Reading users file
users = pd.read_csv('data/users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
movies = pd.read_csv('data/movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])

# CONTENT BASED PROCESS
# Break up the big genre string into a string array
movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')

# COLLABORATIVE PROCESS
# Fill NaN values in user_id and movie_id column with 0
ratings['user_id'] = ratings['user_id'].fillna(0)
ratings['movie_id'] = ratings['movie_id'].fillna(0)
# Replace NaN values in rating column with average of all values
ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())

def contentBasedRecommend(title):
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(movies['genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(cosine_sim[8, 19])

    indices = pd.Series(movies.index, index=movies['title'])
    titles = movies['title']
    # title = "Toy Story (1995)"
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    print(titles.iloc[movie_indices])
 
def collaborativeRecommend():
    # 10% dataset
    small_data = ratings.sample(n = 2)
    print(small_data)
    print(small_data.mean(axis=1))
    # train_data, test_data = train_test_split(small_data, test_size=0.2)

    # train_data_matrix = train_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
    # test_data_matrix = test_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
    
    # # User Similarity Matrix
    # user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
    # user_correlation[np.isnan(user_correlation)] = 0

    # mean_user_rating = ratings.mean(axis=1)
    # # Use np.newaxis so that mean_user_rating has same format as ratings
    # ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    # # print(mean_user_rating)
    # # print(ratings_diff)
    # pred = mean_user_rating[:, np.newaxis] + user_correlation.dot(ratings_diff) / np.array([np.abs(user_correlation).sum(axis=1)]).T
    # print(pred.shape)

