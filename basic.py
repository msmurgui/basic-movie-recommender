# https://blog.tensorflow.org/2020/09/introducing-tensorflow-recommenders.html

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Ratings Data
ratings = tfds.load("movie_lens/100k-ratings", split="train")
# Features of all available movies
movies = tfds.load("movie_lens/100k-movies", split="train")

# Out of all the features available in the dataset, the most
# useful are user ids and movie titles. While TFRS can use
# arbitrarily rich features, let's only use those to keep things simple.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})

movies = movies.map(lambda x: x["movie_title"])


class TwoTowerMovieLensModel(tfrs.Model):
    """MovieLens Prediction Model"""

    def __init__(self):
        super().__init__()
        # How large the representation vectors are for inputs: larger vectors make
        # for a more expressive model but may cause over-fitting.
        embedding_dim = 32
        num_unique_users = 1000
        num_unique_movies = 1700
        eval_batch_size = 128

        # a set of layers that describe how raw user features should be transformed
        # into numerical user representations
        self.user_model = tf.keras.Sequential([
            # We first turn the raw user ids into contiguous integers by looking them
            # up in a vocabulary.
            tf.keras.layers.experimental.preprocessing.StringLookup(
                max_tokens=num_unique_users
            ),
            # We then map the result into embedding vectors
            tf.keras.layers.Embedding(num_unique_users, embedding_dim)
        ])
        # Similar for the movie model
        self.movie_model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                max_tokens=num_unique_movies
            ),
            tf.keras.layers.Embedding(num_unique_movies, embedding_dim)
        ])

        # The `Task` objects has two purposes: (1) it computes the loss and (2)
        # keeps track of metrics.
        self.task = tfrs.tasks.Retrieval(
            # In this case, our metrics are top-k metrics: given a user and a known
            # watched movie, how highly would the model rank the true movie out of
            # all possible movies?
            metrics=tfrs.metrics.FactorizdeTopK(
                candidates=movies.batch(eval_batch_size).map(self.movie_model)
            )
        )

    def compute_loss(self, features, training=False):
        # The `compute_loss` method determines how loss is computed.

        # Compute user and item embeddings.
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])

        # Pass them into the task to get the resulting loss. The lower the loss is, the
        # better the model is at telling apart true watches from watches that did
        # not happen in the training data.
        return self.task(user_embeddings, movie_embeddings)

model = TwoTowerMovieLensModel()
model.compile(optimizer=tf.keras.optimizer.Adagrad(0.1))
model.fit(ratings.batch(4096), verbose=False)

#To sanity-check the model’s recommendations we can use the TFRS BruteForce 
# layer. The BruteForce layer is indexed with precomputed representations of 
# candidates, and allows us to retrieve top movies in response to a query 
# by computing the query-candidate score for all possible candidates:
#Of course, the BruteForce layer is only suitable for very small datasets.
index = tfrs.layers.ann.BruteForce(model.user_model)
index.index(movies.batch(100).map(model.movie_model), movies)
 
# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")
