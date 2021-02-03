# https://www.tensorflow.org/recommenders/examples/basic_retrieval
# We are focusing on a retrieval system: a model that predicts a
# set of movies from the catalogue that the user is likely to watch.
# Often, implicit data is more useful here, and so we are going
# to treat Movielens as an implicit system. This means that
# every movie a user watched is a positive example, and every movie
# they have not seen is an implicit negative example.

import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Ratings data
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all available movies.
movies = tfds.load("movielens/100k-movies", split="train")

for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)

for x in movies.take(1).as_numpy_iterator():
    pprint.pprint(x)

# In this example, we're going to focus on the ratings data.
# Other tutorials explore how to use the movie information
# data as well to improve the model quality.

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})

movies = movies.map(lambda x: x["movie_title"])

# To fit and evaluate the model, we need to split it into a
# training and evaluation set. In an industrial recommender system,
# this would most likely be done by time: the data up to time  would
# be used to predict interactions after .
# In this simple example, however, let's use a random split, putting
# 80% of the ratings in the train set, and 20% in the test set.

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# Let's also figure out unique user ids and movie titles present in the data.
# This is important because we need to be able to map the raw values of our
# categorical features to embedding vectors in our models. To do that, we
# need a vocabulary that maps a raw feature value to an integer in a contiguous
# range: this allows us to look up the corresponding embeddings in our
# embedding tables.

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

print(unique_movie_titles[:10])

# The Model - Two Tower Architecture
# Query tower
embedding_dimension = 32
# Higher values will correspond to models that may be more accurate,
# but will also be slower to fit and more prone to overfitting.

user_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None
    ),
    # We add an additional embedding to account for unknown tokens.
    tf.keras.layers.Embedding(len(unique_user_ids)+1, embedding_dimension)
])

# A simple model like this corresponds exactly to a classic matrix
# factorization approach. While defining a subclass of tf.keras.Model
# for this simple model might be overkill, we can easily extend
# it to an arbitrarily complex model using standard Keras components,
# as long as we return an embedding_dimension-wide output at the end.

# The Candidate Tower
movie_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None
    ),
    tf.keras.layers.Embedding(len(unique_movie_titles)+1, embedding_dimension)
])

# Metrics
# In our training data we have positive (user, movie) pairs. To figure
# out how good our model is, we need to compare the affinity score that
# the model calculates for this pair to the scores of all the other
# possible candidates: if the score for the positive pair is higher
# than for all other candidates, our model is highly accurate.

# In this case, our metrics are top-k metrics: given a user and a known
# watched movie, how highly would the model rank the true movie out of
# all possible movies?

metrics = tfrs.metrics.FactorizedTopK(
    candidates=movies.batch(128).map(movie_model)
)

# Loss
# In this instance, we'll make use of the Retrieval task object: a convenience
# wrapper that bundles together the loss function and metric computation:

task = tfrs.tasks.Retrieval(
    metrics=metrics
)


class MovieLensModel(tfrs.Model):
    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_titles
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=false) -> tf.Tensor:
        # We pick  out the user features and pass them  into the user model
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into de movie model
        # getting embeddings back
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics
        return self.task(user_embeddings, positive_movie_embeddings)

# Same model but using keras.Model
class NoBaseClassMovielensModel(tf.keras.Model):

    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:

            # Loss computation.
            user_embeddings = self.user_model(features["user_id"])
            positive_movie_embeddings = self.movie_model(
                features["movie_title"])
            loss = self.task(user_embeddings, positive_movie_embeddings)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

        # Loss computation.
        user_embeddings = self.user_model(features["user_id"])
        positive_movie_embeddings = self.movie_model(features["movie_title"])
        loss = self.task(user_embeddings, positive_movie_embeddings)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

#Fitting and evaluating
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

#Then shuffle, batch, and cache the training and evaluation data.
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

#Train the model
model.fit(cached_train, epochs=3)

