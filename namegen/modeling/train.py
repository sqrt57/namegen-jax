import collections
from typing import Any

import jax
import jax.numpy as jnp

from namegen.dataset import Batch
from namegen.modeling.model import BigramModel

__all__ = [
    'train_bigram_model',
    'random_bigram_model',
]

def train_bigram_model(dict_size: int, dataset: Batch, prior : int | float = 0):
    features = dataset.features
    labels = dataset.labels
    N = jnp.zeros((dict_size, dict_size), dtype=jnp.int32)
    N = N.at[features[labels>=0], labels[labels>=0]].add(1)
    model = BigramModel(N+prior)
    return model

def random_bigram_model(dict_size: int):
    return BigramModel(jnp.ones((dict_size, dict_size)))
