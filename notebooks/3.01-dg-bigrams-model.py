# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append("..")

# %%
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit
from flax import nnx
jax.config.update("jax_numpy_rank_promotion", "warn")
jax.config.update('jax_default_device', jax.devices('cpu')[0])

from namegen.dataset import read_uk_towns_and_counties_list, Batch, get_dataset
from namegen.modeling.tokenizer import CharTokenizer
from namegen.modeling.model import Model
from namegen.modeling.train import train_bigram_model, random_bigram_model
from namegen.modeling.predict import generate, calculate_loss

# %%
# %%time
towns_list = read_uk_towns_and_counties_list('../data')

# %%
# %%time
tokenizer = CharTokenizer(towns_list)
dict_size = tokenizer.dict_size()
print(tokenizer.dict_size())
print(tokenizer.str_to_indices("hello"))
print(tokenizer.indices_to_str(tokenizer.str_to_indices("hello")))

# %%
# %%time
dataset = get_dataset(towns_list, tokenizer)
features = dataset.features
labels = dataset.labels
print(dataset.features.shape)
print(dataset.labels.shape)

# %%
# %%time
random_model = random_bigram_model(tokenizer.dict_size())
model = train_bigram_model(tokenizer.dict_size(), dataset)
model._pair_counts.sum()

# %%
print(calculate_loss(dataset, random_model).mean())
print(calculate_loss(dataset, model).mean())

# %%
key = jax.random.key(65765767)
generate(key, tokenizer, model)

# %%
key = jax.random.key(65765767)

df = pd.DataFrame()
df['random'] = generate(key, tokenizer, random_model)
df['0.1'] = generate(key, tokenizer, model, T=0.1)
df['0.8'] = generate(key, tokenizer, model, T=0.8)
df['1.0'] = generate(key, tokenizer, model, T=1.0)
df['1.2'] = generate(key, tokenizer, model, T=1.2)
df['1.5'] = generate(key, tokenizer, model, T=1.5)

df

# %%
fig = plt.figure(figsize=(16,16))
ax = fig.add_axes([0,0,1,1])
ax.imshow(model._pair_counts, cmap='Blues')
for i in range(dict_size):
    for j in range(dict_size):
        chstr = tokenizer._alphabet[i] + ' ' + tokenizer._alphabet[j]
        ax.text(j, i, chstr, ha="center", va="bottom", color='gray', size='large')
        ax.text(j, i, model._pair_counts[i, j].item(), ha="center", va="top", color='gray', size='large')
ax.axis('off');
