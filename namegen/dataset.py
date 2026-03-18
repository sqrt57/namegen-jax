from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

from namegen.modeling.tokenizer import CharTokenizer

__names__ = [
    'read_uk_towns_and_counties',
    'Batch',
    'get_dataset',
]

def read_uk_towns_and_counties_list(data_path: str | Path) -> list[str]:
    df = pd.read_csv(Path(data_path) / 'raw' / 'uk-towns.csv', skipinitialspace=True)
    lst = pd.concat([df['Town'], df['County']]).str.lower().drop_duplicates().to_list()
    for sep in ['/', '(', ')']:
        result = []
        for s in lst:
            result.extend(s.split(sep))
        lst = result
    return sorted(list(set([s.strip() for s in result if s.strip() != ''])))

def random_split(key: jax.Array, strings: list[str], fraction: float) -> tuple[list[str], list[str]]:
    nfraction = int(fraction*len(strings))
    permutation = jax.random.permutation(key, len(strings))
    strings_permutation = [strings[i] for i in permutation.tolist()]
    strings_part1 = strings_permutation[:nfraction]
    strings_part2 = strings_permutation[nfraction:]
    return (strings_part1, strings_part2)

Batch = namedtuple('Batch', 'features labels')

def get_dataset(strings: list[str], tokenizer: CharTokenizer) -> Batch:
    nlength = max(len(s) for s in strings) + 1
    ndataset = len(strings)
    words = [tokenizer.str_to_indices(string) for string in strings]
    features = np.zeros((ndataset, nlength), dtype=jnp.int32)
    labels = np.zeros((ndataset, nlength), dtype=jnp.int32)-1
    for i, word in enumerate(words):
        word_len = word.shape[0]
        features[i, 1:word_len+1] = word
        labels[i, 0:word_len] = word
        labels[i, word_len] = 0
    return Batch(jnp.array(features), jnp.array(labels))
