from collections import namedtuple
from pathlib import Path

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
    max_length = max(len(s) for s in strings) + 1
    features = []
    labels = []
    zero = jnp.zeros(1, dtype=jnp.int32)
    for string in strings:
        word = tokenizer.str_to_indices(string)
        word_len = word.shape[0]
        zeros = jnp.zeros(max_length-word_len-1, dtype=jnp.int32)
        word_features = jnp.concat((zero, word, zeros))
        word_labels = jnp.concat((word, zero, zeros-1))
        features.append(word_features)
        labels.append(word_labels)
    features = jnp.stack(features, axis=0)
    labels = jnp.stack(labels, axis=0)
    return Batch(features, labels)
