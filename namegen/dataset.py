from dataclasses import dataclass
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

@dataclass(frozen=True)
class Batch:
    features: jax.Array
    labels: jax.Array

def get_dataset(strings: list[str], tokenizer: CharTokenizer) -> Batch:
    features = []
    labels = []
    zero = jnp.zeros(1, dtype=jnp.int32)
    for string in strings:
        word = tokenizer.str_to_indices(string)
        word_features = jnp.concat((zero, word))
        word_labels = jnp.concat((word, zero))
        features.append(word_features)
        labels.append(word_labels)
    max_length = max(f.shape[0] for f in features)
    features = [jnp.concat((f, jnp.zeros(max_length-f.shape[0], dtype=jnp.int32))) for f in features]
    labels = [jnp.concat((l, jnp.zeros(max_length-l.shape[0], dtype=jnp.int32)-1)) for l in labels]
    features = jnp.stack(features, axis=0)
    labels = jnp.stack(labels, axis=0)
    return Batch(features, labels)
