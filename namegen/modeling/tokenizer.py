from collections import Counter
import jax
import jax.numpy as jnp

__all__ = [
    'CharTokenizer',
]

def get_alphabet(strings: list[str]) -> str:
    counts = Counter(''.join(strings))
    alphabet = ""
    for char, count in counts.most_common():
        alphabet += char
    return alphabet

class CharTokenizer:
    def __init__(self, strings: list[str], alphabet: str | None = None):
        self._alphabet = get_alphabet(strings)
        if alphabet is not None:
            self._alphabet = alphabet + ''.join([c for c in self._alphabet if c not in alphabet])
        if '_' not in self._alphabet:
            self._alphabet = '_' + self._alphabet
        self._nalphabet = len(self._alphabet)
        self._ctoi = {char: i for i, char in enumerate(self._alphabet)}
        self._itoc = {i: char for i, char in enumerate(self._alphabet)}

    def dict_size(self) -> int:
        return self._nalphabet

    def str_to_indices(self, string: str) -> jax.Array:
        return jnp.array([self._ctoi[c] for c in string], dtype=jnp.int32)

    def indices_to_str(self, indices: jax.Array) -> str:
        return ''.join(self._itoc[i.item()] for i in indices)