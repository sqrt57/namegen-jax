import jax
import jax.numpy as jnp

__all__ = [
    'Model',
    'BigramModel',
]

class Model:
    def dict_size(self) -> int:
        raise NotImplementedError
    
    def context_size(self) -> int:
        raise NotImplementedError

    def __call__(self, idx: jax.Array) -> jax.Array:
        raise NotImplementedError

class BigramModel(Model):
    def __init__(self, pair_counts: jax.Array, config: dict | None = None):
        super().__init__()
        self._dict_size = pair_counts.shape[0]
        self._pair_counts = pair_counts
        self._p = jnp.log(pair_counts)

    def dict_size(self):
        return self._dict_size

    def context_size(self):
        return 1

    def __call__(self, idx: jax.Array):
        return self._p[idx]
