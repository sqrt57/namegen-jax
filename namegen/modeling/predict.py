import jax
import jax.numpy as jnp
import optax

from namegen.dataset import Batch
from namegen.modeling.tokenizer import CharTokenizer
from namegen.modeling.model import Model

__all__ = [
    'generate',
]

def generate(key: jax.Array, tokenizer: CharTokenizer, model: Model, N=20, T=1, max_len=100):
    context_size = model.context_size()
    x = jnp.zeros((N, 1), dtype=jnp.int32)
    for i in range(max_len):
        x_win = x[:, -context_size:] if x.shape[1] >= context_size else x
        logits = model(x_win)
        logits = logits[:, -1, :]
        key, subkey = jax.random.split(key)
        next = jax.random.categorical(subkey, logits / T, axis=1)
        x = jnp.concat((x, next.reshape(-1, 1)), axis=1)
        if (x[:,1:] == 0).any(axis=1).all(axis=0).item():
            break
    result = []
    for n in range(N):
        row = x[n, 1:]
        zeros = jnp.nonzero(row == 0)[0]
        if len(zeros) > 0:
            row = row[:zeros[0].item()]
        word = tokenizer.indices_to_str(row)
        result.append(word)
    return result

def calculate_loss(dataset: Batch, model: Model):
    dict_size = model.dict_size()
    pred = model(dataset.features).reshape(-1, dict_size)
    labels = dataset.labels.ravel()
    return optax.losses.softmax_cross_entropy_with_integer_labels(pred[labels >= 0], labels[labels >= 0]).mean()
