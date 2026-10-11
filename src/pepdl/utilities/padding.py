"""Fixed-width token ids **and the padding mask** for the unified-model predictors.

# Why the mask is not optional

Every unified-model head was trained with ``padding_mask = (attention_mask == 0)``
(``pepdl-train``'s dataset adapter emits the attention mask; every training and evaluation step
passes it). A forward pass without it lets the encoder attend to the padding positions, an input
the model never saw, and the error grows with the amount of padding, i.e. it is worst for short
peptides. Measured on the ionmob ``meier_test`` split, CCS without the mask is off by +35 Å² on
2+ (MAE 32 Å²); with it, +3 Å² (MAE 12 Å² balanced over charges 2-4). The CCS, RT and intensity
paths all omitted it; only the ionization path passed it.

# The mask comes from lengths, never from token values

``pad_token_id`` is not 0, and id 0 is a real token (``-[UNIMOD:258]``). Rows are padded with
``pad_id`` up to the longest tokenisation in the input and with 0 from there to ``pad_len``, the
historical layout the chunked builders reproduce. So neither ``ids == pad_id`` nor ``ids == 0``
identifies padding. The mask is ``position >= row length``, which is all the training adapter's
attention mask says, too.

# Truncation matches training

The training adapter truncates an over-long tokenisation as ``tokens[:max_length - 1] + [SEP]``,
so the model always saw a closing ``[SEP]``. A plain slice would drop it. Peptides this long
(more than ``pad_len - 2`` residues) are rare, but the two paths must agree.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def token_ids_and_mask(
    tokenizer, sequences: List[str], pad_len: int = 50, chunk_size: int = 65536
) -> Tuple[NDArray, NDArray]:
    """``(ids, padding_mask)`` for ``sequences``, built ``chunk_size`` rows at a time.

    ``ids`` is ``(n, pad_len)`` in the narrowest integer dtype the ids fit; ``padding_mask`` is
    ``(n, pad_len)`` bool, True at padding. Both are host arrays, so a caller moves one inference
    batch at a time to the device.

    The id layout (PAD up to the input-wide ``width``, then 0) is independent of ``chunk_size``,
    because ``width`` is taken from the whole input, not per chunk. Under the mask those positions
    no longer reach the output, but the layout is kept so ids stay comparable with the one-shot
    tokenisation.
    """
    pad_id = int(tokenizer.pad_token_id)
    sep_id = int(tokenizer.sep_token_id)
    n = len(sequences)
    # int16 halves the resident matrix. `vocab_size` is a COUNT, and only bounds the ids under a
    # dense-id invariant the tokenizer API does not promise, so the guess is checked against the
    # ids actually observed, per chunk, and widened if it was ever wrong. Do not replace that check
    # with a try/except OverflowError: NumPy 1.x casts out-of-range values silently.
    narrow = max(int(tokenizer.vocab_size), pad_id, sep_id) < 2 ** 15
    ids = np.zeros((n, pad_len), dtype=np.int16 if narrow else np.int32)
    lengths = np.zeros(n, dtype=np.int64)

    for start in range(0, n, chunk_size):
        chunk = sequences[start:start + chunk_size]
        # pad=False: the per-chunk width is deliberately not applied here (see the docstring).
        encoded, _ = tokenizer.encode_batch(tokenizer.tokenize_batch(chunk), pad=False)
        if ids.dtype == np.int16:
            hi = max((max(row) for row in encoded if row), default=0)
            lo = min((min(row) for row in encoded if row), default=0)
            if hi > 2 ** 15 - 1 or lo < -(2 ** 15):
                ids = ids.astype(np.int32)
        for j, row in enumerate(encoded):
            keep = row if len(row) <= pad_len else list(row[:pad_len - 1]) + [sep_id]
            lengths[start + j] = len(keep)
            ids[start + j, :len(keep)] = keep

    padding_mask = np.zeros((n, pad_len), dtype=bool)
    if n == 0:
        return ids, padding_mask

    width = int(lengths.max())
    col = np.arange(pad_len)
    for start in range(0, n, chunk_size):  # chunked so the boolean blocks stay bounded too
        stop = min(start + chunk_size, n)
        is_pad = col[None, :] >= lengths[start:stop, None]
        ids[start:stop][is_pad & (col[None, :] < width)] = pad_id
        padding_mask[start:stop] = is_pad
    return ids, padding_mask
