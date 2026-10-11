"""Token ids and the padding mask, for every unified-model predictor.

Every head was trained with ``padding_mask = (attention_mask == 0)``. The CCS, RT and intensity
paths used to call the model without it, and CCS was off by +35 Å² on 2+ as a result (see
``pepdl.utilities.padding``). These tests pin the contract the fix rests on:

* the chunked builder reproduces an INDEPENDENT one-shot oracle: the tokenizer's own
  ``__call__(padding=True)`` ids and attention mask, zero-extended to ``pad_len``, with the
  training adapter's truncation (``tokens[:pad_len - 1] + [SEP]``);
* the mask is a function of each row's own length, never of token values. Id 0 is a real token
  (``-[UNIMOD:258]``) and must stay unmasked; ``[CLS]`` and ``[SEP]`` are never masked;
* every predictor routes through the same builder, so they cannot drift again.

The model-level tests (does the mask reach the forward call; is the CCS right) are in
``test_padding_mask_models.py``, which needs the weights.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mscorepy")

PAD_LEN = 50

# "K" is the shortest (most padding); the 30-mer is the digest's max length; the UNIMOD entries
# exercise composite tokens, and the C-terminal one tokenises to id 0.
SEQS = ["PEPTIDEK", "AC[UNIMOD:4]DEK", "K", "ACDEFGHIKLMNPQRSTVWY", "M" * 29 + "K",
        "PEPTIDEK-[UNIMOD:258]"]


def _tokenizer():
    from pepdl.utilities.tokenizers import ProformaTokenizer

    return ProformaTokenizer.with_defaults()


def _predictor(module, name):
    """A predictor with only the pieces the tokenisation path touches (no weights loaded)."""
    import importlib

    cls = getattr(importlib.import_module(module), name)
    obj = cls.__new__(cls)
    obj.tokenizer = _tokenizer()
    obj._device = torch.device("cpu")
    return obj


PREDICTORS = [
    ("pepdl.ccs.predictors", "DeepPeptideIonMobilityApex"),
    ("pepdl.rt.predictors", "DeepChromatographyApex"),
    ("pepdl.ionization.predictors", "DeepChargeStateDistribution"),
]


def _oracle(seqs, pad_len=PAD_LEN):
    """One-shot ids and padding mask, from the tokenizer's own whole-batch call."""
    tok = _tokenizer()
    sep = int(tok.sep_token_id)
    out = tok(list(seqs), padding=True)
    ids = np.zeros((len(seqs), pad_len), dtype=np.int64)
    mask = np.ones((len(seqs), pad_len), dtype=bool)
    for i, (row, att) in enumerate(zip(out["input_ids"], out["attention_mask"])):
        n = sum(att)
        if n > pad_len:  # the training adapter's truncation: keep a closing [SEP]
            row, att = list(row[:pad_len - 1]) + [sep], [1] * pad_len
        row, att = list(row[:pad_len]), list(att[:pad_len])
        ids[i, :len(row)] = row
        mask[i, :len(att)] = [a == 0 for a in att]
    return ids, mask


def _built(seqs, pad_len=PAD_LEN, chunk_size=65536):
    from pepdl.utilities.padding import token_ids_and_mask

    ids, mask = token_ids_and_mask(_tokenizer(), list(seqs), pad_len=pad_len, chunk_size=chunk_size)
    return ids.astype(np.int64), mask


def test_pad_token_id_is_not_zero_and_zero_is_a_real_token():
    """Guards the comparisons below, and is the reason the mask cannot come from token values."""
    tok = _tokenizer()
    assert int(tok.pad_token_id) != 0
    assert 0 in tok.encode(tok.tokenize("PEPTIDEK-[UNIMOD:258]"))


@pytest.mark.parametrize("chunk_size", [1, 2, 3, len(SEQS), len(SEQS) + 10])
def test_chunked_build_matches_the_one_shot_oracle(chunk_size):
    want_ids, want_mask = _oracle(SEQS)
    got_ids, got_mask = _built(SEQS, chunk_size=chunk_size)
    assert np.array_equal(got_ids, want_ids)
    assert np.array_equal(got_mask, want_mask)
    assert int(_tokenizer().pad_token_id) in got_ids[SEQS.index("K")], "expected a real PAD region"


@pytest.mark.parametrize("chunk_size", [1, 2, 8])
@pytest.mark.parametrize(
    "seqs, why",
    [
        (["PEPTIDEK", "ANOTHERK"], "all rows the same length: no PAD-id region"),
        (["K", "K", "K"], "all rows minimal and equal"),
        (["A" * 48, "K"], "width lands exactly on pad_len for one row"),
        (["A" * 90, "A" * 91], "every row truncated: no padding anywhere"),
        (["A" * 90, "PEPTIDEK"], "one truncated row beside a short one"),
        (["PEPTIDEK"], "a single sequence"),
    ],
)
def test_width_edge_cases_match_the_oracle(seqs, why, chunk_size):
    want_ids, want_mask = _oracle(seqs)
    got_ids, got_mask = _built(seqs, chunk_size=chunk_size)
    assert np.array_equal(got_ids, want_ids), why
    assert np.array_equal(got_mask, want_mask), why


@pytest.mark.parametrize("pad_len", [8, 32, 64])
def test_non_default_pad_len(pad_len):
    want_ids, want_mask = _oracle(SEQS, pad_len=pad_len)
    got_ids, got_mask = _built(SEQS, pad_len=pad_len, chunk_size=2)
    assert got_ids.shape == got_mask.shape == (len(SEQS), pad_len)
    assert np.array_equal(got_ids, want_ids)
    assert np.array_equal(got_mask, want_mask)


def test_mask_is_true_exactly_past_each_rows_own_length():
    tok = _tokenizer()
    ids, mask = _built(SEQS)
    for i, s in enumerate(SEQS):
        n = len(tok.encode(tok.tokenize(s)))
        assert mask[i].tolist() == [False] * n + [True] * (PAD_LEN - n), s
        assert ids[i, 0] == tok.cls_token_id and ids[i, n - 1] == tok.sep_token_id, s


def test_a_real_id_zero_token_is_not_masked():
    ids, mask = _built(["PEPTIDEK-[UNIMOD:258]", "K"])
    zero_at = int(np.flatnonzero(ids[0] == 0)[0])
    assert not mask[0, zero_at], "the -[UNIMOD:258] token was masked as padding"


def test_truncation_keeps_a_closing_sep_like_training():
    tok = _tokenizer()
    ids, mask = _built(["A" * 90])
    assert ids[0, -1] == tok.sep_token_id
    assert not mask.any()


def test_empty_input():
    ids, mask = _built([])
    assert ids.shape == mask.shape == (0, PAD_LEN)


def test_ids_fit_the_narrowed_dtype():
    """The int16 narrowing keys off `vocab_size`, a COUNT, not an id ceiling: an id outside
    int16 must widen the matrix rather than wrap."""
    from pepdl.utilities.padding import token_ids_and_mask

    ids, _ = token_ids_and_mask(_tokenizer(), SEQS)
    assert ids.dtype in (np.int16, np.int32)
    assert int(ids.max()) <= np.iinfo(ids.dtype).max
    assert int(ids.min()) >= 0


@pytest.mark.parametrize("module, name", PREDICTORS)
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_every_predictor_returns_ids_and_mask_from_the_shared_builder(module, name, chunk_size):
    apex = _predictor(module, name)
    want_ids, want_mask = _oracle(SEQS)
    ids, mask = apex._token_ids(SEQS, chunk_size=chunk_size)
    assert np.array_equal(ids.astype(np.int64), want_ids)
    assert np.array_equal(mask, want_mask)
    tokens, tmask = apex._preprocess_sequences(SEQS)
    assert torch.equal(tokens, torch.as_tensor(want_ids))
    assert torch.equal(tmask, torch.as_tensor(want_mask))


def test_intensity_preprocess_returns_the_mask_at_the_model_width():
    from pepdl.intensity.predictors import DeepPeptideIntensityPredictor

    apex = DeepPeptideIntensityPredictor.__new__(DeepPeptideIntensityPredictor)
    apex.tokenizer = _tokenizer()
    apex._torch = torch
    apex.model = type("M", (), {"max_seq_len": 32})()
    tokens, mask, charge, ce = apex._preprocess(SEQS, [2] * len(SEQS), [0.3] * len(SEQS))
    want_ids, want_mask = _oracle(SEQS, pad_len=32)
    assert torch.equal(tokens, torch.as_tensor(want_ids))
    assert torch.equal(mask, torch.as_tensor(want_mask))
    assert charge.tolist() == [2] * len(SEQS) and ce.shape == (len(SEQS),)
