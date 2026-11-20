"""Checks for the Q4_0 / Q4_1 path of the 32x256 families (Qwen3, Llama, LFM2, Gemma3, Gemma4 E2B/E4B, Phi4, Qwen2,
Qwen2-VL, Qwen3-VL, Nanbeige), on synthetic tensors, no GGUF needed.

    python check_q4nx_packing.py          (or: pytest check_q4nx_packing.py)

1. GGUFTensor.unpack_as() returns a triple that decodes to the source weights under the *target* type's rules, for
   Q4_K, Q8_0, Q6_K and the other Q4 type as source. unpack() does not for Q4_K (its min is subtracted, Q4_1's added).
2. A source already in the target type comes back from unpack_as() bit for bit as unpack() returns it.
3. With optimize_ddr, _pack_q4nx writes each chunk's nibbles as rows 0-15 of all 256 columns, then rows 16-31;
   scales and mins are unchanged, and without the flag the old order is kept.
"""
import numpy as np
import torch
from gguf import dequantize, quantize
from gguf.constants import GGMLQuantizationType as T

from q4nx.gguf_tensor import GGUFTensor
from q4nx.models.qwen3 import Qwen3

ROWS, COLS = 64, 512
RNG = np.random.default_rng(0)


def _q4k_blocks(rows, cols):
    """Random but valid Q4_K data. dmin is set so the subtracted min is the size of the scaled quants, as in real
    weights, where the min centres each group on zero."""
    nb = rows * cols // 256
    b = RNG.integers(0, 256, size=(nb, 144), dtype=np.uint8)
    b[:, 0:2] = np.full(nb, 2e-3, np.float16).view(np.uint8).reshape(nb, 2)
    b[:, 2:4] = np.full(nb, 1.6e-2, np.float16).view(np.uint8).reshape(nb, 2)
    return b.reshape(rows, -1)


def _q6k_blocks(rows, cols):
    nb = rows * cols // 256
    b = RNG.integers(0, 256, size=(nb, 210), dtype=np.uint8)
    b[:, 208:210] = np.full(nb, 1e-3, np.float16).view(np.uint8).reshape(nb, 2)
    return b.reshape(rows, -1)


def _source(qtype):
    if qtype == T.Q4_K:
        data = _q4k_blocks(ROWS, COLS)
    elif qtype == T.Q6_K:
        data = _q6k_blocks(ROWS, COLS)
    else:
        data = quantize(RNG.standard_normal((ROWS, COLS)).astype(np.float32) * 0.02, qtype)
    return GGUFTensor("blk.0.ffn_up.weight", (COLS, ROWS), data, qtype)


def _decode(triple, qtype):
    d, m, q = (x.to(torch.float32) for x in triple)
    s = d.repeat_interleave(32, dim=1)
    return (q * s if qtype == T.Q4_0 else q * s + m.repeat_interleave(32, dim=1)).numpy()


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_unpack_as_matches_source():
    for target in (T.Q4_0, T.Q4_1):
        for src in (T.Q4_K, T.Q8_0, T.Q6_K, T.Q4_0, T.Q4_1):
            t = _source(src)
            ref = dequantize(t.data, src).astype(np.float32)
            err = _rel(_decode(t.unpack_as(target), target), ref)
            assert err < 0.15, f"{src.name} -> {target.name}: rel-L2 {err:.3f}"


def test_unpack_misreads_q4k_as_q4_1():
    t = _source(T.Q4_K)
    err = _rel(_decode(t.unpack(T.Q4_1), T.Q4_1), dequantize(t.data, T.Q4_K).astype(np.float32))
    assert err > 1.0, f"unpack() of Q4_K read as Q4_1 now decodes at rel-L2 {err:.3f}; was the old path fixed?"


def test_same_type_unchanged():
    for qtype in (T.Q4_0, T.Q4_1):
        t = _source(qtype)
        for a, b in zip(t.unpack(qtype), t.unpack_as(qtype)):
            assert torch.equal(a, b), qtype.name


def _packer(optimize_ddr):
    c = object.__new__(Qwen3)
    c.row_block_size, c.col_block_size, c.keep_block_in_2D, c.optimize_ddr = 32, 256, False, optimize_ddr
    return c


def test_half_row_nibble_order():
    d = torch.rand(ROWS, COLS // 32) * 0.01
    m = -torch.rand(ROWS, COLS // 32) * 0.01
    q = torch.from_numpy(RNG.integers(0, 16, size=(ROWS, COLS)).astype(np.float32))
    old = _packer(False)._pack_q4nx(d, m, q).numpy().view(np.uint8)
    new = _packer(True)._pack_q4nx(d, m, q).numpy().view(np.uint8)
    assert old.shape == new.shape == (ROWS // 32 * COLS // 256, 5120)
    assert np.array_equal(old[:, :1024], new[:, :1024]), "scales / mins must not move"
    # decode the new chunks as FLM lays them out: [row half 2][col 256][row 16], low nibble = even row
    nib = np.stack([new[:, 1024:] & 15, new[:, 1024:] >> 4], -1).reshape(-1, 2, 256, 16)
    got = nib.transpose(0, 1, 3, 2).reshape(-1, 32, 256)
    want = q.numpy().astype(np.uint8).reshape(ROWS // 32, 32, COLS // 256, 256).transpose(0, 2, 1, 3).reshape(-1, 32, 256)
    assert np.array_equal(got, want)
    # the old order is [col 256][row 32]
    nib = np.stack([old[:, 1024:] & 15, old[:, 1024:] >> 4], -1).reshape(-1, 256, 32)
    assert np.array_equal(nib.transpose(0, 2, 1), want)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
