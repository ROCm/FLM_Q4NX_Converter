import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from gguf import GGUFReader, GGMLQuantizationType, dequantize, quantize

from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor, _refit_one_side
from ..model_converter import __Q4NX_Converter

Q8_0 = GGMLQuantizationType.Q8_0
Q4_K = GGMLQuantizationType.Q4_K
BF16 = GGMLQuantizationType.BF16


def _snap(t: np.ndarray, u: np.ndarray):
    """Round per-group scale / min onto the q4_k (bf16 super-block x uint8) grid."""
    S, s8 = _refit_one_side(t.astype(np.float32).reshape(-1, 8))
    M, m8 = _refit_one_side(u.astype(np.float32).reshape(-1, 8))
    return (S[:, None] * s8).reshape(t.shape), (M[:, None] * m8).reshape(u.shape)


def _q4k_chunk(x: np.ndarray):
    mn, mx = np.minimum(x.min(-1), 0.0), x.max(-1)
    best = None
    for steps in (14.0, 14.5, 15.0, 15.5, 16.0, 17.0):
        t, u = _snap((mx - mn) / steps, -mn)
        for it in range(4):
            inv = np.divide(1.0, t, out=np.zeros_like(t), where=t > 0)
            q = np.clip(np.rint((x + u[..., None]) * inv[..., None]), 0, 15)
            if it == 3:
                break
            qm, xm = q.mean(-1), x.mean(-1)
            vq = np.square(q - qm[..., None]).sum(-1)
            t_ls = np.divide(((q - qm[..., None]) * (x - xm[..., None])).sum(-1), vq, out=t.copy(), where=vq > 0)
            t, u = _snap(np.maximum(t_ls, 0.0), np.maximum(t_ls * qm - xm, 0.0))
        err = np.square(t[..., None] * q - u[..., None] - x).sum(-1)
        if best is None:
            best = [err, t, u, q]
        else:
            take = err < best[0]
            best = [np.where(take, err, best[0]), np.where(take, t, best[1]),
                    np.where(take, u, best[2]), np.where(take[..., None], q, best[3])]
    return best[1], best[2], best[3].reshape(x.shape[0], -1)


def _q4k_from_floats(w: torch.Tensor, rows_per_chunk: int = 256):
    """(t, u, q) for `_pack_q4k` from float weights (gguf has no Q4_K encoder): w ~ t * q - u per group of 32.

    Like ggml's make_qkx2: a few range-scaled starting points, each refined by least squares,
    with scale and min kept on the packer's grid so its own re-fit changes nothing.
    """
    g = w.to(torch.float32).numpy().reshape(w.shape[0], -1, 32)
    chunks = [g[i:i + rows_per_chunk] for i in range(0, g.shape[0], rows_per_chunk)]
    with ThreadPoolExecutor(os.cpu_count()) as pool:   # numpy releases the GIL in these ufuncs
        parts = list(pool.map(_q4k_chunk, chunks))
    cat = lambda i: torch.from_numpy(np.ascontiguousarray(np.concatenate([p[i] for p in parts]), dtype=np.float32))
    return cat(0), cat(1), cat(2)


class Qwen38_27B(__Q4NX_Converter, model_arch=ModelArch.QWEN38_27B):
    """Qwen3.8-27B (and other 64-layer / 5120-hidden Qwen3.x GGUFs), laid out like FastFlowLM/Qwen3.8-27B-NPU2.

    Differs from the smaller Qwen3.5 converter in what the FLM qwen3.8-mtp engine reads:
    3 delta-net v heads per qk group, every matrix in Q4_K (re-encoded from any source type),
    embed_tokens / lm_head / ssm_alpha / ssm_beta packed as Q8_0 (alpha/beta rows zero-padded to 64,
    plus bf16 prefill twins), and the MTP block kept under model.mtp.*.
    """

    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using Qwen38_27B converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def _field(self, key):
        return self.gguf_reader.fields[f"qwen35.{key}"].contents()

    def _floats(self, t: GGUFTensor) -> torch.Tensor:
        return torch.from_numpy(dequantize(t.data, t.tensor_type).astype(np.float32))

    def _unpack_as(self, t: GGUFTensor, target: GGMLQuantizationType):
        if target == BF16:
            return [self._floats(t)]
        if t.tensor_type == target:
            return list(t.unpack(target))
        w = self._floats(t)
        if target == Q8_0:
            return list(GGUFTensor.unpack_q8_0(quantize(w.numpy(), Q8_0), w.shape[1]))
        if target == Q4_K:
            return list(_q4k_from_floats(w))
        raise ValueError(f"unsupported target {target.name}")

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        if weights_type == "language":
            self._convert_language()
            self._extract_tokenizer_json(q4nx_path)
        elif weights_type == "vision":
            self._convert_vision()
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen38 model")
        self._export_q4nx_tensors(q4nx_path)

    def _convert_language(self):
        n_mtp = int(self.gguf_reader.fields["qwen35.nextn_predict_layers"].contents()) \
            if "qwen35.nextn_predict_layers" in self.gguf_reader.fields else 0
        n_layers = self.num_layers - n_mtp
        n_qk = self._field("ssm.group_count")
        dk = self._field("ssm.state_size")
        r = self._field("ssm.inner_size") // dk // n_qk   # v heads per qk group, 3 for the 27B
        qk_dim = 2 * n_qk * dk
        dh = self._field("attention.value_length")
        print(f"[INFO] {n_layers} decoder layers + {n_mtp} MTP, {r} v heads per qk group")

        # llama.cpp tiles the v heads (head k of every group, then head k+1); FLM wants them grouped like HF
        def ungroup(x, p, dim=0):
            pat = '(q g p) c -> (g q p) c' if dim == 0 else 'c (q g p) -> c (g q p)'
            return rearrange(x, pat, q=r, p=p).contiguous()

        def rows(parts, fn):
            return [fn(x) for x in parts]

        if not self._has_lm_head():
            raise ValueError("Qwen3.8-27B has an untied lm_head; this GGUF has no output.weight")

        for gguf_tensor in self.gguf_tensors.values():
            name = gguf_tensor.name
            if name not in self.forward_name_map:
                print(f"[WARN] No name mapping for {name}, skipping")
                continue
            new_name = self.forward_name_map[name]
            m = re.match(r"model\.layers\.(\d+)\.", new_name)
            layer_id = int(m.group(1)) if m else -1
            if layer_id >= n_layers:
                new_name = new_name.replace(f"model.layers.{layer_id}.", f"model.mtp.layers.{layer_id - n_layers}.", 1)
            target = self.tensor_q4nx_type_map[name]
            print(f"Processing tensor: {name} ({gguf_tensor.tensor_type.name}) -> {new_name} ({target.name})")
            parts = self._unpack_as(gguf_tensor, target)

            if layer_id >= 0 and self._is_full_attention_layer(layer_id):
                if new_name.endswith("self_attn.q_proj.weight"):   # interleaved [q|gate] per head -> [all q | all gate]
                    parts = rows(parts, lambda x: rearrange(x, '(g p h) c -> (p g h) c', p=2, h=dh).contiguous())
            elif layer_id >= 0:
                if new_name.endswith("self_attn.gate_proj.weight"):
                    parts = rows(parts, lambda x: ungroup(x, dk))
                elif new_name.endswith("qkv_proj.weight"):
                    parts = rows(parts, lambda x: torch.cat([x[:qk_dim], ungroup(x[qk_dim:], dk)]))
                elif new_name.endswith("ssm_out_proj.weight"):   # columns; t/u hold one entry per 32 columns
                    t, u, q = parts
                    parts = [ungroup(t, dk // 32, 1), ungroup(u, dk // 32, 1), ungroup(q, dk, 1)]
                elif new_name.endswith(("ssm_alpha_proj.weight", "ssm_beta_proj.weight")):
                    twin = ungroup(self._floats(gguf_tensor), 1)
                    self.q4nx_tensors[new_name.replace("_proj.weight", "_proj.bf16.weight")] = twin.to(torch.bfloat16)
                    pad = (-parts[0].shape[0]) % self.row_block_size
                    parts = rows(parts, lambda x: F.pad(ungroup(x, 1), (0, 0, 0, pad)))
                elif new_name.endswith("ssm_conv1d.weight"):
                    d = parts[0]
                    parts = [torch.cat([d[:qk_dim], ungroup(d[qk_dim:], dk)]).T.contiguous()]
                elif new_name.endswith(("ssm_a", "ssm_dt.bias")):
                    self.q4nx_tensors[new_name] = rearrange(parts[0].float(), '(q g) -> (g q)', q=r).contiguous()
                    continue

            self.q4nx_tensors[new_name] = self._pack(*parts, tensor_type=target)

    def _convert_vision(self):
        # plain row-major bf16, no (mm_n, mm_k) blocking; only the MLP width is zero-padded (4304 -> 4608)
        for gguf_tensor in self.gguf_tensors.values():
            name = self.forward_name_map[gguf_tensor.name]
            w = self._floats(gguf_tensor)
            if name.startswith("model.visual.blocks.") and name.endswith(".mlp.linear_fc2.weight"):
                w = F.pad(w, (0, (-w.shape[1]) % 512))
            elif name.startswith("model.visual.blocks.") and ".mlp.linear_fc1." in name:
                w = F.pad(w, (0, 0) * (w.dim() - 1) + (0, (-w.shape[0]) % 512))
            self.q4nx_tensors[name] = w.to(torch.bfloat16)
        w = "model.visual.patch_embed.proj.weight"
        self.q4nx_tensors[w] = torch.stack([self.q4nx_tensors.pop(w), self.q4nx_tensors.pop(w + ".1")], dim=2)
