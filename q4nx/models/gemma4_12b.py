from ..model_converter import __Q4NX_Converter, create_dir_if_not_exists
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from gguf.constants import GGMLQuantizationType
from safetensors.torch import save_file
import numpy as np
import torch
import os

Q4_GROUP = 32


def _blocks(t, qtype: GGMLQuantizationType):
    """(scale (N, K/32) float32, codes (N, K) int8) of a tensor in `qtype`, re-quantizing it first if it is stored otherwise."""
    data = t.data
    if t.tensor_type != qtype:
        data = quantize(np.ascontiguousarray(dequantize(t.data, t.tensor_type), dtype=np.float32), qtype)
    width = 18 if qtype == GGMLQuantizationType.Q4_0 else 34
    blk = np.ascontiguousarray(data).view(np.uint8).reshape(-1, width)
    d = blk[:, :2].copy().view(np.float16).astype(np.float32)
    if qtype == GGMLQuantizationType.Q4_0:
        qs = blk[:, 2:]
        q = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).astype(np.int8) - 8
    else:
        q = blk[:, 2:].view(np.int8)
    n = t.shape[-1] if len(t.shape) > 1 else 1  # gguf shape is (cols, rows)
    return d.reshape(n, -1), q.reshape(n, -1)


class Gemma4_12B(__Q4NX_Converter, model_arch=ModelArch.GEMMA4_12B):
    """Gemma 4 12B: dense, no per-layer inputs, 16x256 Q4_0 / Q8_0 chunks with no min, and one projector file."""

    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()
        self._create_name_maps()

    def _pack16(self, scale: np.ndarray, q: np.ndarray, qtype: GGMLQuantizationType) -> torch.Tensor:
        """(N/16, K/256, chunk) int8: bf16 scales [8 group][16 row], then codes [256 col][16 row] (int4 pairs low = even row)."""
        R, C = self.row_block_size, self.col_block_size
        N, K = q.shape
        P, U = N // R, K // C
        s = torch.from_numpy(scale.reshape(P, R, U, C // Q4_GROUP).transpose(0, 2, 3, 1).copy()).to(torch.bfloat16)
        s = s.view(torch.uint8).reshape(P, U, -1)
        codes = q.reshape(P, R, U, C).transpose(0, 2, 3, 1)  # P U col row
        if qtype == GGMLQuantizationType.Q4_0:
            n = (codes.astype(np.uint8) & 0x0F).reshape(P, U, C, R // 2, 2)
            data = (n[..., 0] | (n[..., 1] << 4)).reshape(P, U, -1)
        else:
            data = np.ascontiguousarray(codes).view(np.uint8).reshape(P, U, -1)
        return torch.cat([s, torch.from_numpy(np.ascontiguousarray(data))], dim=-1).view(torch.int8)

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        if weights_type in ("vision", "audio"):
            # The 12B has no vision or audio tower: the runtime reads the projector GGUF as is, by GGUF name.
            for name, t in self.gguf_tensors.items():
                self.q4nx_tensors[name] = torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type))).to(torch.bfloat16)
            create_dir_if_not_exists(q4nx_path)
            out = os.path.join(q4nx_path, "image_audio_weights.q4nx")
            print(f"[INFO] Saving {len(self.q4nx_tensors)} projector tensors to {out}")
            save_file(self.q4nx_tensors, out)
            return
        if weights_type != "language":
            raise ValueError(f"Unsupported weights_type: {weights_type} for Gemma4-12B")

        if not self._has_lm_head():
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            self.q4nx_tensors["lm_head.weight"] = self._pack16(*_blocks(self.gguf_tensors["token_embd.weight"], GGMLQuantizationType.Q8_0),
                                                               GGMLQuantizationType.Q8_0)

        for name, t in self.gguf_tensors.items():
            q4nx_name = self.forward_name_map[name]
            qtype = self.tensor_q4nx_type_map[name]
            if name == "token_embd.weight":
                w = dequantize(t.data, t.tensor_type) * float(self.hidden_size) ** 0.5
                self.q4nx_tensors[q4nx_name] = torch.from_numpy(w).contiguous().to(torch.bfloat16)
            elif qtype in (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0):
                self.q4nx_tensors[q4nx_name] = self._pack16(*_blocks(t, qtype), qtype)
            else:
                w = torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type)))
                self.q4nx_tensors[q4nx_name] = w.float() if qtype == GGMLQuantizationType.F32 else w.to(torch.bfloat16)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
