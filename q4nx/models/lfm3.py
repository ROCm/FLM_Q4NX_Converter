from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from einops import rearrange
import torch
from gguf.constants import GGMLQuantizationType
from ..gguf_tensor import GGUFTensor

class LFM3(__Q4NX_Converter, model_arch=ModelArch.LFM3):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()
        self._create_name_maps()

    def _pack_q4nx_half_rows(self, *args) -> torch.Tensor:
        """`_pack_q4nx`, with each chunk's nibbles stored as rows 0-15 of all columns, then rows 16-31 (FLM's LFM3 order)."""
        packed = self._pack_q4nx(*args)
        dm = 2 * 2 * self.row_block_size * (self.col_block_size // 32)  # bf16 scales + mins
        q = packed[:, dm:].view(torch.uint8)
        nib = torch.stack([q & 0x0F, q >> 4], dim=-1)  # (n, col * row/2, 2) -> one nibble per (col, row)
        nib = rearrange(nib, 'n (c h r) b -> n (h c r b)', c=self.col_block_size, h=2, b=2)
        nib = rearrange(nib, 'n (k b) -> n k b', b=2)
        out = packed.clone()
        out[:, dm:] = (nib[..., 0] | (nib[..., 1] << 4)).view(torch.int8)
        return out

    @staticmethod
    def _unpack_as(t: GGUFTensor, qtype: GGMLQuantizationType):
        """unpack() hands Q4_K / Q8_0 blocks back in their own layout, which _pack_q4nx would misread: re-quantize first."""
        if t.tensor_type != qtype:
            w = torch.from_numpy(dequantize(t.data, t.tensor_type)).to(torch.bfloat16).float().numpy()  # as unpack() rounds
            t = GGUFTensor(name=t.name, shape=t.shape, data=quantize(w, qtype), tensor_type=qtype)
        return t.unpack(qtype)

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type != "language":
            raise ValueError("LFM3 has language weights only")
        self.q4nx_tensors = {}

        if not self._has_lm_head():
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self._unpack_as(self.gguf_tensors["token_embd.weight"], GGMLQuantizationType.Q4_1)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx_half_rows(*unpacked)

        for name, gguf_tensor in self.gguf_tensors.items():
            if name not in self.forward_name_map:  # rope_freqs: the runtime computes its own
                print(f"[INFO] Skip {name}")
                continue
            q4nx_name = self.forward_name_map[name]
            if gguf_tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16) \
                    or self.tensor_q4nx_type_map[name] == GGMLQuantizationType.BF16 or name == "token_embd.weight":
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                self.q4nx_tensors[q4nx_name] = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                continue
            unpacked = self._unpack_as(gguf_tensor, self.tensor_q4nx_type_map[name])
            self.q4nx_tensors[q4nx_name] = self._pack_q4nx_half_rows(*unpacked)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
