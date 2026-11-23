from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize
import torch
from gguf.constants import GGMLQuantizationType

class LFM3(__Q4NX_Converter, model_arch=ModelArch.LFM3):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()
        self._create_name_maps()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type != "language":
            raise ValueError("LFM3 has language weights only")
        self.q4nx_tensors = {}

        if not self._has_lm_head():
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack_as(GGMLQuantizationType.Q4_1)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

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
            unpacked = gguf_tensor.unpack_as(self.tensor_q4nx_type_map[name])
            self.q4nx_tensors[q4nx_name] = self._pack_q4nx(*unpacked)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
