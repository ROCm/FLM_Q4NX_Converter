import gguf

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange, unpack

class Gemma4(__Q4NX_Converter, model_arch=ModelArch.GEMMA4):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

        
        # read "embedding_length_per_layer_input" from gguf metadata
        field = self.gguf_reader.fields.get("gemma4.embedding_length_per_layer_input", None)
        self.embedding_length_per_layer_input = field.contents() if field is not None else None 
    def initialize(self):
        super().initialize()

    
    
    
    def _quantize_embedding_int8(self, weight: torch.Tensor, group_size: int = 32):
        """
        Quantize an embedding-table weight (rows=vocab, cols=hidden) into
        the plain per-row / per-group symmetric int8 format expected by
        the FLM runtime for lookup-style tensors (as opposed to the
        packed block format used for matmul weights).

        For every row, the columns are split into contiguous groups of
        `group_size` elements. Each group gets its own scale:
            scale = max(|group|) / 127
            q = round(group / scale), clamped to [-127, 127]

        Returns
        -------
        (q4nx_weight, q4nx_scale) : Tuple[torch.Tensor, torch.Tensor]
            q4nx_weight: int8 tensor of shape (rows, cols)
            q4nx_scale:  float32 tensor of shape (rows, cols // group_size)
        """
        assert weight.ndim == 2, "Embedding weight must be a 2D matrix"
        rows, cols = weight.shape
        assert cols % group_size == 0, f"Embedding hidden size {cols} must be divisible by group_size {group_size}"

        w = weight.to(torch.float32)
        w_grouped = w.view(rows, cols // group_size, group_size)

        amax = w_grouped.abs().amax(dim=-1)
        scale = (amax / 127.0).clamp(min=1e-8)

        q = torch.round(w_grouped / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        q = q.reshape(rows, cols).contiguous()
        scale = scale.to(torch.float32).contiguous()

        return q, scale

    def reshape_matrix_to_block_matrix_for_mvm(self, weight: torch.Tensor, row_block_size: int=32) -> torch.Tensor:
        """
            Assume the weights is a 2D matrix of W x H
            The function will reshape weights into (blocks, H, row_block_size)
            Where blocks = W // row_block_size
            
            In addition, each block(row_block_size x H) is in column-major order
        """
        assert weight.ndim == 2, "Input weight must be a 2D matrix"
        
        W, H = weight.shape
        
        
        assert W % row_block_size == 0, f"Weight rows {W} must be divisible by row_block_size {row_block_size}"
        blocks = W // row_block_size
        
        
        # Step 1: Reshape to (blocks, row_block_size, H)
        weight = weight.contiguous()

        weight = rearrange(weight,
                           '(blocks row_block_size) H -> blocks row_block_size H',
                           blocks=blocks, row_block_size=row_block_size
                           )
        # Step 2: Transpose each block to be in column-major order
        weight = rearrange(weight,
                           'blocks row_block_size H -> blocks H row_block_size',
                           )
        return weight
        

    
    
    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}

        if weights_type == "language":
            
            if not self._has_lm_head():
                print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
                unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
                self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)      

            for key, gguf_tensor in self.gguf_tensors.items():
                if "token_embd.weight"  == gguf_tensor.name:
                    w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                    w = w * float(self.hidden_size) **0.5
                    w = torch.from_numpy(w).contiguous()
                    q_w, scale = self._quantize_embedding_int8(w)
                    name = self.forward_name_map[gguf_tensor.name]
                    self.q4nx_tensors[name] = q_w
                    self.q4nx_tensors[f"{name}.scale"] = scale
                    continue
                elif "per_layer_token_embd.weight"  ==  gguf_tensor.name:
                    w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                    w = w*float(self.embedding_length_per_layer_input)**0.5
                    w = torch.from_numpy(w).contiguous()
                    q_w, scale = self._quantize_embedding_int8(w)
                    name = self.forward_name_map[gguf_tensor.name]
                    self.q4nx_tensors[name] = q_w
                    self.q4nx_tensors[f"{name}.scale"] = scale
                    continue
                elif "per_layer_model_proj.weight" in gguf_tensor.name:
                    unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])
                    w=unpacked[0]
                    
                    # since output projection == self.embedding_length_per_layer_input* self.num_layers, we want to actually split the matrxi
                    
                    assert w.shape[0] == self.embedding_length_per_layer_input * self.num_layers, f"Expected output projection weight shape[0] to be {self.embedding_length_per_layer_input * self.num_layers}, but got {w.shape[0]}"
                    
                    # let us save a prefill version of  W for debugging
                    #NOTE: TODO: We use vision MM
                    w_for_prefill = self.vision_mm_weight_rearrange(w).contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[f"{self.forward_name_map[gguf_tensor.name]}_prefill"] = w_for_prefill
                    
                    w_per_layer = w.reshape(self.num_layers, self.embedding_length_per_layer_input, w.shape[1])
                    
                    for layer_idx in range(self.num_layers):
                        layer_w = w_per_layer[layer_idx]
                        layer_w = self.reshape_matrix_to_block_matrix_for_mvm(layer_w)
                        layer_w = layer_w.contiguous().to(torch.bfloat16)
                        self.q4nx_tensors[f"{self.forward_name_map[gguf_tensor.name]}_layer{layer_idx}"] = layer_w
                    
                    
                    # w = self.reshape_matrix_to_block_matrix_for_mvm(unpacked[0])
                    # w = w.contiguous().to(torch.bfloat16)
                    # self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w

                    continue
                                        
                elif "inp_gate.weight" in gguf_tensor.name or ".proj.weight" in gguf_tensor.name:
                    
                    unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])
                    
                    
                    if "inp_gate.weight" in gguf_tensor.name or ".proj.weight" in gguf_tensor.name:
                        # dedicate for prefill with vision MM
                        w_for_prefill = self.vision_mm_weight_rearrange(unpacked[0]).contiguous().to(torch.bfloat16)
                        self.q4nx_tensors[f"{self.forward_name_map[gguf_tensor.name]}_prefill"] = w_for_prefill                    
                    
                    
                    
                    w = self.reshape_matrix_to_block_matrix_for_mvm(unpacked[0])
                    w = w.contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    
                    continue
                
                unpacked = gguf_tensor.unpack(self.tensor_q4nx_type_map[gguf_tensor.name])

                torch.set_printoptions(threshold=16, edgeitems=5, linewidth=200)

                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

            self._extract_tokenizer_json(q4nx_path)
        elif weights_type == "vision":
            
            for key, gguf_tensor in self.gguf_tensors.items():
                unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
                assert len(unpacked) == 1
                assert type(unpacked[0]) == torch.Tensor, "Vision model tensors"

                
                weights = unpacked[0]
                if weights.dtype != torch.bfloat16:
                    # convert to bfloat16
                    weights = weights.to(torch.bfloat16)
                if gguf_tensor.name not in self.forward_name_map:
                    # check is not start with "v"
                    if not gguf_tensor.name.startswith("v"):
                        continue
                    else:
                        raise ValueError(f"Tensor name {gguf_tensor.name} not found in forward_name_map for vision model")
                new_name = self.forward_name_map[gguf_tensor.name]
                
                if new_name == "model.vision.embedding_projection.weight":
                    #TODO: no need transpose?
                    
                    weights = self.vision_mm_weight_rearrange(weights)                    
                elif new_name == "model.vision.patch_embedder.position_embedding_table":
                    assert weights.ndim == 3
                    assert weights.shape[0] ==2
                    #TODO: also transpose each weights first ?
                    
                    
                    weights = torch.stack([
                        self.vision_mm_weight_rearrange(weights[0].T.contiguous()),
                        self.vision_mm_weight_rearrange(weights[1].T.contiguous()),
                    ])
                    
                elif new_name == "model.vision.patch_embd.weight":
                    assert(weights.ndim == 4), f"Expected patch embedding weight to be 4D, but got {weights.ndim}D"
                    
                    # Revert llama.cpp's permute(0, 3, 1, 2) to get back to original 2D (n_embd, ksize_sq_c)
                    weights = weights.permute(0, 2, 3, 1).contiguous()
                    weights = weights.reshape(weights.shape[0], -1)
                    # weights = weights.T.contiguous()
                    
                    weights = self.vision_mm_weight_rearrange(weights)
                elif new_name.endswith("ffn_down.weight") or new_name.endswith("ffn_gate.weight") or new_name.endswith("ffn_up.weight") \
                    or new_name.endswith("k_proj.weight") or new_name.endswith("q_proj.weight")\
                    or new_name.endswith("v_proj.weight") or new_name.endswith("out_proj.weight") \
                    or new_name.endswith("gate_proj.weight") or new_name.endswith("up_proj.weight") or new_name.endswith("down_proj.weight"):
                        
                        
                    weights = self.vision_mm_weight_rearrange(weights)


                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = weights
                
        elif weights_type == "audio":
            for key, gguf_tensor in self.gguf_tensors.items():
                unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
                assert len(unpacked) == 1
                assert type(unpacked[0]) == torch.Tensor, "Audio model tensors"

                
                weights = unpacked[0]
                if weights.dtype != torch.bfloat16:
                    # convert to bfloat16
                    weights = weights.to(torch.bfloat16)
                    
                if gguf_tensor.name not in self.forward_name_map:
                    if not gguf_tensor.name.startswith("a"):
                        continue
                    else:
                        raise ValueError(f"Tensor name {gguf_tensor.name} not found in forward_name_map for audio model")
                new_name = self.forward_name_map[gguf_tensor.name]

                if new_name.endswith("conv_dw.weight"):
                    weights = weights.T.contiguous()
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = weights
                    continue

                matrix_suffix_for_weight_rearrange = [
                    "attn_k_proj.weight", "attn_out_proj.weight",
                    # "attn_k_proj.rel_weight",
                    "attn_q_proj.weight", "attn_v_proj.weight",
                    "ffn.down_proj.weight", "ffn.down_proj_1.weight",
                    "ffn.up_proj.weight", "ffn.up_proj_1.weight",
                    "conv_pw_1.weight", "conv_pw_2.weight",
                    "model.audio.embedding_projection.weight",
                    "model.audio.encode_input_projection.weight",
                    "model.audio.pre_encoder.weight"
                ]
                
                
                for suf in matrix_suffix_for_weight_rearrange:

                    if new_name.endswith(suf) or new_name == suf:
                        weights = self.audio_mm_weight_rearrange(weights).contiguous()
                        break

            

            
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = weights
    
                
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Gemma4 model")
        self._export_q4nx_tensors(q4nx_path)
