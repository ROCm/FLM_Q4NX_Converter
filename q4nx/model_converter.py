

from abc import ABC, abstractmethod
from gguf import GGUFReader
from .constants import ModelArch, ModelArchNames
from .constants import ModelArchConfigs
from .gguf_tensor import GGUFTensor, GGMLQuantizationType, _refit_one_side
from typing import List, Dict, Type
import os
import json
import re
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
from .utils import round_up_to_multiple, get_relativeL2, get_relativeL1, get_rmse, get_cosine_similarity, create_dir_if_not_exists
from safetensors.torch import save_file
from q4nx.gguf_tensor import GGUFTensor
from gguf import Q8_0, GGUFReader, dequantize, quantize, GGMLQuantizationType
# Registry to store model classes by architecture
_MODEL_REGISTRY: Dict[ModelArch, Type['__Q4NX_Converter']] = {}

class __Q4NX_Converter(ABC):
    model_arch: ModelArch
    gguf_reader: GGUFReader
    gguf_tensors: Dict[str, GGUFTensor]
    q4nx_tensors: Dict[str, torch.Tensor]
    hidden_size: int
    num_layers: int
    embed_length:int
    q4nx_config: Dict

    row_block_size: int
    col_block_size: int
    parallel_size: int   # vector len size for efficient parallel vector operation (MXFP4 only:
                         # q4nx/q8nx/q4k blocks are plain column major over row_block_size rows)
    keep_block_in_2D: bool

    default_tensor_type: GGMLQuantizationType
    
    # specific for vision models
    vision_MM_K:int|None
    vision_MM_N:int|None
    
    
    audio_MM_K:int|None
    audio_MM_N:int|None
    
    forward_name_map: Dict[str, str]
    backward_name_map: Dict[str, str]
    tensor_q4nx_type_map: Dict[str, GGMLQuantizationType]

    def __init__(self):
        raise TypeError("This class is virtual, do not instantiate it directly")


    def __init_subclass__(cls, model_arch: ModelArch, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.model_arch = model_arch
        # Register the subclass in the registry
        _MODEL_REGISTRY[model_arch] = cls

    def initialize(self):
        self._read_gguf_tensors()
        self._read_gguf_metadata()
        self._load_config()

    @abstractmethod
    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        pass

    def _read_gguf_tensors(self):
        """
        Load the GGUF file and parse the tensors.

        Returns:
            None
        """
        self.gguf_tensors = {}
        for tensor in self.gguf_reader.tensors:
            self.gguf_tensors[tensor.name] = GGUFTensor(
                name=tensor.name,
                shape=tuple(tensor.shape.tolist()),
                data=tensor.data,
                tensor_type=tensor.tensor_type
            )

    def _read_gguf_metadata(self):
        for field in self.gguf_reader.fields.values():

            if field.name.endswith("embedding_length"):
                self.hidden_size = field.contents()
            elif field.name.endswith("feed_forward_length"):
                self.intermediate_size = field.contents()
            elif field.name.endswith("block_count"):
                self.num_layers = field.contents()


    def _load_config(self, config_file_path: str = "configs"):
        config_path = os.path.join(config_file_path, ModelArchConfigs[self.model_arch])
        print(f"[INFO] Loading Q4NX config from {config_path}")
        self.q4nx_config = json.load(open(config_path))
        self.row_block_size = self.q4nx_config["q4nx_config"]["row_block_size"]
        self.col_block_size = self.q4nx_config["q4nx_config"]["col_block_size"]
        self.parallel_size = self.q4nx_config["q4nx_config"]["parallel_size"]
        self.keep_block_in_2D = self.q4nx_config["q4nx_config"]["keep_block_in_2D"]
        if self.q4nx_config["default_tensor_type"] == "Q4_0":
            self.default_tensor_type = GGMLQuantizationType.Q4_0
        elif self.q4nx_config["default_tensor_type"] == "Q4_1":
            self.default_tensor_type = GGMLQuantizationType.Q4_1
        elif self.q4nx_config["default_tensor_type"] == "Q8_0":
            self.default_tensor_type = GGMLQuantizationType.Q8_0
        elif self.q4nx_config["default_tensor_type"] == "Q4_K":
            self.default_tensor_type = GGMLQuantizationType.Q4_K
        else:
            raise ValueError("Unsupported default_tensor_type in config")
        
        if "vision_config" in self.q4nx_config:
            self.vision_MM_K = self.q4nx_config["vision_config"]["vision_MM_K"]
            self.vision_MM_N = self.q4nx_config["vision_config"]["vision_MM_N"]
        else:
            self.vision_MM_K = None
            self.vision_MM_N = None
            
        if "audio_config" in self.q4nx_config:
            self.audio_MM_K = self.q4nx_config["audio_config"]["audio_MM_K"]
            self.audio_MM_N = self.q4nx_config["audio_config"]["audio_MM_N"]
        else:
            self.audio_MM_K = None
            self.audio_MM_N = None
        self._create_name_maps()

    def get_ggml_type(self, q4nx_name: str) -> GGMLQuantizationType:
        if q4nx_name == "Q4_0":
            return GGMLQuantizationType.Q4_0
        elif q4nx_name == "Q4_1":
            return GGMLQuantizationType.Q4_1
        elif q4nx_name == "Q8_0":
            return GGMLQuantizationType.Q8_0
        elif q4nx_name == "Q4_K":
            return GGMLQuantizationType.Q4_K
        elif q4nx_name == "BF16":
            return GGMLQuantizationType.BF16

        else:
            raise ValueError(f"Unsupported q4nx_name: {q4nx_name}")

    def _create_name_maps(self):
        print("[INFO] Creating name maps...")
        self.forward_name_map = {}
        self.backward_name_map = {}
        self.tensor_q4nx_type_map = {}

        # --- Phase 1: Collect every unique {bid} gguf_name pattern from the config ---
        bid_templates = {
            param_info["gguf_name"]
            for param_info in self.q4nx_config["name_map"].values()
            if "{bid}" in param_info["gguf_name"]
        }

        # --- Phase 2: For each unique pattern, detect the layer count from actual GGUF tensors ---
        bid_patterns = {}  # gguf_name_template -> bid_range
        for gguf_template in bid_templates:
            regex = re.compile(
                "^" + re.escape(gguf_template).replace(r"\{bid\}", r"(\d+)") + "$"
            )
            found = sorted(
                int(match.group(1))
                for name in self.gguf_tensors
                for match in [regex.match(name)]
                if match
            )
            if found:
                num_layers = max(found) + 1
                print(f"[INFO] Detected {num_layers} layers for pattern '{gguf_template}'")
                bid_patterns[gguf_template] = range(num_layers)
            else:
                bid_patterns[gguf_template] = range(0)

        # --- Phase 3: Apply the mapping using the pre-detected ranges ---
        for param_info in self.q4nx_config["name_map"].values():
            gguf_template = param_info["gguf_name"]
            if "{bid}" in gguf_template:
                bid_range = bid_patterns[gguf_template]
                if not bid_range:
                    continue  # pattern not present in this GGUF file
                for bid in bid_range:
                    gguf_name = gguf_template.format(bid=bid)
                    q4nx_name = param_info["q4nx_name"].format(bid=bid)
                    self.forward_name_map[gguf_name] = q4nx_name
                    self.backward_name_map[q4nx_name] = gguf_name
                    if "default_tensor_type" in param_info:
                        self.tensor_q4nx_type_map[gguf_name] = self.get_ggml_type(param_info["default_tensor_type"])
                    else:
                        self.tensor_q4nx_type_map[gguf_name] = self.default_tensor_type
                    if bid == 0:
                        print(f"\tConverted {gguf_name} to {q4nx_name}")
            else:
                self.forward_name_map[gguf_template] = param_info["q4nx_name"]
                self.backward_name_map[param_info["q4nx_name"]] = gguf_template
                if "default_tensor_type" in param_info:
                    self.tensor_q4nx_type_map[gguf_template] = self.get_ggml_type(param_info["default_tensor_type"])
                else:
                    self.tensor_q4nx_type_map[gguf_template] = self.default_tensor_type
                print(f"\tConverted {gguf_template} to {param_info['q4nx_name']}")

        # sort the name map by the name alphabetically
        self.forward_name_map = dict(sorted(self.forward_name_map.items(), key=lambda item: item[0]))
        self.backward_name_map = dict(sorted(self.backward_name_map.items(), key=lambda item: item[0]))
        self.tensor_q4nx_type_map = dict(sorted(self.tensor_q4nx_type_map.items(), key=lambda item: item[0]))


    def _has_lm_head(self) -> bool:
        for key in self.gguf_tensors.keys():
            if "lm_head.weight" in key or key == "output.weight":
                return True 
        return False

    # ---- Qwen3.5-family layer bookkeeping -----------------------------------
    def _is_full_attention_layer(self, layer_id: int) -> bool:
        """Full (gated softmax) attention block, as opposed to gated delta-net.

        Only a full-attention block has `attn_q.weight`.  Asking the GGUF is
        right for every layout; "every Nth block" is not (d3:edge is a 0.8B-size
        model with full attention at 2, 5, 8, 9, 10, 11).
        """
        return f"blk.{layer_id}.attn_q.weight" in self.gguf_tensors

    def _drop_mtp_tensors(self):
        """Remove the next-token-prediction (MTP) block from `gguf_tensors`.

        Newer llama.cpp exports append it as block `block_count - 1` and count it
        in `block_count` (`nextn_predict_layers` = 1).  It is a full-attention
        block with extra `nextn.*` tensors, so left in it would be written as a
        bogus extra decoder layer `model.layers.<block_count - 1>` (and, for the
        modulo layer rule, converted as a delta-net layer).  Converters that use
        the MTP head keep it themselves (see qwen38.py) and do not call this.
        """
        n_mtp = 0
        for name, field in self.gguf_reader.fields.items():
            if name.endswith(".nextn_predict_layers"):
                n_mtp = int(field.contents())
        if n_mtp <= 0:
            return
        first = self.num_layers - n_mtp
        drop = [k for k, t in self.gguf_tensors.items()
                if (m := re.match(r"^blk\.(\d+)\.", t.name)) and int(m.group(1)) >= first]
        for k in drop:
            del self.gguf_tensors[k]
        print(f"[INFO] Dropped {len(drop)} tensors of the MTP block (blk.{first}..): "
              f"this model has no use for them, {first} decoder layers remain")
        self.num_layers = first

    def _export_q4nx_tensors(self, q4nx_path: str):
        print(f"[INFO] Saving Q4NX tensors to {q4nx_path}/model.q4nx...")
        create_dir_if_not_exists(q4nx_path)
        save_file(self.q4nx_tensors, os.path.join(q4nx_path, "model.q4nx"))

    def _pack_MXFP4_q4nx(self, scales:torch.Tensor, data:torch.Tensor,    
            )->torch.Tensor:
    
        MXFP4_BLOCK_SIZE= 32
        MXFP4_BLOCK_SIZE_data_in_byte = 16 # because 4 bit data, so 32 data is 16 byte
        

        
        #TODO: for now, only support safetensor with multiple expert, aka expect  scalse to be shape of [num_expert/batch, rows, cols]
        # expect data to be shape of [num_expert/batch, rows, cols_in_byte/MXFP4_BLOCK_SIZE_data_in_byte, MXFP4_BLOCK_SIZE_data_in_byte ]        
        assert len(scales.shape) == 3
        assert len(data.shape) ==4
        assert scales.shape[0] == data.shape[0] and scales.shape[1] == data.shape[1] and scales.shape[2] == data.shape[2]
        assert data.shape[3] == MXFP4_BLOCK_SIZE_data_in_byte
        
        
        if scales.shape[1] % self.row_block_size !=0:
            padd_size = (self.row_block_size - scales.shape[1]) % self.row_block_size
            scales = F.pad(scales, (0, 0, 0, padd_size), "constant", 0)
            data = F.pad(data, (0, 0, 0,0, 0, self.row_block_size - data.shape[-3] % self.row_block_size), "constant", 0)
            
        if (scales.shape[2] * MXFP4_BLOCK_SIZE) % self.col_block_size !=0:
            addition_padd_size = (self.col_block_size - ((scales.shape[2] * MXFP4_BLOCK_SIZE) % self.col_block_size)) // MXFP4_BLOCK_SIZE

            scales: torch.Tensor = F.pad(scales, (0, addition_padd_size, 0, 0), "constant", 0)
            data = F.pad(data, (0, 0, 0, addition_padd_size, 0, 0), "constant", 0)
        


        # # calcuate dimension for scales and biases

        scales= scales.contiguous()
        data = data.contiguous()
  
        

        row_div_q4_row = scales.shape[1] // self.row_block_size
        col_div_q4_col = scales.shape[2] // (self.col_block_size // MXFP4_BLOCK_SIZE)
        scales = rearrange(
            scales, 
            "batch (row_div_q4_row q4_row) (col_div_q4_col q4_col_div32) -> batch row_div_q4_row col_div_q4_col q4_row q4_col_div32", 
            row_div_q4_row=row_div_q4_row,  
            col_div_q4_col=col_div_q4_col,
            q4_row=self.row_block_size, 
            q4_col_div32=self.col_block_size//MXFP4_BLOCK_SIZE
        ).contiguous()            
        


        # combine the block dim
        # Get the dimensions *before* the last two and pass the full shape to view()
        data: torch.Tensor = data.reshape(*data.shape[:-2], -1)
        assert len(data.shape) == 3
        data_row_div = data.shape[1] // self.row_block_size
        data_col_div = data.shape[2] // (self.col_block_size // 2)
        data = rearrange(
            data,
            "batch (row_div_q4_row q4_row) (col_div_q4_col q4_col) -> batch row_div_q4_row col_div_q4_col q4_row q4_col",
            row_div_q4_row=data_row_div,
            col_div_q4_col=data_col_div,
            q4_row=self.row_block_size,
            q4_col=(self.col_block_size // 2)
        ).contiguous()

        # at this step, both scales and data are 
            # 1. row major within the blocks
            # 2. Also row major in block level


        assert self.row_block_size % self.parallel_size == 0
        
        # Now, rearrange data to column major order with stride of 16
        data = rearrange(
            data,
            "batch row_div col_div (q4_row_div_col_stride col_stride) (q4_col one) -> \
            batch row_div col_div (q4_row_div_col_stride q4_col) (col_stride one)",
            col_stride = self.parallel_size, # 16 element, since each data is half-byte
            one = 1,
        ).contiguous()
        
        # at this point, the data is shape of 
        # the data block is col-major order with col_stride
        #[batch, row_div, col_div (q4_row_div_col_stride x q4_col )  , col_stride]
        
        # however, for each col_stride(normally 16) the data is in uint8_t
        # thus, it actually mean the last_dim, col_stride is 2 column of 16
        """
        
        EX: if col_stride=16 of uint8_t
        [
            a0,
            a1,
            a2,
            a3,
            a4
            ...
        ]
        
        But in int4, this actuall represnt
        [
            a0_0, a0_1,
            a1_0, a1_1,
            a2_0, a2_1,
            ......
            
        ]
        where a0_0, a0_1 share a common scale
        Thus, this mean the AIE kernel need to do a even_odd filter, as show in code 
        https://github.com/ngdxzy/FastFlowLM_Dev/blob/30b43b59d77f5759e943cea52ff7a259ca0fa776/npu_framework/gpt_npu_bin/kernel/mvm_MXFP4.hpp#L76
        """
        
        # Thus, in this code, let us do the even odd filter for it
        """
        Thus, we reorder the col_stride from
        [
            a0_0, a0_1,a1_0, a1_1, a2_0, a2_1,
            ......
        ]
        to 
        [
            a0_0, a1_0, a2_0, .... a0_1, a1_1
            
        ]
        """
        
        # data shape: [..., col_stride] (e.g., [..., 16])
        # Assumes col_stride is an even number (like 16)

        # 1. Extract low and high nibble streams
        # low_nibbles shape: [..., 16], content: [a0_0, a1_0, a2_0, ..., a15_0]
        low_nibbles = data & 0x0F
        # high_nibbles shape: [..., 16], content: [a0_1, a1_1, a2_1, ..., a15_1]
        high_nibbles = (data >> 4) & 0x0F

        # 2. Pack the low_nibbles stream
        # [a0_0, a1_0, a2_0, ...] -> [ (a0_0 | a1_0<<4), (a2_0 | a3_0<<4), ... ]
        # Slicing [..., 0::2] gets evens: [a0_0, a2_0, ..., a14_0]
        # Slicing [..., 1::2] gets odds: [a1_0, a3_0, ..., a15_0]
        low_part_packed = (low_nibbles[..., 0::2] & 0x0F) | ((low_nibbles[..., 1::2] & 0x0F) << 4)
        # low_part_packed shape: [..., 8]

        # 3. Pack the high_nibbles stream
        # [a0_1, a1_1, a2_1, ...] -> [ (a0_1 | a1_1<<4), (a2_1 | a3_1<<4), ... ]
        high_part_packed = (high_nibbles[..., 0::2] & 0x0F) | ((high_nibbles[..., 1::2] & 0x0F) << 4)
        # high_part_packed shape: [..., 8]

        # 4. Concatenate the two 8-byte halves to form the new 16-byte col_stride
        # Final shape: [..., 16]
        data_reordered = torch.cat([low_part_packed, high_part_packed], dim=-1)
        data = data_reordered.contiguous()
        # NOW, this code change is reflected in https://github.com/ngdxzy/FastFlowLM_Dev/commit/028680d1f670d817fae0e7efe947bb1a4c19c8a3
        
        
        
        # but for scale, simply change to column major order, NOTE: the stride of 16 does not apply to scale here 
        # aka, scales remain to be  self.row_block_size x (self.col_block_size//MXFP4_BLOCK_SIZE) but in colum major order  
        scales = rearrange(
            scales,
            "batch row_div col_div q4_row q4_col -> \
                     batch row_div col_div q4_col q4_row"
        ).contiguous()
        
   


        # # Apply the padding.
        # scales_expanded = F.pad(scales, paddings, "constant", 0)
        scales = scales.reshape(*scales.shape[:-2], -1).contiguous()
        padding_amount = scales.shape[-1] * 3  # padding to align with q4nx requirement. Because in q41, 2byte for scale and 2byte for bias
                                                    # given mxfp4 there is only 1 byte for scale, so need to pad 3 byte
        paddings = (0, padding_amount)
        scales = F.pad(scales, paddings, "constant", 0).contiguous()


        
        # at this point, 
        # data should be shape of [batch, row_div col_div, q4_row, q4_col]
        # scales should be shape of [batch, row_div, col_div, (self.col_block_size//MXFP4_BLOCK_SIZE), q4_row*4]
        
        # now, given they are both uint8, first change data and scales shape by combine the last two dim
        # data now be [batch, row_div, col_div, q4_row*q4_col]
        #scale shoule be shape of [batch, row_div, col_div, (self.col_block_size//MXFP4_BLOCK_SIZE) * q4_row*4 ]

        data = data.reshape(*data.shape[:-2], -1).contiguous()
        # scales = scales_expanded.reshape(*scales_expanded.shape[:-2], -1).contiguous()
        
        
        merged = torch.cat([scales, data], dim=-1).contiguous()
        return merged
        
    
  
    # def force_pack_q8_to_q4nx_size(self, tensor_data:GGUFTensor):
        
    #         # TODO: FIXME: DEBUG force convert Q80
    #         w = dequantize(tensor_data.data, tensor_data.tensor_type)
    #         w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
    #         print(w)
    #         w = w.to(dtype=torch.float32).numpy()
            
    #         data_q80 = quantize(w, GGMLQuantizationType.Q8_0).copy()
    #         # create a new GGUF type
    #         data_q80_gguf = GGUFTensor("data_q80", data_q80.shape, data_q80, GGMLQuantizationType.Q8_0)
                        
    #         unpacked =data_q80_gguf.unpack(self.default_tensor_type)

            
    #         # override to turn keep_block_in_2D = True
    #         old_keep_block_in_2D =  self.keep_block_in_2D
    #         self.keep_block_in_2D = True
    #         val = self._pack_q4nx_8b(*unpacked)
    #         self.keep_block_in_2D = old_keep_block_in_2D
            
    #         return val
    #         # scales, data = GGUFTensor.unpack_q8_0( data_q80)
            
    #         # m_tmp = scales.clone()
            
    #         # # override 

    #         # col_block_size_old = self.col_block_size
    #         # keep_block_in_2D_old = self.keep_block_in_2D
            
    #         # cur_q4nx_block_byte_size = int(( self.row_block_size* col_block_size_old   )*(5/8) )
            
    #         # if col_block_size_old == 256:
    #         #     self.col_block_size= 128
    #         # else:
    #         #     #TODO:
    #         #     raise ValueError("Undefine case for now")
    #         # self.keep_block_in_2D= True
    #         # q8nx_pack_result  = self._pack_q8nx(
    #         #     scales, m_tmp, data
    #         # )
            
    #         # # now, we want to padd the last dimension to be size of 5120(for q4nx)
    #         # # Pad the last dimension from 4608 to 5120

            
    #         # padding_size = cur_q4nx_block_byte_size    - q8nx_pack_result.shape[-1]
    #         # q8nx_pack_result = F.pad(q8nx_pack_result, (0, padding_size))
            
    #         # self.keep_block_in_2D = keep_block_in_2D_old
    #         # self.col_block_size = col_block_size_old
    #         # return q8nx_pack_result
            
            
    def _pack(self, *args, tensor_type: GGMLQuantizationType = None) -> torch.Tensor:
        #TODO: do not support q81

        d, m, qw = (list(args) + [None, None])[:3]

        if tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                           GGMLQuantizationType.BF16):
            # Norms, ssm_conv1d and friends ship unquantized and stay that way:
            # there is nothing to pack, the tensor just goes out as bf16.
            return d.to(torch.bfloat16)
        elif tensor_type == GGMLQuantizationType.Q4_K:
            # Same shapes as Q4_1, different semantics: d/m are the effective
            # per-group scale t_j and *subtracted* min u_j, and the super-block
            # re-fit happens inside _pack_q4k.  Dispatch is by type, not by arity.
            return self._pack_q4k(d, m, qw)
        elif tensor_type == GGMLQuantizationType.Q8_0:
            return self._pack_q8nx(d, None, qw )
        elif tensor_type == GGMLQuantizationType.Q8_1:
            return self._pack_q8nx(d, m, qw )
        elif tensor_type == GGMLQuantizationType.Q4_0:
            return self._pack_q4nx(d, None, qw)
        elif tensor_type == GGMLQuantizationType.Q4_1:
            return self._pack_q4nx(d, m, qw)            
        else:
            print(f"Warning: Packing unsupported type {tensor_type} in q4_1 with force conversion" )
            return self._pack_q4nx(d, m, qw)
    
    # def _pack_q4nx_8b(self,  d: torch.Tensor,m: torch.Tensor, qw:torch.Tensor) -> torch.Tensor:
    #     #note, support q80 for now
    #     # d for scale
    #     # m for min
        
    #     # TODO: NOTE:
    #     col_block_size_old = self.col_block_size
    #     keep_block_in_2D_old = self.keep_block_in_2D        
    #     if self.col_block_size == 256:
    #         # force to 128 
    #         self.col_block_size= 128            
    #     else:
    #         raise ValueError("Undefine case for now")
    #     self.keep_block_in_2D= True
        
    #     cur_q4nx_block_byte_size = int(( self.row_block_size* col_block_size_old   )*(5/8) )
                    
    #     q8nx_pack_result = self._pack_q8nx(data=qw, scales=d, m = m )

    #     padding_size = cur_q4nx_block_byte_size    - q8nx_pack_result.shape[-1]
    #     q8nx_pack_result = F.pad(q8nx_pack_result, (0, padding_size))
        
    #     self.keep_block_in_2D = keep_block_in_2D_old
    #     self.col_block_size = col_block_size_old
    #     return q8nx_pack_result
        
    def _pack_q8nx(self,   d: torch.Tensor, m: torch.Tensor = None, qw: torch.Tensor = None) -> torch.Tensor:
        scales = d

        data=qw
        #note, support q80 for now
        """ Q8NX format similar to Q4NX

        Q8NX format:
            - chunk shape: row_block_size x col_block_size
            - scale shape: (col_block_size // Q8_group_size) x row_block_size
            - m shape: (col_block_size // Q8_group_size) x row_block_size
            - quant shape: (row_block_size // parallel) x col_block_size x (parallel) #NOTE: num_int8_in_byte is 1 for Q8
        
        
        Parameters
        ----------
        scales : torch.Tensor
            _description_
        data : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_
            
        Notes
        -------
        Only support Q8_0 for now
        """
        
        
        Q8_group_size =  32
        # assert len(scales.shape) == 3
        # assert len(data.shape) ==3
        # data are shape of [row, col//Q8_group_size, Q8_group_size]
        # merge last two dim of scales and data
        
        if scales.shape[-1] == 1:
            scales = scales.reshape(*scales.shape[:-2], -1).contiguous()
        if data.shape[-1] == 1:
            data = data.reshape(*data.shape[:-2], -1).contiguous()
        if m is not None and  m.shape[-1] == 1: 
            m = m.reshape(*m.shape[:-2], -1).contiguous()
        else:
            scales = scales.contiguous()
            data = data.contiguous()
            if m is not None:
                m = m.contiguous() 
            
        rows, cols = data.shape[0], data.shape[1]
        
        

        
        if cols % self.col_block_size != 0:
            cols_padded = round_up_to_multiple(cols, self.col_block_size)
            
            scale_pad_amount = (cols_padded -cols) // Q8_group_size
            data_pad_amount = cols_padded - cols
            
            scales = F.pad(scales, (0, scale_pad_amount), "constant", 0)
            if m is not None:
                m = F.pad(m, (0, scale_pad_amount), "constant", 0)
            data = F.pad(data, (0, data_pad_amount), "constant", 0)
        

        
        if self.keep_block_in_2D:
            
            # does two things
            # 1. split scales to logical block of (row_div_r, col_div_c, r, c) but in row major order
            # 2. change the block of (r,c) to column major order as (c,r)
            scales = rearrange(
                scales,
                "(row_div_r r) (col_div_c c) -> row_div_r col_div_c (c r)",
                r=self.row_block_size,
                c=self.col_block_size // Q8_group_size

            ).contiguous()
            
            if m is not None:
                m = rearrange(
                    m,
                    "(row_div_r r) (col_div_c c) -> row_div_r col_div_c (c r)",
                    r=self.row_block_size,
                    c=self.col_block_size // Q8_group_size                
                ).contiguous()
                
            # The kernel keeps the whole row span of a block resident at once
            # (PARALLEL_ROWS == QXNX_ROW_BLOCK_SIZE == row_block_size), so one
            # column is row_block_size contiguous int8 and columns simply ascend:
            # a block is plain column major, with no parallel-strided re-order.
            data = rearrange(
                data,
                "(row_div_r r) (col_div_c c) -> row_div_r col_div_c (c r)",
                r=self.row_block_size,
                c=self.col_block_size
            ).contiguous()

            # now, convert scales from float16 to bfloat16
            assert(scales.dtype == torch.float16)
            scales = scales.to(torch.bfloat16)
            
            # also convert m from float16 to bfloat16
            if m is not None:
                assert(m.dtype == torch.float16)
                m = m.to(torch.bfloat16)
            
            
        else:
            raise ValueError("Only support keep_block_in_2D for now")
        
        scales = scales.view(torch.int8)
        if m is not None:
            m = m.view(torch.int8)
        data = data.view(torch.int8)
        
        scales_np = scales.numpy()

        data_np = data.numpy()
        
        if m is not None:
            m_np = m.numpy()            
            merged = np.concatenate([scales_np,  m_np, data_np], axis = -1).copy()
        else:
            # We pack to q8nx, if there is no bias, we don't pack bias into it
            merged = np.concatenate([scales_np,  data_np], axis = -1).copy()
        return torch.from_numpy(merged)       
    
    def _pack_q4nx(self, *args) -> torch.Tensor:
        """


        Q4NX format:
            - chunk shape: row_block_size x col_block_size
            - scale shape: (col_block_size // Q4_group_size) x row_block_size
            - min shape:   (col_block_size // Q4_group_size) x row_block_size
            - quant shape: col_block_size x (row_block_size // NUM_int4_in_byte)

        Therefore:
            - d: (rows, cols // Q4_group_size) -> (rows // row_block_size, cols // (col_block_size * Q4_group_size), col_block_size // Q4_group_size, Q4_group_size)
            - m: (rows, cols // Q4_group_size) -> (rows // row_block_size, cols // (col_block_size * Q4_group_size), col_block_size // Q4_group_size, Q4_group_size)

        Q4_0 / Q4_1 only -- Q4_K carries the same shapes but different semantics
        and is dispatched to _pack_q4k by tensor type in _pack.
        """
        d, m, qw = (list(args) + [None, None])[:3]

        Q4_group_size =  32
        NUM_int4_in_byte = 2
        d = d.contiguous()
        m = m.contiguous() if m is not None else None
        qw = qw.contiguous() if qw is not None else None
        
        if m is None: # not packed, could be a float or bf16 tensor
            return d.to(torch.bfloat16)

        rows, cols = qw.shape

        if cols%self.col_block_size != 0:
            cols_padded = round_up_to_multiple(cols, self.col_block_size)
            
            d_m_pad_amount = (cols_padded -cols) // Q4_group_size
            qw_pad_amount = cols_padded-cols
            
            d = F.pad(d, (0, d_m_pad_amount), "constant", 0)
            m = F.pad(m, (0, d_m_pad_amount), "constant", 0)
            qw = F.pad(qw, (0, qw_pad_amount), "constant", 0)
            
            
        if not self.keep_block_in_2D:
            # chunk wise
            d = rearrange(d, '(p r) (q c) -> (p q) (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            m = rearrange(m, '(p r) (q c) -> (p q) (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            # dm done

            qw = rearrange(qw, '(p r) (q c) -> (p q) r c', r = self.row_block_size, c = self.col_block_size)
            qw = rearrange(qw, 'n (r b) c -> n c r b', b = NUM_int4_in_byte).contiguous().to(torch.int8)

            qw[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(qw[..., 1], 4), 0xF0)
            qw[..., 0] = torch.bitwise_or(torch.bitwise_and(qw[..., 0], 0x0F), qw[..., 1])
            qw = qw[..., 0].contiguous()
            qw = rearrange(qw, 'n c r -> n (c r)').contiguous()
        else:
            # chunk wise
            d = rearrange(d, '(p r) (q c) -> p q (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            m = rearrange(m, '(p r) (q c) -> p q (c r)', r = self.row_block_size, c = self.col_block_size // Q4_group_size).contiguous()
            # dm done

            qw = rearrange(qw, '(p r) (q c) -> p q r c', r = self.row_block_size, c = self.col_block_size)
            qw = rearrange(qw, 'p q (r b) c -> p q c r b', b = NUM_int4_in_byte).contiguous().to(torch.int8)

            qw[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(qw[..., 1], 4), 0xF0)
            qw[..., 0] = torch.bitwise_or(torch.bitwise_and(qw[..., 0], 0x0F), qw[..., 1])
            qw = qw[..., 0].contiguous()
            qw = rearrange(qw, 'p q c r -> p q (c r)').contiguous()
            
            
            
        d = d.to(torch.bfloat16).view(torch.int8)
        d = d.numpy()        
        if m is not None:
            m = m.to(torch.bfloat16).view(torch.int8)
            m = m.numpy()       
                 
        qw = qw.view(torch.int8)
        qw = qw.numpy()
        if m is not None:
            merged = np.concatenate([d, m, qw], axis = -1).copy()
        else:
            merged = np.concatenate([d, qw], axis = -1).copy()

        merged = torch.from_numpy(merged)
        return merged

    def _pack_q4k(self, t: torch.Tensor, u: torch.Tensor, q: torch.Tensor,
                  search: int = 3) -> torch.Tensor:
        """Pack the FLM q4_k format (uint8 s'/m' per group + bf16 S'/M' per super-block).

        Input is what GGUFTensor.unpack_q4_k returns, *after* any model-specific
        reorder: the effective per-group scale t_j and subtracted min u_j in exact
        float32, shape (rows, cols // 32), plus the uint4 quants (rows, cols).  The
        (BF16 P', uint8 p'_j) re-fit of quant.md is done here, per 256-column
        super-block, so it always fits the 8 groups that really share a super-block
        in the packed output -- a reorder that shuffles columns at a granularity
        finer than 256 (ssm_out_proj moves 128-column chunks) would otherwise leave
        the super-block metadata unrepresentable.

        Mirrors `q4k_block_t` in the decoding kernels' model_spec.h, one struct
        per row_block_size x col_block_size chunk, everything column major over
        the chunk (the whole row span is resident at once, so there is no
        parallel-strided re-order):

            uint8 scales[col_block_size // Q4_group_size][row_block_size]
            uint8 mins  [col_block_size // Q4_group_size][row_block_size]
            uint4 qs    [col_block_size][row_block_size // NUM_int4_in_byte]
            bf16  S     [col_block_size // Q4K_super_block_size][row_block_size]
            bf16  M     [col_block_size // Q4K_super_block_size][row_block_size]

        With the shipping 32x256 chunk that is 256 + 256 + 4096 + 64 + 64 = 4736 B,
        i.e. 4.625 bits per weight, matching get_quantization_byte_size().

        GGUF's Q4_K subtracts the min (w = S' s'_j q - M' m'_j) while the kernel
        adds both accumulators, so M' is stored negated here.
        """
        Q4_group_size = 32
        Q4K_super_block_size = 256
        NUM_int4_in_byte = 2

        t = t.to(torch.float32).contiguous()
        u = u.to(torch.float32).contiguous()
        q = q.contiguous()

        cols = q.shape[-1]
        assert cols % Q4K_super_block_size == 0, "Q4_K needs a multiple of 256 columns"
        assert self.col_block_size % Q4K_super_block_size == 0

        if cols % self.col_block_size != 0:
            cols_padded = round_up_to_multiple(cols, self.col_block_size)
            t = F.pad(t, (0, (cols_padded - cols) // Q4_group_size), "constant", 0)
            u = F.pad(u, (0, (cols_padded - cols) // Q4_group_size), "constant", 0)
            q = F.pad(q, (0, cols_padded - cols), "constant", 0)

        # Re-fit each side onto (BF16 super-block value, uint8 per-group value).
        groups_per_super = Q4K_super_block_size // Q4_group_size
        S, s8 = _refit_one_side(t.numpy().reshape(-1, groups_per_super), search=search)
        M, m8 = _refit_one_side(u.numpy().reshape(-1, groups_per_super), search=search)
        S = torch.from_numpy(S).view(t.shape[0], -1)
        M = torch.from_numpy(M).view(u.shape[0], -1)
        s8 = torch.from_numpy(s8).view_as(t)
        m8 = torch.from_numpy(m8).view_as(u)

        # scales / mins: one group of 32 columns contributes row_block_size uint8,
        # groups ascending -- the same block layout the bf16 scales use in q4nx.
        s8 = rearrange(s8, '(p r) (u c) -> p u (c r)', r=self.row_block_size,
                       c=self.col_block_size // Q4_group_size).contiguous()
        m8 = rearrange(m8, '(p r) (u c) -> p u (c r)', r=self.row_block_size,
                       c=self.col_block_size // Q4_group_size).contiguous()
        # S / M: one entry per row per super-block; a 256-column chunk holds exactly one.
        S = rearrange(S, '(p r) (u c) -> p u (c r)', r=self.row_block_size,
                      c=self.col_block_size // Q4K_super_block_size).contiguous()
        M = rearrange(M, '(p r) (u c) -> p u (c r)', r=self.row_block_size,
                      c=self.col_block_size // Q4K_super_block_size).contiguous()

        # quants: one column = row_block_size nibbles, rows paired into a byte with
        # the even row in the low nibble, columns ascending.
        q = rearrange(q, '(p r) (u c) -> p u r c', r=self.row_block_size, c=self.col_block_size)
        q = rearrange(q, 'p u (r b) c -> p u c r b', b=NUM_int4_in_byte).contiguous().to(torch.int8)
        q[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(q[..., 1], 4), 0xF0)
        q[..., 0] = torch.bitwise_or(torch.bitwise_and(q[..., 0], 0x0F), q[..., 1])
        q = rearrange(q[..., 0].contiguous(), 'p u c r -> p u (c r)').contiguous()

        s8 = s8.to(torch.uint8).view(torch.int8).numpy()
        m8 = m8.to(torch.uint8).view(torch.int8).numpy()
        S = S.to(torch.bfloat16).view(torch.int8).numpy()
        M = (-M).to(torch.bfloat16).view(torch.int8).numpy()   # kernel adds the min path
        q = q.view(torch.int8).numpy()

        merged = torch.from_numpy(np.concatenate([s8, m8, q, S, M], axis=-1).copy())
        if not self.keep_block_in_2D:
            merged = merged.reshape(-1, merged.shape[-1])
        return merged

    def _multi_modal_mm_weight_rearrange(self, weight: torch.Tensor,
                                            kernel_mm_K:int|None, kernel_mm_N:int|None
                                            ) -> torch.Tensor:
        
        assert len(weight.shape) ==2
        assert kernel_mm_K is not None and kernel_mm_N is not None, "kernel_mm_K and kernel_mm_N must be set for vision model weight rearrangement"
        MM_N_K_padding = max(kernel_mm_N, kernel_mm_K)
        origin_N, origin_K = weight.shape
        

        pad_N = (MM_N_K_padding - origin_N % MM_N_K_padding) % MM_N_K_padding
        pad_K = (MM_N_K_padding - origin_K % MM_N_K_padding) % MM_N_K_padding
        
        if pad_N >0 or pad_K >0:
            weight = F.pad(weight, (0, pad_K, 0, pad_N), "constant", 0)
        weight = rearrange( weight, 
                        "(N mm_n) (K mm_k) -> N K (mm_n mm_k)",
                        mm_k = kernel_mm_K, mm_n = kernel_mm_N).contiguous()

        return weight
    
    def vision_mm_weight_rearrange(self, weight: torch.Tensor) -> torch.Tensor:
        """Rearrange the vision MM weights


        In vision MM, the weights matrix is used as the B matrix in A@B operation.    
        Given a weight tensor of shape (out_features, in_features) correspond to (N, K)
        
        The function assume input weights is in row-major of (N, K),
        which should view as a K x N matrix in col-major.
        
        For efficent memory access, the function reorder the matrix to be in block of (N//MM_N, K//MM_K)( MM_N, MM_K) where each
        block is in row-major order and the blocks are also in row-major order.
        
        Then data can also be viewed as in col-major order with block of (K//MM_K, N//MM_N) within each (MM_K, MM_N) block in col-major order.

        For simplicity, the row, column dimension of the matrix is padded to be multiple of max(MM_N, MM_K). 
        Because in MLP operations, the output dim of up/gate projection needs to match the input dim of down projection.
        In other words, the N dim of up/gate projection equals to the K dim of down projection.
    
    
        Parameters
        ----------
        weight : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            Reorder vision weights 

        Notes
        ----------
        mm_n and mm_k are the data block size that each CT can process in one time.
        
        """
        return self._multi_modal_mm_weight_rearrange(weight=weight, kernel_mm_K= self.vision_MM_K, kernel_mm_N= self.vision_MM_N)
        
    def audio_mm_weight_rearrange(self, weight: torch.Tensor) -> torch.Tensor:
        return self._multi_modal_mm_weight_rearrange(weight=weight, kernel_mm_K= self.audio_MM_K, kernel_mm_N= self.audio_MM_N)
    def _extract_tokenizer_json(self, output_folder: str) -> dict:
        """
        Extract tokenizer information from GGUF file and convert to HuggingFace tokenizer.json format.
        
        Args:
            output_folder: Path to save the tokenizer.json file
            
        Returns:
            Dictionary containing tokenizer configuration
        """
        print("[INFO] Extracting tokenizer JSON...")
        tokenizer_json_path = os.path.join(output_folder, "tokenizer.json")
        
        # Extract tokenizer metadata from GGUF
        tokenizer_model = None
        tokens = []
        scores = []
        token_types = []
        merges = []
        bos_token_id = None
        eos_token_id = None
        unk_token_id = None
        pad_token_id = None
        
        for field in self.gguf_reader.fields.values():
            # Tokenizer model type
            if field.name == "tokenizer.ggml.model":
                tokenizer_model = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
            
            # Token list
            elif field.name == "tokenizer.ggml.tokens":
                tokens = [str(field.parts[i], encoding='utf-8') for i in field.data]
            
            # Token scores (for SentencePiece)
            elif field.name == "tokenizer.ggml.scores":
                scores = field.data.tolist() if hasattr(field.data, 'tolist') else list(field.data)
            
            # Token types
            elif field.name == "tokenizer.ggml.token_type":
                token_types = field.data.tolist() if hasattr(field.data, 'tolist') else list(field.data)
            
            # BPE merges
            elif field.name == "tokenizer.ggml.merges":
                merges = [str(field.parts[i], encoding='utf-8') for i in field.data]
            
            # Special token IDs
            elif field.name == "tokenizer.ggml.bos_token_id":
                bos_token_id = int(field.contents())
                print(f"[INFO] BOSS token ID: {bos_token_id}")
            elif field.name == "tokenizer.ggml.eos_token_id":
                eos_token_id = int(field.contents())
                print(f"[INFO] EOS token ID: {eos_token_id}")
            elif field.name == "tokenizer.ggml.unknown_token_id":
                unk_token_id = int(field.contents())
                print(f"[INFO] Unknown token ID: {unk_token_id}")
            elif field.name == "tokenizer.ggml.padding_token_id":
                pad_token_id = int(field.contents())
                print(f"[INFO] Padding token ID: {pad_token_id}")
        
        # Build vocab dictionary for HuggingFace format
        vocab = {}
        for idx, token in enumerate(tokens):
            vocab[token] = idx
        
        # Determine tokenizer type and create appropriate HuggingFace config
        if tokenizer_model == "llama" or tokenizer_model == "gpt2":
            # BPE tokenizer
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": self._build_added_tokens(tokens, bos_token_id, eos_token_id, unk_token_id, pad_token_id),
                "normalizer": None,
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "post_processor": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True
                },
                "decoder": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True
                },
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": tokens[unk_token_id] if unk_token_id is not None and unk_token_id < len(tokens) else None,
                    "continuing_subword_prefix": "",
                    "end_of_word_suffix": "",
                    "fuse_unk": False,
                    "byte_fallback": False,
                    "vocab": vocab,
                    "merges": merges
                }
            }
        else:
            # Default to SentencePiece-like tokenizer (for llama, etc.)
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": self._build_added_tokens(tokens, bos_token_id, eos_token_id, unk_token_id, pad_token_id),
                "normalizer": {
                    "type": "Sequence",
                    "normalizers": []
                },
                "pre_tokenizer": {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "add_prefix_space": True,
                    "prepend_scheme": "first"
                },
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [
                        {
                            "SpecialToken": {
                                "id": tokens[bos_token_id] if bos_token_id is not None and bos_token_id < len(tokens) else "<s>",
                                "type_id": 0
                            }
                        },
                        {"Sequence": {"id": "A", "type_id": 0}}
                    ],
                    "pair": [
                        {
                            "SpecialToken": {
                                "id": tokens[bos_token_id] if bos_token_id is not None and bos_token_id < len(tokens) else "<s>",
                                "type_id": 0
                            }
                        },
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {
                            "SpecialToken": {
                                "id": tokens[eos_token_id] if eos_token_id is not None and eos_token_id < len(tokens) else "</s>",
                                "type_id": 0
                            }
                        },
                        {"Sequence": {"id": "B", "type_id": 1}}
                    ],
                    "special_tokens": {}
                },
                "decoder": {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "add_prefix_space": True,
                    "prepend_scheme": "first"
                },
                "model": {
                    "type": "Unigram",
                    "unk_id": unk_token_id,
                    "vocab": [[token, score] for token, score in zip(tokens, scores)] if scores else [[token, 0.0] for token in tokens]
                }
            }
        
        # Save tokenizer.json
        create_dir_if_not_exists(output_folder)
        with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Tokenizer saved to {tokenizer_json_path}")
        return tokenizer_json
    
    def _build_added_tokens(self, tokens: list, bos_id: int, eos_id: int, unk_id: int, pad_id: int) -> list:
        """
        Build the added_tokens list for HuggingFace tokenizer format.
        
        Args:
            tokens: List of all tokens
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID
            unk_id: Unknown token ID
            pad_id: Padding token ID
            
        Returns:
            List of added token dictionaries
        """
        added_tokens = []
        special_token_ids = {
            bos_id: "bos_token",
            eos_id: "eos_token", 
            unk_id: "unk_token",
            pad_id: "pad_token"
        }
        
        for token_id, token_type in special_token_ids.items():
            if token_id is not None and token_id < len(tokens):
                added_tokens.append({
                    "id": token_id,
                    "content": tokens[token_id],
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                })
        
        return added_tokens


#: (decoder layers without the MTP block, hidden size) -> converter, for GGUFs whose
#: `general.architecture` is `qwen35` and whose name says Qwen3.x (3.5, 3.6, ...).
_QWEN35_GEOMETRY = {
    (24, 1024): ModelArch.QWEN35_08B,
    (24, 2048): ModelArch.QWEN35_2B,
    (32, 2560): ModelArch.QWEN35_4B,
    (32, 4096): ModelArch.QWEN35_9B,
}
if hasattr(ModelArch, "QWEN35_27B"):
    _QWEN35_GEOMETRY[(64, 5120)] = ModelArch.QWEN35_27B


def _resolve_qwen35(reader: GGUFReader, basename: str | None, name: str | None) -> ModelArch | None:
    """Pick the Qwen3.5-family converter for a stock llama.cpp `qwen35` GGUF.

    Stock exports name the model `general.basename = "Qwen3.5"` (size only in
    `general.size_label`), which the alias table cannot match by prefix except as the
    plain `qwen3` alias -- the wrong converter.  The size is in the geometry instead.
    Returns None for anything that does not look like a Qwen3.x export, so MiniCPM-V and
    other `qwen35`-architecture models keep resolving through their own names.
    """
    labels = [x.lower() for x in (basename, name) if x]
    if not any(x.startswith(("qwen3.", "qwen3_", "qwen3 ")) for x in labels):
        return None
    if any(x.startswith(("qwen3.8", "qwen3_8", "qwen3 8")) for x in labels):
        if hasattr(ModelArch, "QWEN38_27B"):
            return ModelArch.QWEN38_27B
        raise ValueError("Qwen3.8 is not supported by this converter version")
    if any("omni" in x for x in labels):
        return ModelArch.QWEN35_OMNI
    if "qwen35.block_count" in reader.fields:
        block_count = reader.fields["qwen35.block_count"].contents()
        mtp = reader.fields.get("qwen35.nextn_predict_layers")
        layers = block_count - (mtp.contents() if mtp is not None else 0)
        hidden = reader.fields["qwen35.embedding_length"].contents()
        arch = _QWEN35_GEOMETRY.get((layers, hidden))
    else:
        # the mmproj ("clip") GGUF has no text geometry; the projector output is the text hidden size
        # (`general.size_label` there is the tower's size, not the model's)
        layers, hidden = None, reader.fields["clip.vision.projection_dim"].contents()
        arch = next((a for (_, h), a in _QWEN35_GEOMETRY.items() if h == hidden), None)
    if arch is None:
        # falling through would hand a Qwen3.x GGUF to the plain Qwen3 converter and fail much later
        raise ValueError(f"Qwen3.x GGUF with {layers} layers / hidden {hidden} matches no known size "
                         f"({sorted(_QWEN35_GEOMETRY)}); pass -f <family>, e.g. -f qwen3.5-27B")
    print(f"[INFO] Detected {ModelArchNames[arch][0]} ({layers} layers, hidden {hidden})" if layers else
          f"[INFO] Detected {ModelArchNames[arch][0]} (mmproj, hidden {hidden})")
    return arch


def get_model_arch_from_gguf(reader: GGUFReader, override_model_arch:str="") -> ModelArch:
    """
    Read the model architecture from a GGUF file.
    
    Args:
        gguf_path: Path to the GGUF file
        override_model_arch: Non-empty string override this model architecture type
        
    Returns:
        ModelArch enum value
        
    Raises:
        ValueError: If the architecture is not recognized or supported
    """
    
    if override_model_arch != "":
        for arch_enum, arch_names in ModelArchNames.items():
            for arch_name in arch_names:            
                if override_model_arch.lower().startswith(arch_name.lower()):
                    return arch_enum
        print("Warning: Did not find matching override model arch, attempting to load base on gguf information")

    
    # Get architecture from GGUF metadata
    # The architecture is typically stored in the 'general.architecture' field
    arch_str:str|None = None
    basename_str:str|None = None
    name_str:str|None = None
    for field in reader.fields.values():
        if field.name == 'general.architecture':
            arch_str = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
        elif field.name == 'general.basename':
            basename_str = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
        elif field.name == 'general.name':
            name_str = str(field.parts[field.data[0]], encoding='utf-8') if field.data else None
    

    # Map the architecture string to ModelArch enum
    for arch_enum, arch_names in ModelArchNames.items():
        for arch_name in arch_names:
            if arch_str.lower() == arch_name.lower():
                return arch_enum
    

    is_qwen35_mmproj = (arch_str is not None and arch_str.lower() == "clip"
                        and "clip.vision.projection_dim" in reader.fields)
    if arch_str is not None and (arch_str.lower() == "qwen35" or is_qwen35_mmproj):
        resolved = _resolve_qwen35(reader, basename_str, name_str)
        if resolved is not None:
            return resolved

    for arch_enum, arch_names in ModelArchNames.items():
        for arch_name in arch_names:
            if basename_str is not None and basename_str.lower().startswith(arch_name.lower()):
                return arch_enum


    raise ValueError(f"Unsupported model architecture: {arch_str}")


def get_registered_models() -> Dict[ModelArch, Type['__Q4NX_Converter']]:
    """
    Get the dictionary of registered model converters.
    
    Returns:
        Dictionary mapping ModelArch to converter classes
    """
    return _MODEL_REGISTRY.copy()


def create_converter(gguf_path: str, override_model_arch:str) -> __Q4NX_Converter:
    """
    Factory function to create the appropriate converter based on the GGUF model architecture.
    
    Args:
        gguf_path: Path to the GGUF file
        
    Returns:
        An instance of the appropriate converter class
        
    Raises:
        ValueError: If the architecture is not recognized or no converter is available
        
    Example:
        >>> converter = create_converter("model.gguf")
        >>> converter.load_gguf("model.gguf")
        >>> converter.convert("configs")
    """
    # Read the architecture from the GGUF file
    reader = GGUFReader(gguf_path)
    model_arch = get_model_arch_from_gguf(reader,override_model_arch )

    
    # Look up the appropriate converter class
    if model_arch not in _MODEL_REGISTRY:
        available = [ModelArchNames.get(arch, str(arch)) for arch in _MODEL_REGISTRY.keys()]
        raise ValueError(
            f"No converter available for architecture: {ModelArchNames.get(model_arch, 'unknown')}. "
            f"Available converters: {available if available else 'None (no models imported?)'}"
        )
    
    converter_class = _MODEL_REGISTRY[model_arch]

    converter_instance = converter_class(reader)

    return converter_instance