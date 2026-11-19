"""
Model converters for different architectures.
Import this module to automatically register all available model converters.
"""

from .qwen3vl import Qwen3VL
from .llama import Llama
from .lfm2 import LFM2
from .qwen3 import Qwen3
from .qwen35 import Qwen35, Qwen35_9B, Qwen35_2B, Qwen35_08B
from .qwen35_omni import Qwen35Omni
from .qwen2 import Qwen2
from .qwen2vl import Qwen2VL
from .gemma3 import Gemma3
from .gemma4 import Gemma4
from .phi4 import Phi4
from .gpt_oss import GPTOSS
from .nanbeige import Nanbeige
from .qwen35_moe import QWEN35_A3B
from .qwen38 import Qwen38_27B
__all__ = ['Qwen3VL', 'Llama', 'LFM2', 'Qwen3', 'Qwen2', 'Qwen2VL', 'Gemma3', 'Gemma4', 'Phi4', 'GPTOSS', 'Nanbeige', 'Qwen35', 'Qwen35_2B', 'Qwen35_9B', 'Qwen35_08B', 'Qwen35Omni', 'QWEN35_A3B', 'Qwen38_27B']