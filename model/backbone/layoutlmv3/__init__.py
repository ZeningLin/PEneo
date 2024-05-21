"""
LayoutLMv3 backbone, official implementation by Microsoft Research Asia.

Codes in this folder are all copied from the official implementation of LayoutLMv3:

https://github.com/microsoft/unilm/tree/master/layoutlmv3

It is observed that this version has faster forward speed
than the Huggingface Transformers implementation. Therefore, we use the
official version for our model.

"""

from .configuration_layoutlmv3 import LayoutLMv3Config
from .modeling_layoutlmv3 import LayoutLMv3Model
from .processing_layoutlmv3 import LayoutLMv3Processor
from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer
from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast
