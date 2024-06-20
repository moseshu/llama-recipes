import os
model = os.getenv('FT_MODEL_TYPE')
model = str(model).lower()
print(f"model type is {model}")
if model=='qw':
    from llama_recipes.qwen.modeling_qwen import QWenBlock as LlamaDecoderLayer
    from llama_recipes.qwen.modeling_qwen import QWenLMHeadModel as LlamaForCausalLM
    from llama_recipes.qwen.tokenization_qwen import QWenTokenizer as LlamaTokenizer
    from llama_recipes.qwen.configuration_qwen import QWenConfig as LlamaConfig
elif model=='mixtral':
    from transformers import MixtralConfig as LlamaConfig
    from transformers import MixtralForCausalLM as LlamaForCausalLM
    from transformers import LlamaTokenizer
    from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer as LlamaDecoderLayer
elif model=="mistral":
    from transformers import MistralConfig as LlamaConfig
    from transformers import MistralForCausalLM as LlamaForCausalLM
    from transformers import LlamaTokenizer
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer as LlamaDecoderLayer

elif model == "qwen":
    from transformers import Qwen2ForCausalLM as LlamaForCausalLM
    from transformers import Qwen2Config as LlamaConfig
    from transformers import Qwen2TokenizerFast as LlamaTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as LlamaDecoderLayer

elif model=="llama":
    from transformers import (
    LlamaForCausalLM,
    AutoTokenizer as LlamaTokenizer,
    LlamaConfig,)
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    

