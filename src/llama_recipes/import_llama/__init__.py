import os
model = os.getenv('FT_MODEL_TYPE')
model = str(model).lower()
print(f"model type is {model}")
if model=='mixtral':
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
else:
    raise Exception(f'请设置环境变量: "FT_MODEL_TYPE",其值必须属于llama,mistral,mixtral,qwen')
    

