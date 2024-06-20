# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import functools

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


# def get_llama_wrapper():
#     """we register our main layer class and use the fsdp transformer wrapping policy
#     ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
#     """
#     # ====   use new transformer wrapper

#     llama_auto_wrap_policy = functools.partial(
#         transformer_auto_wrap_policy,
#         transformer_layer_cls={
#             LlamaDecoderLayer,
#         },
#     )

#     return llama_auto_wrap_policy

# add plicy support qwen
def get_llama_wrapper(cfg):
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    if cfg.parallel_granularity == 'decoder_layer':
        print('wrap granularity: LlamaDecoderLayer')
        llama_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            },
        )
    elif cfg.parallel_granularity == 'weight':
        print('wrap granularity: weight')
        def lambda_policy_fn(module):
            if (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
            ):
                return True
            return False

        llama_auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    else:
        minimal_units_modules = cfg.parallel_granularity.split('-')
        assert len(minimal_units_modules) > 0
        print('最小并行粒度：', minimal_units_modules)
        def lambda_policy_fn(module):
            module_name = module._get_name()
            if module_name in minimal_units_modules:
                print(module_name, 'in', minimal_units_modules)
                return True
            print(module_name, 'not in', minimal_units_modules)
            return False
        llama_auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    return llama_auto_wrap_policy
