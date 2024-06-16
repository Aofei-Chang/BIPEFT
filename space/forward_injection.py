import torch
import torch.nn as nn
import torch.nn.functional as F

# from lora.t5_forward import t5_forward, stack_forward, block_forward, SelfAttn_forward, attn_forward, ffn_forward, dense_forward
from space.t5_forward_mom import t5_forward, stack_forward, block_forward, SelfAttn_forward, CrossAttn_forward, attn_forward, ffn_forward, dense_forward

def set_lora_forward(model):
    # change the forward of vit model
    bound_method = t5_forward.__get__(model, model.__class__)
    setattr(model, 'forward', bound_method)
    encoder, decoder = model.encoder, model.decoder

    bound_method_encoder = stack_forward.__get__(encoder, encoder.__class__)
    setattr(encoder, 'forward', bound_method_encoder)
    bound_method_decoder = stack_forward.__get__(decoder, decoder.__class__)
    setattr(decoder, 'forward', bound_method_decoder)

    for blk in encoder.block:
        bound_blk_method = block_forward.__get__(blk, blk.__class__)
        setattr(blk, 'forward', bound_blk_method)
        attn_wrapper, attn = blk.layer[0], blk.layer[0].SelfAttention
        bound_attn_wrapper_method = SelfAttn_forward.__get__(attn_wrapper, attn_wrapper.__class__)
        setattr(attn_wrapper, 'forward', bound_attn_wrapper_method)
        bound_attn_method = attn_forward.__get__(attn, attn.__class__)
        setattr(attn, 'forward', bound_attn_method)

        ffn_wrapper, ffn = blk.layer[-1], blk.layer[-1].DenseReluDense
        bound_ffn_wrapper_method = ffn_forward.__get__(ffn_wrapper, ffn_wrapper.__class__)
        setattr(ffn_wrapper, 'forward', bound_ffn_wrapper_method)
        bound_ffn_method = dense_forward.__get__(ffn, ffn.__class__)
        setattr(ffn, 'forward', bound_ffn_method)


    for blk in decoder.block:
        bound_blk_method = block_forward.__get__(blk, blk.__class__)
        setattr(blk, 'forward', bound_blk_method)
        attn_wrapper, attn = blk.layer[0], blk.layer[0].SelfAttention
        cross_attn_wrapper = blk.layer[1]
        bound_cross_attn_wrapper_method = CrossAttn_forward.__get__(cross_attn_wrapper, cross_attn_wrapper.__class__)
        bound_attn_wrapper_method = SelfAttn_forward.__get__(attn_wrapper, attn_wrapper.__class__)
        setattr(cross_attn_wrapper, 'forward', bound_cross_attn_wrapper_method)
        setattr(attn_wrapper, 'forward', bound_attn_wrapper_method)
        bound_attn_method = attn_forward.__get__(attn, attn.__class__)
        setattr(attn, 'forward', bound_attn_method)

        ffn_wrapper, ffn = blk.layer[-1], blk.layer[-1].DenseReluDense
        bound_ffn_wrapper_method = ffn_forward.__get__(ffn_wrapper, ffn_wrapper.__class__)
        setattr(ffn_wrapper, 'forward', bound_ffn_wrapper_method)
        bound_ffn_method = dense_forward.__get__(ffn, ffn.__class__)
        setattr(ffn, 'forward', bound_ffn_method)

    print("Set new forward functions in T5 with lora weight as input!")


