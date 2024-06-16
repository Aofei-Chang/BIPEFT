import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.utils import logging

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

logger = logging.get_logger(__name__)

def t5_forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gumbel_weights=None,
        dimension_mask=None,
        iterative_order=None, main_forward=True
    ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
        Labels for computing the sequence classification/regression loss.
        Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
        All labels set to ``-100`` are ignored (masked), the loss is only
        computed for labels in ``[0, ..., config.vocab_size]``
    kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
        Used to hide legacy arguments that have been deprecated.

Returns:
    :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
        Classification loss (cross entropy).
    prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
    decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
        Contains pre-computed key and value hidden-states of the attention blocks.
        Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
        Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
        Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
        of shape :obj:`(batch_size, sequence_length, hidden_size)`.

        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
        Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
        :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.

Examples::

    >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

    >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
    >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
    >>> input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
    >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
    >>> loss, prediction_scores = outputs[:2]

    >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
    >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
    >>> input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
    >>> outputs = model.generate(input_ids)
    """

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    encoder_prefix, decoder_prefix = None, None
    if hasattr(self, "prefix_module"):
        gumbel_prefix, dimension_mask_prefix = None, None
        if gumbel_weights is not None:
            gumbel_prefix = gumbel_weights['prefix']
            dimension_mask_prefix = dimension_mask['prefix_dimension_mask']
        prefix = self.prefix_module.eject(gumbel_prefix, dimension_mask=dimension_mask_prefix, iterative_order=iterative_order, main_forward=main_forward)
        # print("prefix 1", prefix.size())
        encoder_prefix, decoder_prefix = prefix[:24], prefix[24:]

    encoder_gumbel_weights, decoder_gumbel_weights, final_norm_gumbel_weights = None, None, None
    encoder_dimension_mask, decoder_dimension_mask = None, None
    # print("gumbel weights in fors", gumbel_weights)
    if gumbel_weights is not None:
        # encoder_layers = len(self.encoder.block)

        encoder_gumbel_weights = {
            "matrix": gumbel_weights['encoder'],
            "binary": gumbel_weights['encoder_binary'],
            "final_norm": gumbel_weights['final_norm'][:2, :],
            "tag": "encoder"
        }
        decoder_gumbel_weights = {
            "matrix": gumbel_weights['decoder'],
            "binary": gumbel_weights['decoder_binary'],
            "final_norm": gumbel_weights['final_norm'][2:, :],
            "tag": "decoder"
        }
        if dimension_mask is not None:
            encoder_dimension_mask = dimension_mask["encoder_dimension_mask"]
            decoder_dimension_mask = dimension_mask["decoder_dimension_mask"]
    # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
        if self.config.num_layers == self.config.num_decoder_layers:
            warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gumbel_weights=encoder_gumbel_weights,
            dimension_mask=encoder_dimension_mask,
            iterative_order=iterative_order, main_forward=main_forward,
            prefix_stack=encoder_prefix
        )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # If decoding with past key value states, only the last tokens
    # should be given as an input
    if past_key_values is not None:
        assert labels is None, "Decoder should not use cached key value states when training."
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        if decoder_inputs_embeds is not None:
            decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)
        hidden_states = hidden_states.to(self.decoder.first_device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.decoder.first_device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        gumbel_weights=decoder_gumbel_weights,
        dimension_mask=decoder_dimension_mask,
        iterative_order=iterative_order, main_forward=main_forward,
        prefix_stack=decoder_prefix
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.encoder.first_device)
        self.lm_head = self.lm_head.to(self.encoder.first_device)
        sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)

    lm_logits = self.lm_head(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    if not return_dict:
        output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return ((loss,) + output) if loss is not None else output

    return Seq2SeqLMOutput(
        loss=loss,
        logits=lm_logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


def stack_forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gumbel_weights=None,
        dimension_mask=None,
        iterative_order=None, main_forward=True,
        prefix_stack=None
    ):
    # print("stack_for:", iterative_order, main_forward)
    # Model parallel
    if self.model_parallel:
        torch.cuda.set_device(self.first_device)
        self.embed_tokens = self.embed_tokens.to(self.first_device)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(
            f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

    if inputs_embeds is None:
        assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape

    # required mask seq length can be calculated via length of past
    mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

    if use_cache is True:
        assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
    if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_seq_length = encoder_hidden_states.shape[1]
        encoder_attention_mask = torch.ones(
            batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
        )

    # initialize past_key_values with `None` if past does not exist
    if past_key_values is None:
        past_key_values = [None] * len(self.block)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
    present_key_value_states = () if use_cache else None
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)

    #assgin gumbel weights
    gumbel_matrix, gumbel_binary, gumbel_final_norm = None, None, None
    tag = None
    if gumbel_weights is not None:
        gumbel_matrix, gumbel_binary = gumbel_weights['matrix'], gumbel_weights['binary']
        tag = gumbel_weights['tag']
        gumbel_final_norm = {
            "lora": None,
            "adapter": None,
            "bitfit": gumbel_weights['final_norm'][0],
            "lnfit": gumbel_weights['final_norm'][1]
        }

    for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
        layer_head_mask = head_mask[i]
        cross_attn_layer_head_mask = cross_attn_head_mask[i]

        gumbel_weight_layer, dimension_mask_layer = None, None
        if gumbel_weights is not None:
            gumbel_weight_layer = {
                "matrix": gumbel_matrix[i],
                "binary": gumbel_binary[i],
                "tag": tag
            }
        if dimension_mask is not None:
            dimension_mask_layer = dimension_mask[i]
        prefix_layer = None
        if prefix_stack is not None:
            prefix_layer = prefix_stack[i]

        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if position_bias is not None:
                position_bias = position_bias.to(hidden_states.device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            if encoder_extended_attention_mask is not None:
                encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            if encoder_decoder_position_bias is not None:
                encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            if layer_head_mask is not None:
                layer_head_mask = layer_head_mask.to(hidden_states.device)
            if cross_attn_layer_head_mask is not None:
                cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if getattr(self.config, "gradient_checkpointing", False) and self.training:
            if use_cache:
                logger.warn(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return tuple(module(*inputs, use_cache, output_attentions))

                return custom_forward

            layer_outputs = checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                extended_attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,
                layer_head_mask,
                cross_attn_layer_head_mask,
                None,  # past_key_value is always None with gradient checkpointing
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                gumbel_weight_layer=gumbel_weight_layer,
                dimension_mask_layer=dimension_mask_layer,
                iterative_order=iterative_order, main_forward=main_forward,
                prefix_layer=prefix_layer
            )

        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        if use_cache is False:
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        hidden_states, present_key_value_state = layer_outputs[:2]

        # We share the position biases between the layers - the first layer store them
        # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)
        position_bias = layer_outputs[2]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = layer_outputs[-1]
            # encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
        # append next layer key value states
        if use_cache:
            present_key_value_states = present_key_value_states + (present_key_value_state,)

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[3],)
            if self.is_decoder:
                all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.final_layer_norm(hidden_states, gumbel_final_norm)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=present_key_value_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        cross_attentions=all_cross_attentions,
    )

def block_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        gumbel_weight_layer=None,
        dimension_mask_layer=None,
        iterative_order=None, main_forward=True,
        prefix_layer=None
    ):
    # print("block_for:", iterative_order, main_forward)
    if past_key_value is not None:
        assert self.is_decoder, "Only decoder can use `past_key_values`"
        expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

        if len(past_key_value) != expected_num_past_key_values:
            raise ValueError(
                f"There should be {expected_num_past_key_values} past states. "
                f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                f"Got {len(past_key_value)} past key / value states"
            )

        self_attn_past_key_value = past_key_value[:2]
        cross_attn_past_key_value = past_key_value[2:]
    else:
        self_attn_past_key_value, cross_attn_past_key_value = None, None
    # gumbel_weight_layer, {matrix:[6+2, dims] (lora, adapter), binary:[8+2, 2] (bitfit, lnfit) if encoder else binary:[9+3, 2]}
    gumbel_weight_layer_attn, dimension_mask_layer_attn, gumbel_weight_layer_ffn, dimension_mask_layer_ffn = [None] * 4
    gumbel_matrix_adapter_attn, gumbel_matrix_adapter_ffn = None, None
    gumbel_bianry_cross_attn = None
    gumbel_weight_cross_attn = None
    if gumbel_weight_layer is not None:
        tag = gumbel_weight_layer['tag']
        gumbel_matrix = gumbel_weight_layer['matrix']
        gumbel_binary = gumbel_weight_layer['binary']
        gumbel_matrix_attn, gumbel_matrix_ffn = gumbel_matrix[:4], gumbel_matrix[4:6]  # for qkvo
        gumbel_matrix_adapter = gumbel_matrix[6:8]
        gumbel_matrix_sapa_ffn, gumbel_matrix_prefix = None, None
        gumbel_matrix_adapter_attn, gumbel_matrix_adapter_ffn = gumbel_matrix_adapter[0], gumbel_matrix_adapter[1]
        if gumbel_matrix.shape[0] > 8:
            gumbel_matrix_sapa_ffn = gumbel_matrix[8:10]
            gumbel_matrix_adapter_ffn = (gumbel_matrix_adapter_ffn, gumbel_matrix_sapa_ffn[0], gumbel_matrix_sapa_ffn[1])

        if tag == "encoder":
            gumbel_binary_attn, gumbel_binary_attn_ln, gumbel_binary_ffn, gumbel_binary_ffn_ln = gumbel_binary[:4], gumbel_binary[4:4+2], gumbel_binary[6:6+2], gumbel_binary[8:6+2+2]  # in self.attn, 4 bitfit, 1ln+1bitfit
        else:
            gumbel_binary_attn, gumbel_binary_attn_ln, gumbel_binary_ffn, gumbel_binary_ffn_ln= gumbel_binary[:4], gumbel_binary[4:4+2], gumbel_binary[6:6+2], gumbel_binary[8:6+2+2]
            gumbel_bianry_cross_attn = gumbel_binary[6+2+2:] # 1 ln + 1 bitfit

        gumbel_weight_layer_attn = {
            "matrix": gumbel_matrix_attn,
            "binary": gumbel_binary_attn,
            "layer_norm": gumbel_binary_attn_ln,
            "prefix": None
        }
        gumbel_weight_layer_ffn = {
            "matrix": gumbel_matrix_ffn,
            "binary": gumbel_binary_ffn,
            "layer_norm": gumbel_binary_ffn_ln,
        }
        gumbel_weight_cross_attn = {
            "matrix": None,
            "binary": gumbel_bianry_cross_attn
        }
    dimension_mask_attn_adapter, dimension_mask_ffn_adapter = None, None
    if dimension_mask_layer is not None:
        dimension_mask_layer_attn, dimension_mask_layer_ffn = dimension_mask_layer[:4], dimension_mask_layer[4:6]  # for ffn (w_i, w_o)
        dimension_mask_layer_adapter = dimension_mask_layer[6:8] # for adapter
        dimension_mask_sapa_ffn, gumbel_matrix_prefix = None, None
        if dimension_mask_layer.shape[0] > 8:
            dimension_mask_sapa_ffn = dimension_mask_layer[8:10]
        dimension_mask_attn_adapter, dimension_mask_ffn_adapter = dimension_mask_layer_adapter[0], dimension_mask_layer_adapter[1]
        if dimension_mask_sapa_ffn is not None:
            dimension_mask_ffn_adapter = (dimension_mask_ffn_adapter, dimension_mask_sapa_ffn[0], dimension_mask_sapa_ffn[1])

    self_attention_outputs = self.layer[0](
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        layer_head_mask=layer_head_mask,
        past_key_value=self_attn_past_key_value,
        use_cache=use_cache,
        # here, these 2 vars are needed for PEFT with adapter
        gumbel_weight_self=gumbel_matrix_adapter_attn,
        dimension_mask_self=dimension_mask_attn_adapter,
        output_attentions=output_attentions,
        #here, these 2 vars are needed for sub modules, which are other modules in PEFT class
        gumbel_weight_layer=gumbel_weight_layer_attn,
        dimension_mask_layer=dimension_mask_layer_attn,
        iterative_order=iterative_order, main_forward=main_forward,
        prefix=prefix_layer
    )
    hidden_states, present_key_value_state = self_attention_outputs[:2]
    attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    do_cross_attention = self.is_decoder and encoder_hidden_states is not None
    if do_cross_attention:
        # the actual query length is unknown for cross attention
        # if using past key value states. Need to inject it here
        if present_key_value_state is not None:
            query_length = present_key_value_state[0].shape[2]
        else:
            query_length = None

        cross_attention_outputs = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            gumbel_weight_layer=gumbel_weight_cross_attn,
            output_attentions=output_attentions,
        )
        hidden_states = cross_attention_outputs[0]

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Combine self attn and cross attn key value states
        if present_key_value_state is not None:
            present_key_value_state = present_key_value_state + cross_attention_outputs[1]

        # Keep cross-attention outputs and relative position weights
        attention_outputs = attention_outputs + cross_attention_outputs[2:]

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states,
                                   # here, these 2 vars are needed for sub modules, which are other modules in PEFT class
                                   gumbel_weight_layer=gumbel_weight_layer_ffn, dimension_mask_layer=dimension_mask_layer_ffn,
                                   # here, these 2 vars are needed for PEFT with adapter
                                   gumbel_weight_self=gumbel_matrix_adapter_ffn, dimension_mask_self=dimension_mask_ffn_adapter,
                                   iterative_order=iterative_order, main_forward=main_forward)

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)

    if use_cache:
        outputs = outputs + (present_key_value_state,) + attention_outputs
    else:
        outputs = outputs + attention_outputs

    return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


def SelfAttn_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        gumbel_weight_self=None,
        dimension_mask_self=None,
        gumbel_weight_layer=None,
        dimension_mask_layer=None,
        iterative_order=None, main_forward=True,
        prefix=None
    ):
        # print("self_attn:", iterative_order, main_forward)
        gumbel_layer_norm, gumbel_weight_self_dict, dimension_mask_self_dict = None, None, None
        if gumbel_weight_layer is not None:
            gumbel_layer_norm = {
                "adapter": None, "lora": None,
                "bitfit":gumbel_weight_layer['layer_norm'][0], "lnfit":gumbel_weight_layer['layer_norm'][1]
            }

            gumbel_weight_self_dict = {
                "adapter": gumbel_weight_self,
                "lora": None,
                "bitfit": None,
                "lnfit": None
            }

            dimension_mask_self_dict = {
                "adapter": dimension_mask_self,
                "lora": None,
            }

        normed_hidden_states = self.layer_norm(hidden_states, gumbel_weights=gumbel_layer_norm)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            gumbel_weights=gumbel_weight_self_dict,
            dimension_mask=dimension_mask_self_dict,
            gumbel_weight_layer=gumbel_weight_layer,
            dimension_mask_layer=dimension_mask_layer,
            iterative_order=iterative_order, main_forward=main_forward,
            prefix=prefix
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


def CrossAttn_forward(
    self,
    hidden_states,
    key_value_states,
    attention_mask=None,
    position_bias=None,
    layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    query_length=None,
    output_attentions=False,
    gumbel_weight_layer=None
):
    if gumbel_weight_layer is not None:
        gumbel_weight_layer = {
            "lora": None,
            "adapter": None,
            "bitfit": gumbel_weight_layer["binary"][0],
            "lnfit": gumbel_weight_layer["binary"][1],
        }
    normed_hidden_states = self.layer_norm(hidden_states, gumbel_weights=gumbel_weight_layer)
    # normed_hidden_states = self.layer_norm(hidden_states)
    # print("encdec")
    attention_output = self.EncDecAttention(
        normed_hidden_states,
        mask=attention_mask,
        key_value_states=key_value_states,
        position_bias=position_bias,
        layer_head_mask=layer_head_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        query_length=query_length,
        output_attentions=output_attentions,
    )
    layer_output = hidden_states + self.dropout(attention_output[0])
    outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
    return outputs


def attn_forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        gumbel_weight_layer=None,
        dimension_mask_layer=None,
        iterative_order=None, main_forward=True,
        prefix=None,
    ):
    # print(gumbel_weight_layer)
    # print("attn_for:", iterative_order, main_forward)
    """
            Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
            """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    # set gumbel_weights for linears


    gumbel_q, gumbel_k, gumbel_v, gumbel_o = [None] * 4
    gumbel_q_bias, gumbel_k_bias, gumbel_v_bias, gumbel_o_bias = [None] * 4
    mask_q, mask_k, mask_v, mask_o = [None] * 4
    if gumbel_weight_layer is not None:
        lora_gumbel = gumbel_weight_layer['matrix']
        gumbel_q, gumbel_k, gumbel_v, gumbel_o = lora_gumbel
        bitfit_gumbel = gumbel_weight_layer['binary']
        gumbel_q_bias, gumbel_k_bias, gumbel_v_bias, gumbel_o_bias = bitfit_gumbel
        if dimension_mask_layer is not None:
            mask_q, mask_k, mask_v, mask_o = dimension_mask_layer


    if past_key_value is not None:
        assert (
                len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value, gumbel_weight=None, dimension_mask=None):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states, gumbel_weight, dimension_mask=dimension_mask, iterative_order=iterative_order, main_forward=main_forward))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states, gumbel_weight, dimension_mask=dimension_mask, iterative_order=iterative_order, main_forward=main_forward))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    mask_q_dict = {"lora":mask_q, "adapter":None}
    gumbel_q_dict = {"lora":gumbel_q, "adapter":None, "bitfit":gumbel_q_bias, "lnfit":None}
    query_states = shape(self.q(hidden_states, gumbel_weights=gumbel_q_dict, dimension_mask=mask_q_dict,
                                iterative_order=iterative_order, main_forward=main_forward))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    mask_k_dict = {"lora": mask_k, "adapter": None}
    gumbel_k_dict = {"lora": gumbel_k, "adapter": None, "bitfit": gumbel_k_bias, "lnfit": None}
    mask_v_dict = {"lora": mask_v, "adapter": None}
    gumbel_v_dict = {"lora": gumbel_v, "adapter": None, "bitfit": gumbel_v_bias, "lnfit": None}
    
    key_hidden, value_hidden = hidden_states, hidden_states
    prefix_length = 0
    if prefix is not None:

        # print("prefix is not None:", prefix.size())
        batch_size = hidden_states.shape[0]
        prefix_key = prefix[0].unsqueeze(0).expand(batch_size, -1, -1)
        prefix_value = prefix[1].unsqueeze(0).expand(batch_size, -1, -1)
        # print("prefix is not None:", prefix_key.size())
        prefix_length = prefix_key.shape[1]
        # key_hidden = torch.cat((prefix_key, hidden_states), dim=1)
        # value_hidden = torch.cat((prefix_value, hidden_states), dim=1)
        key_hidden = prefix_key
        value_hidden = prefix_value
        # print("key is not None:", key_hidden.size())
        prefix_key_states = project(
            key_hidden, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None,
            gumbel_weight=gumbel_k_dict, dimension_mask=mask_k_dict
        )
        prefix_value_states = project(
            value_hidden, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None,
            gumbel_weight=gumbel_v_dict, dimension_mask=mask_v_dict
        )

    key_states = project(
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None,
        gumbel_weight=gumbel_k_dict, dimension_mask=mask_k_dict
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None,
        gumbel_weight=gumbel_v_dict, dimension_mask=mask_v_dict
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
    # print("sc is not None:", scores.size())
    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.training and self.gradient_checkpointing:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1):, :]

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    # print(scores.size(), "score")
    # print(position_bias.size(), "position_bias")
    if prefix is not None:
        prefix_scores = torch.matmul(query_states, prefix_key_states.transpose(3, 2))

    scores += position_bias
    if prefix is not None:
        # Apply gating to the prefix attention scores
        # print(prefix_scores.size(), "cje")
        # print(scores.size(), "cje")
        prefix_attn_weights = self.prefix_gate * nn.functional.softmax(prefix_scores.float(), dim=-1).type_as(
            scores
        )
        prefix_output = unshape(torch.matmul(prefix_attn_weights, prefix_value_states))  # (batch_size, seq_length, dim)

    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )  # (batch_size, n_heads, seq_length, key_length)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask
    attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    if prefix is not None:
        attn_output = attn_output + prefix_output
    mask_o_dict = {"lora": mask_o, "adapter": None}
    gumbel_o_dict = {"lora": gumbel_o, "adapter": None, "bitfit": gumbel_o_bias, "lnfit": None}
    attn_output = self.o(attn_output, gumbel_o_dict, dimension_mask=mask_o_dict, iterative_order=iterative_order, main_forward=main_forward)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs



__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

#ffn forwards

def ffn_forward(self, hidden_states, gumbel_weight_self=None, dimension_mask_self=None, gumbel_weight_layer=None, dimension_mask_layer=None, iterative_order=None, main_forward=True):
    gumbel_layer_norm, gumbel_weight_self_dict, dimension_mask_self_dict = None, None, None
    if gumbel_weight_layer is not None:
        gumbel_layer_norm = {
            "adapter": None, "lora": None,
            "bitfit": gumbel_weight_layer['layer_norm'][0], "lnfit": gumbel_weight_layer['layer_norm'][1]
        }

        gumbel_weight_self_dict = {
            "adapter": gumbel_weight_self,
            "lora": None,
            "bitfit": None,
            "lnfit": None
        }
        if isinstance(gumbel_weight_self, tuple):
            gumbel_weight_self_dict = {
                "adapter": gumbel_weight_self[0],
                "sa": gumbel_weight_self[1],
                "pa": gumbel_weight_self[2],
                "lora": None,
                "bitfit": None,
                "lnfit": None
            }

        dimension_mask_self_dict = {
            "adapter": dimension_mask_self,
            "lora": None,
        }

        if isinstance(dimension_mask_self, tuple):
            dimension_mask_self_dict = {
                "adapter": dimension_mask_self[0],
                "sa": dimension_mask_self[1],
                "pa": dimension_mask_self[2],
                "lora": None,
            }

    norm_x = self.layer_norm(hidden_states, gumbel_weights=gumbel_layer_norm)
    y = self.DenseReluDense(norm_x, gumbel_weights=gumbel_weight_self_dict, dimension_mask=dimension_mask_self_dict, gumbel_weight_layer=gumbel_weight_layer, dimension_mask_layer=dimension_mask_layer,
                            iterative_order=iterative_order, main_forward=main_forward)
    layer_output = hidden_states + self.dropout(y)
    return layer_output

# def dense_forward()
def dense_forward(self, hidden_states, gumbel_weight_layer=None, dimension_mask_layer=None, iterative_order=None, main_forward=True):
    # set gumbel_weights for linears
    gumbel_ffn1, gumbel_ffn2 = [None] * 2
    gumbel_ffn1_bias, gumbel_ffn2_bias = [None] * 2
    mask_ffn1, mask_ffn2 = [None] * 2
    if gumbel_weight_layer is not None:
        gumbel_ffn1, gumbel_ffn2 = gumbel_weight_layer['matrix']
        gumbel_ffn1_bias, gumbel_ffn2_bias = gumbel_weight_layer['binary']
        if dimension_mask_layer is not None:
            mask_ffn1, mask_ffn2 = dimension_mask_layer


    if gumbel_ffn1 is not None:
        mask_ffn1_dict = {"lora": mask_ffn1, "adapter": None}
        gumbel_ffn1_dict = {"lora": gumbel_ffn1, "adapter": None, "bitfit": gumbel_ffn1_bias, "lnfit": None}
        h = self.wi(hidden_states, gumbel_weights=gumbel_ffn1_dict, dimension_mask=mask_ffn1_dict, iterative_order=iterative_order, main_forward=main_forward)
    else:
        h = self.wi(hidden_states)
    h = F.relu(h)
    h = self.dropout(h)
    if gumbel_ffn2 is not None:
        mask_ffn2_dict = {"lora": mask_ffn2, "adapter": None}
        gumbel_ffn2_dict = {"lora": gumbel_ffn2, "adapter": None, "bitfit": gumbel_ffn1_bias, "lnfit": None}
        h = self.wo(h, gumbel_weights=gumbel_ffn2_dict, dimension_mask=mask_ffn2_dict, iterative_order=iterative_order, main_forward=main_forward)
    else:
        h = self.wo(h)
    return h