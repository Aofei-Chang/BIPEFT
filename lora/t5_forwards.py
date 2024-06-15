import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss


def t5_forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_value_states=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        gumbel_weights=None,
        **kwargs
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

    if "lm_labels" in kwargs:
        warnings.warn(
            "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
            DeprecationWarning,
        )
        labels = kwargs.pop("lm_labels")
    assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

    use_cache = use_cache if use_cache is not None else self.config.use_cache

    encoder_gumbel_weights, decoder_gumbel_weights = None, None
    if gumbel_weights is not None:
        encoder_layers = len(self.encoder.block)
        encoder_gumbel_weights = gumbel_weights[:encoder_layers]
        decoder_gumbel_weights = gumbel_weights[encoder_layers:]
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
            gumbel_weights = encoder_gumbel_weights
        )

    hidden_states = encoder_outputs[0]

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # If decoding with past key value states, only the last tokens
    # should be given as an input
    if decoder_past_key_value_states is not None:
        assert labels is None, "Decoder should not use cached key value states when training."
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        if decoder_inputs_embeds is not None:
            decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_value_states=decoder_past_key_value_states,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        gumbel_weights=decoder_gumbel_weights
    )

    # insert decoder past at right place
    # to speed up decoding
    if use_cache is True:
        past = ((encoder_outputs, decoder_outputs[1]),)
        decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

    sequence_output = decoder_outputs[0]
    # Rescale output before projecting on vocab
    # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
    sequence_output = sequence_output * (self.model_dim ** -0.5)
    lm_logits = self.lm_head(sequence_output)

    decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        decoder_outputs = (loss,) + decoder_outputs

    return decoder_outputs + encoder_outputs

def stack_forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_value_states=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        gumbel_weights=None
):

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        if self.is_decoder:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

    if inputs_embeds is None:
        assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape

    if past_key_value_states is not None:
        assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
            input_shape, (batch_size, 1)
        )
        # required mask seq length can be calculated via length of past
        # key value states and seq_length = 1 for the last token
        mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length
    else:
        mask_seq_length = seq_length

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
    if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_seq_length = encoder_hidden_states.shape[1]
        encoder_attention_mask = torch.ones(
            batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
        )

    # initialize past_key_value_states with `None` if past does not exist
    if past_key_value_states is None:
        past_key_value_states = [None] * len(self.block)

    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

    if self.is_decoder and encoder_attention_mask is not None:
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    present_key_value_states = ()
    all_hidden_states = ()
    all_attentions = ()
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)

    for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        gumbel_weight_layer = None
        if gumbel_weights is not None:
            gumbel_weight_layer = gumbel_weights[i]
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=extended_attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            head_mask=head_mask[i],
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            gumbel_weight_layer=gumbel_weight_layer

        )
        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
        hidden_states, present_key_value_state = layer_outputs[:2]

        if i == 0:
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[3 if output_attentions else 2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
        # append next layer key value states
        present_key_value_states = present_key_value_states + (present_key_value_state,)

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if use_cache is True:
        assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)
        outputs = outputs + (present_key_value_states,)
    if output_hidden_states:
        outputs = outputs + (all_hidden_states,)
    if output_attentions:
        outputs = outputs + (all_attentions,)
    return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)

def block_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
        output_attentions=False,
        gumbel_weight_layer=None
):

    if past_key_value_state is not None:
        assert self.is_decoder, "Only decoder can use `past_key_value_states`"
        expected_num_past_key_value_states = 2 if encoder_hidden_states is None else 4

        error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
            expected_num_past_key_value_states,
            "2 (past / key) for cross attention" if expected_num_past_key_value_states == 4 else "",
            len(past_key_value_state),
        )
        assert len(past_key_value_state) == expected_num_past_key_value_states, error_message

        self_attn_past_key_value_state = past_key_value_state[:2]
        cross_attn_past_key_value_state = past_key_value_state[2:]
    else:
        self_attn_past_key_value_state, cross_attn_past_key_value_state = None, None

    self_attention_outputs = self.layer[0](
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        head_mask=head_mask,
        past_key_value_state=self_attn_past_key_value_state,
        use_cache=use_cache,
        output_attentions=output_attentions,
        gumbel_weight_layer=gumbel_weight_layer
    )
    hidden_states, present_key_value_state = self_attention_outputs[:2]
    attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

    if self.is_decoder and encoder_hidden_states is not None:
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
            layer_head_mask=head_mask,
            past_key_value=cross_attn_past_key_value_state,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = cross_attention_outputs[0]
        # Combine self attn and cross attn key value states
        if present_key_value_state is not None:
            present_key_value_state = present_key_value_state + cross_attention_outputs[1]

        # Keep cross-attention outputs and relative position weights
        attention_outputs = attention_outputs + cross_attention_outputs[2:]

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states)
    outputs = (hidden_states,)

    # Add attentions if we output them
    outputs = outputs + (present_key_value_state,) + attention_outputs
    return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)

def SelfAttn_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
        output_attentions=False,
        gumbel_weight_layer=None
):
    norm_x = self.layer_norm(hidden_states)
    attention_output = self.SelfAttention(
        norm_x,
        mask=attention_mask,
        position_bias=position_bias,
        head_mask=head_mask,
        past_key_value_state=past_key_value_state,
        use_cache=use_cache,
        output_attentions=output_attentions,
        gumbel_weight_layer=gumbel_weight_layer
    )
    y = attention_output[0]
    layer_output = hidden_states + self.dropout(y)
    outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
    return outputs

def attn_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        gumbel_weight_layer=None
):
    """
    Self-attention (if kv is None) or attention over source sentence (provided by kv).
    """
    # Input is (bs, qlen, dim)
    # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
    # past_key_value_state[0] is (bs, n_heads, q_len - 1, dim_per_head)
    # print([attr for attr in dir(self) if not attr.startswith('__')])
    self.d_kv = self.key_value_proj_dim
    bs, qlen, dim = input.size()

    #set gumbel_weights for linears
    gumbel_q, gumbel_k, gumbel_v, gumbel_o = [None] * 4
    if gumbel_weight_layer is not None:
        if gumbel_weight_layer.shape[0] > 2:
            gumbel_q, gumbel_k, gumbel_v, gumbel_o = gumbel_weight_layer
        else:
            gumbel_q, gumbel_v = gumbel_weight_layer

    if past_key_value_state is not None:
        assert self.is_decoder is True, "Encoder cannot cache past key value states"
        assert (
                len(past_key_value_state) == 2
        ), "past_key_value_state should have 2 past states: keys and values. Got {} past states".format(
            len(past_key_value_state)
        )
        real_qlen = qlen + past_key_value_state[0].shape[2] if query_length is None else query_length
    else:
        real_qlen = qlen

    if kv is None:
        klen = real_qlen
    else:
        klen = kv.size(1)

    def shape(x):
        """  projection """
        return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

    def unshape(x):
        """  compute context """
        return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

    q = shape(self.q(input, gumbel_q))  # (bs, n_heads, qlen, dim_per_head)

    if kv is None:
        k = shape(self.k(input, gumbel_k))  # (bs, n_heads, qlen, dim_per_head)
        v = shape(self.v(input, gumbel_v))  # (bs, n_heads, qlen, dim_per_head)
    elif past_key_value_state is None:
        k = v = kv
        k = shape(self.k(k, gumbel_k))  # (bs, n_heads, qlen, dim_per_head)
        v = shape(self.v(v, gumbel_v))  # (bs, n_heads, qlen, dim_per_head)

    if past_key_value_state is not None:
        if kv is None:
            k_, v_ = past_key_value_state
            k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
            v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
        else:
            k, v = past_key_value_state

    if self.is_decoder and use_cache is True:
        present_key_value_state = ((k, v),)
    else:
        present_key_value_state = (None,)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

    if position_bias is None:
        if not self.has_relative_attention_bias:
            raise ValueError("No position_bias provided and no weights to compute position_bias")
        position_bias = self.compute_bias(real_qlen, klen)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value_state is not None:
            position_bias = position_bias[:, :, -1:, :]

        if mask is not None:
            position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

    scores += position_bias
    weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
    weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

    # Mask heads if we want to
    if head_mask is not None:
        weights = weights * head_mask

    context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
    context = unshape(context)  # (bs, qlen, dim)

    context = self.o(context, gumbel_o)

    outputs = (context,) + present_key_value_state

    if output_attentions:
        outputs = outputs + (weights,)
    if self.has_relative_attention_bias:
        outputs = outputs + (position_bias,)
    return outputs