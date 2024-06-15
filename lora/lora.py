# Aofei Chang at April 22 2024
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.distributions import dirichlet

from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration
from gumbel_module import GumbleSoftmax, gumbel_sample_weight, measure_entropy, calculate_zeta_for_shifting, bernoulli_sample
from lora.peft_modules import LoRA_PEFT
from lora.forward_injection import set_lora_forward



def weights(model: nn.Module):
    res = []
    for n, p in model.named_parameters():
        if 'arch' not in n and p.requires_grad == True:
            res.append(p)
        else:
            continue
    return res


class LoRA_T5(nn.Module):
    def __init__(self, backbone:T5ForConditionalGeneration, r: int, model_config:T5Config, args=None):
        super(LoRA_T5, self).__init__()

        assert r > 0
        self.r = r
        self.num_encoder_layers = model_config.num_layers
        self.num_decoder_layers = model_config.num_decoder_layers
        self.use_search = args.use_search
        self.iter_search = args.iter_search
        self.use_beta = args.use_beta
        self.retrain = False
        if args is not None:
            self.retrain = args.retrain
        self.eval_mode = False
        self.search_all_lora = args.search_all_lora
        self.use_qkvo = args.use_qkvo
        self.GumbleSoftmax = GumbleSoftmax()
        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        self.w_As_final, self.w_Bs_final = [], []
        self.args=args
        self.print_eval = True

        self.search_lora_dim = [0, 8]
        self.search_lora_dim2 = [1, 4, 8]
        if not self.iter_search:
            self.search_lora_dim = [0, 1, 4, 8]
        self.search_lora_dim_final = [0, 1, 4, 8]

        lora_possible_positions = 2
        self.last_zeta = 0
        if self.use_qkvo:
            lora_possible_positions = 4
        if self.search_all_lora:
            lora_possible_positions = 6
        self.num_options = lora_possible_positions
        self.search_multiple = (args.is_LoRA) and (args.is_adapter)
        if self.search_multiple:
            self.num_options = self.num_options+2
        else:
            self.num_options = self.num_options
        self.progressive_fix = args.progressive_fix
        print(f"search multiple: {self.search_multiple}")
        print(f"progressively fix: {args.progressive_fix}")
        # parameters for progressively shrinking and budget control
        self.budget_abs = args.budget_abs
        self.use_budget = args.use_budget

        num_layes = self.num_decoder_layers+self.num_encoder_layers
        if self.iter_search:
            self.arch_weights = nn.Parameter(1e-3 * torch.randn(num_layes, self.num_options, len(self.search_lora_dim), dtype=torch.float32))
            self.arch_weights2 = nn.Parameter(1e-3 * torch.randn(num_layes, self.num_options, len(self.search_lora_dim2), dtype=torch.float32))
            self.binary_search_mask, self.dimension_search_mask = None, None
            if self.progressive_fix:
                self.max_records = torch.zeros_like(self.arch_weights)
                self.max_records2 = torch.zeros_like(self.arch_weights2)
                self.cur_max_arch, self.cur_max_arch2 = None, None
                self.binary_search_mask = torch.zeros(num_layes, self.num_options)
                self.dimension_search_mask = torch.zeros(num_layes, self.num_options)
        else:
            self.arch_weights = nn.Parameter(1e-3 * torch.randn(self.num_decoder_layers + self.num_encoder_layers,
                                                                self.num_options, len(self.search_lora_dim),
                                                                dtype=torch.float32))
        self.binary_mask = torch.zeros(num_layes, self.num_options)  # for binary selection
        self.dimension_mask = torch.zeros(num_layes, self.num_options)  # for binary selection

        for param in backbone.parameters():
            param.requires_grad = False

        # Here, we INSERT the lora modules
        for t_layer_i, blk in enumerate(backbone.encoder.block):
            attn = blk.layer[0].SelfAttention
            ffn = blk.layer[-1].DenseReluDense
            d_model = attn.d_model
            inner_dim = attn.inner_dim

            w_a_linear_q = nn.Linear(d_model, r, bias=False)
            w_b_linear_q = nn.Linear(r, inner_dim, bias=False)
            w_a_linear_v = nn.Linear(d_model, r, bias=False)
            w_b_linear_v = nn.Linear(r, inner_dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            attn.q = LoRA_PEFT(attn.q, w_a_linear_q, w_b_linear_q)
            attn.v = LoRA_PEFT(attn.v, w_a_linear_v, w_b_linear_v)

            if self.use_qkvo:
                w_a_linear_o = nn.Linear(inner_dim, r, bias=False)
                w_b_linear_o = nn.Linear(r, d_model, bias=False)
                w_a_linear_k = nn.Linear(d_model, r, bias=False)
                w_b_linear_k = nn.Linear(r, inner_dim, bias=False)
                self.w_As.append(w_a_linear_o)
                self.w_Bs.append(w_b_linear_o)
                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)

                attn.o = LoRA_PEFT(attn.o, w_a_linear_o, w_b_linear_o)
                attn.k = LoRA_PEFT(attn.k, w_a_linear_k, w_b_linear_k)

            if self.search_all_lora:
                ffn_dim = ffn.wi.out_features
                in_dim, out_dim = ffn.wi.in_features, ffn.wo.out_features

                w_a_linear_ffn1 = nn.Linear(in_dim, r, bias=False)
                w_b_linear_ffn1 = nn.Linear(r, ffn_dim, bias=False)
                w_a_linear_ffn2 = nn.Linear(ffn_dim, r, bias=False)
                w_b_linear_ffn2 = nn.Linear(r, out_dim, bias=False)

                self.w_As.append(w_a_linear_ffn1)
                self.w_Bs.append(w_b_linear_ffn1)
                self.w_As.append(w_a_linear_ffn2)
                self.w_Bs.append(w_b_linear_ffn2)

                ffn.wi = LoRA_PEFT(ffn.wi, w_a_linear_ffn1, w_b_linear_ffn1)
                ffn.wo = LoRA_PEFT(ffn.wo, w_a_linear_ffn2, w_b_linear_ffn2)

        for t_layer_i, blk in enumerate(backbone.decoder.block):
            attn = blk.layer[0].SelfAttention
            ffn = blk.layer[-1].DenseReluDense
            # cross_attn = blk.layer[1].EncDecAttention
            d_model = attn.d_model
            inner_dim = attn.inner_dim
            w_a_linear_q = nn.Linear(d_model, r, bias=False)
            w_b_linear_q = nn.Linear(r, inner_dim, bias=False)
            w_a_linear_v = nn.Linear(d_model, r, bias=False)
            w_b_linear_v = nn.Linear(r, inner_dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            # decoder_d_model = cross_attn.d_model, decoder_inner_dim = cross_attn.inner_dim
            #
            # w_a_linear_q, decoder_a_linear_q = nn.Linear(d_model, r, bias=False), nn.Linear(decoder_d_model, r,
            #                                                                                 bias=False)
            # w_b_linear_q, decoder_b_linear_q = nn.Linear(r, inner_dim, bias=False), nn.Linear(r, decoder_inner_dim,
            #                                                                                   bias=False)
            # w_a_linear_v, decoder_a_linear_v = nn.Linear(d_model, r, bias=False), nn.Linear(decoder_d_model, r,
            #                                                                                 bias=False)
            # w_b_linear_v, decoder_b_linear_v = nn.Linear(r, inner_dim, bias=False), nn.Linear(r, decoder_inner_dim,
            #                                                                                   bias=False)
            # self.w_As.extend([w_a_linear_q, decoder_a_linear_q, w_a_linear_v, decoder_a_linear_v])
            # self.w_Bs.extend([w_b_linear_q, decoder_b_linear_q, w_b_linear_v, decoder_b_linear_v])

            attn.q = LoRA_PEFT(attn.q, w_a_linear_q, w_b_linear_q)
            attn.v = LoRA_PEFT(attn.v, w_a_linear_v, w_b_linear_v)

            if self.use_qkvo:
                w_a_linear_o = nn.Linear(inner_dim, r, bias=False)
                w_b_linear_o = nn.Linear(r, d_model, bias=False)
                w_a_linear_k = nn.Linear(d_model, r, bias=False)
                w_b_linear_k = nn.Linear(r, inner_dim, bias=False)
                self.w_As.append(w_a_linear_o)
                self.w_Bs.append(w_b_linear_o)
                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)

                attn.o = LoRA_PEFT(attn.o, w_a_linear_o, w_b_linear_o)
                attn.k = LoRA_PEFT(attn.k, w_a_linear_k, w_b_linear_k)
            if self.search_all_lora:
                ffn_dim = ffn.wi.out_features
                in_dim, out_dim = ffn.wi.in_features, ffn.wo.out_features

                w_a_linear_ffn1 = nn.Linear(in_dim, r, bias=False)
                w_b_linear_ffn1 = nn.Linear(r, ffn_dim, bias=False)
                w_a_linear_ffn2 = nn.Linear(ffn_dim, r, bias=False)
                w_b_linear_ffn2 = nn.Linear(r, out_dim, bias=False)

                self.w_As.append(w_a_linear_ffn1)
                self.w_Bs.append(w_b_linear_ffn1)
                self.w_As.append(w_a_linear_ffn2)
                self.w_Bs.append(w_b_linear_ffn2)

                ffn.wi = LoRA_PEFT(ffn.wi, w_a_linear_ffn1, w_b_linear_ffn1)
                ffn.wo = LoRA_PEFT(ffn.wo, w_a_linear_ffn2, w_b_linear_ffn2)

        self.reset_parameters()
        self.t5_model = backbone

        for name, param in self.t5_model.named_parameters():
            if "LoRA" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


        self._arch_parameters = [
            self.arch_weights
        ]
        if self.iter_search:
            self._arch_parameters = [
                self.arch_weights,
                self.arch_weights2
            ]
        # set new forward
        if self.use_search:
            set_lora_forward(self.t5_model)

        self.iterative_order = True

    def arch_parameters(self):
        return self._arch_parameters

    def finalize_lora_weight(self, original_a, original_b, rank_index):
        if self.iter_search:
            rank = self.search_lora_dim_final[rank_index]
        else:
            rank = self.search_lora_dim[rank_index]
        dim, r = original_a.weight.shape
        r, dim2 = original_b.weight.shape

        w_a_linear = nn.Linear(dim, rank, bias=False)
        w_b_linear = nn.Linear(rank, dim2, bias=False)
        w_a_linear.weight = nn.Parameter(original_a.weight[:rank, :])
        w_b_linear.weight = nn.Parameter(original_b.weight[:, :rank])

        return w_a_linear, w_b_linear

    def recover_lora(self):
        lora_index = 0
        for t_layer_i, blk in enumerate(self.t5_model.encoder.block):
            attn = blk.layer[0].SelfAttention
            w_a_linear_q = self.w_As[lora_index]
            w_b_linear_q = self.w_Bs[lora_index]
            lora_index += 1
            w_a_linear_v = self.w_As[lora_index]
            w_b_linear_v = self.w_Bs[lora_index]
            lora_index += 1

            attn.q = LoRA_PEFT(attn.q, w_a_linear_q, w_b_linear_q)
            attn.v = LoRA_PEFT(attn.v, w_a_linear_v, w_b_linear_v)

            if self.use_qkvo:
                w_a_linear_o = self.w_As[lora_index]
                w_b_linear_o = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_k = self.w_As[lora_index]
                w_b_linear_k = self.w_Bs[lora_index]
                lora_index += 1
                attn.o = LoRA_PEFT(attn.o, w_a_linear_o, w_b_linear_o)
                attn.k = LoRA_PEFT(attn.k, w_a_linear_k, w_b_linear_k)

        bia = len(self.t5_model.encoder.block)
        for t_layer_i, blk in enumerate(self.t5_model.decoder.block):
            t_layer_i += bia
            attn = blk.layer[0].SelfAttention

            w_a_linear_q = self.w_As[lora_index]
            w_b_linear_q = self.w_Bs[lora_index]
            lora_index += 1
            w_a_linear_v = self.w_As[lora_index]
            w_b_linear_v = self.w_Bs[lora_index]
            lora_index += 1

            attn.q = LoRA_PEFT(attn.q, w_a_linear_q, w_b_linear_q)
            attn.v = LoRA_PEFT(attn.v, w_a_linear_v, w_b_linear_v)

            if self.use_qkvo:
                w_a_linear_o = self.w_As[lora_index]
                w_b_linear_o = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_k = self.w_As[lora_index]
                w_b_linear_k = self.w_Bs[lora_index]
                lora_index += 1
                attn.o = LoRA_PEFT(attn.o, w_a_linear_o, w_b_linear_o)
                attn.k = LoRA_PEFT(attn.k, w_a_linear_k, w_b_linear_k)

    def finalize_arch(self):
        arch_weights = self.arch_weights
        self.w_As_final, self.w_Bs_final = [], []
        max_indices_layers = []
        indices_compute = []
        for t_layer_i in range(arch_weights.shape[0]):
            if self.iter_search:
                weight_layer = arch_weights[t_layer_i]
                weight_layer2 = self.arch_weights2[t_layer_i]
                max_indices_binary = torch.max(weight_layer, dim=-1).indices
                max_indices_dim = torch.max(weight_layer2, dim=-1).indices
                max_indices_dim, max_indices_binary = max_indices_dim.tolist(), max_indices_binary.tolist()
                final_indices = []
                for i in range(len(max_indices_binary)):
                    if max_indices_binary[i] > 0:
                        final_indices.append(max_indices_dim[i] + 1) # +1 for final dimension selection
                    else:
                        final_indices.append(0)
                indices_compute.append(final_indices)
                max_indices_layers.append(final_indices)
            else:
                weight_layer = arch_weights[t_layer_i]
                max_indices = torch.max(weight_layer, dim=-1).indices
                indices_compute.append(max_indices)
                max_indices = max_indices.tolist()
                max_indices_layers.append(max_indices)
        print("Finalized params: ", compute_search_size(indices_compute))

        lora_index = 0
        for t_layer_i, blk in enumerate(self.t5_model.encoder.block):
            attn = blk.layer[0].SelfAttention

            max_q, max_k, max_v, max_o = max_indices_layers[t_layer_i][:4]
            w_a_linear_q = self.w_As[lora_index]
            w_b_linear_q = self.w_Bs[lora_index]
            lora_index += 1
            w_a_linear_v = self.w_As[lora_index]
            w_b_linear_v = self.w_Bs[lora_index]
            lora_index += 1

            w_a_linear_q_final, w_b_linear_q_final = self.finalize_lora_weight(w_a_linear_q, w_b_linear_q, max_q)
            w_a_linear_v_final, w_b_linear_v_final = self.finalize_lora_weight(w_a_linear_v, w_b_linear_v, max_v)
            if self.retrain:
                self.w_As_final.extend([w_a_linear_q_final, w_a_linear_v_final])
                self.w_Bs_final.extend([w_b_linear_q_final, w_b_linear_v_final])

            attn.q = LoRA_PEFT(attn.q, w_a_linear_q_final, w_b_linear_q_final)
            attn.v = LoRA_PEFT(attn.v, w_a_linear_v_final, w_b_linear_v_final)

            if self.use_qkvo:
                w_a_linear_o = self.w_As[lora_index]
                w_b_linear_o = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_k = self.w_As[lora_index]
                w_b_linear_k = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_o_final, w_b_linear_o_final = self.finalize_lora_weight(w_a_linear_o, w_b_linear_o, max_o)
                w_a_linear_k_final, w_b_linear_k_final = self.finalize_lora_weight(w_a_linear_k, w_b_linear_k, max_k)
                attn.o = LoRA_PEFT(attn.o, w_a_linear_o_final, w_b_linear_o_final)
                attn.k = LoRA_PEFT(attn.k, w_a_linear_k_final, w_b_linear_k_final)

                if self.retrain:
                    self.w_As_final.extend([w_a_linear_o_final, w_a_linear_k_final])
                    self.w_Bs_final.extend([w_b_linear_o_final, w_b_linear_k_final])
            if self.search_all_lora:
                ffn = blk.layer[-1].DenseReluDense
                max_ffn1, max_ffn2 = max_indices_layers[t_layer_i][4:]

                w_a_linear_ffn1 = self.w_As[lora_index]
                w_b_linear_ffn1 = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_ffn2 = self.w_As[lora_index]
                w_b_linear_ffn2 = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_ffn1_final, w_b_linear_ffn1_final = self.finalize_lora_weight(w_a_linear_ffn1,
                                                                                         w_b_linear_ffn1, max_ffn1)
                w_a_linear_ffn2_final, w_b_linear_ffn2_final = self.finalize_lora_weight(w_a_linear_ffn2,
                                                                                         w_b_linear_ffn2, max_ffn2)
                ffn.wi = LoRA_PEFT(ffn.wi, w_a_linear_ffn1_final, w_b_linear_ffn1_final)
                ffn.wo = LoRA_PEFT(ffn.wo, w_a_linear_ffn2_final, w_b_linear_ffn2_final)

                if self.retrain:
                    self.w_As_final.extend([w_a_linear_ffn1_final, w_a_linear_ffn2_final])
                    self.w_Bs_final.extend([w_b_linear_ffn1_final, w_b_linear_ffn2_final])

        bia = len(self.t5_model.encoder.block)
        for t_layer_i, blk in enumerate(self.t5_model.decoder.block):
            t_layer_i += bia
            attn = blk.layer[0].SelfAttention

            max_q, max_k, max_v, max_o = max_indices_layers[t_layer_i][:4]
            w_a_linear_q = self.w_As[lora_index]
            w_b_linear_q = self.w_Bs[lora_index]
            lora_index += 1
            w_a_linear_v = self.w_As[lora_index]
            w_b_linear_v = self.w_Bs[lora_index]
            lora_index += 1

            w_a_linear_q_final, w_b_linear_q_final = self.finalize_lora_weight(w_a_linear_q, w_b_linear_q, max_q)
            w_a_linear_v_final, w_b_linear_v_final = self.finalize_lora_weight(w_a_linear_v, w_b_linear_v, max_v)

            attn.q = LoRA_PEFT(attn.q, w_a_linear_q_final, w_b_linear_q_final)
            attn.v = LoRA_PEFT(attn.v, w_a_linear_v_final, w_b_linear_v_final)

            if self.use_qkvo:
                w_a_linear_o = self.w_As[lora_index]
                w_b_linear_o = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_k = self.w_As[lora_index]
                w_b_linear_k = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_o_final, w_b_linear_o_final = self.finalize_lora_weight(w_a_linear_o, w_b_linear_o, max_o)
                w_a_linear_k_final, w_b_linear_k_final = self.finalize_lora_weight(w_a_linear_k, w_b_linear_k, max_k)
                attn.o = LoRA_PEFT(attn.o, w_a_linear_o_final, w_b_linear_o_final)
                attn.k = LoRA_PEFT(attn.k, w_a_linear_k_final, w_b_linear_k_final)

            if self.search_all_lora:
                ffn = blk.layer[-1].DenseReluDense
                max_ffn1, max_ffn2 = max_indices_layers[t_layer_i][4:]

                w_a_linear_ffn1 = self.w_As[lora_index]
                w_b_linear_ffn1 = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_ffn2 = self.w_As[lora_index]
                w_b_linear_ffn2 = self.w_Bs[lora_index]
                lora_index += 1
                w_a_linear_ffn1_final, w_b_linear_ffn1_final = self.finalize_lora_weight(w_a_linear_ffn1,
                                                                                         w_b_linear_ffn1, max_ffn1)
                w_a_linear_ffn2_final, w_b_linear_ffn2_final = self.finalize_lora_weight(w_a_linear_ffn2,
                                                                                         w_b_linear_ffn2, max_ffn2)
                ffn.wi = LoRA_PEFT(ffn.wi, w_a_linear_ffn1_final, w_b_linear_ffn1_final)
                ffn.wo = LoRA_PEFT(ffn.wo, w_a_linear_ffn2_final, w_b_linear_ffn2_final)

                if self.retrain:
                    self.w_As_final.extend([w_a_linear_ffn1_final, w_a_linear_ffn2_final])
                    self.w_Bs_final.extend([w_b_linear_ffn1_final, w_b_linear_ffn2_final])

    def modify_arch_mask(self, binary_stage=True):
        if binary_stage: #binary search stage
            dimension_weights = self.arch_weights2
            max_indices_temp = dimension_weights.argmax(dim=-1)
            dimension_weights = dirichlet.Dirichlet(F.elu(dimension_weights.clone()) + 1).sample()
            max_indices = dimension_weights.argmax(dim=-1)  # shape: [layers, possible_positions]
            if self.progressive_fix: # here if we need to use early-stop
                # replace the max_indices with search_mask if it is freezed
                max_indices[self.dimension_search_mask] = max_indices_temp[self.dimension_search_mask]
            self.dimension_mask = max_indices
            self.binary_mask = None
        else:
            binary_weights = self.arch_weights
            max_indices_temp = binary_weights.argmax(dim=-1)
            binary_weights = dirichlet.Dirichlet(F.elu(binary_weights.clone()) + 1).sample()
            max_indices = binary_weights.argmax(dim=-1) #shape: [layers, possible_positions]
            if self.progressive_fix: # here if we need to use early-stop
                # replace the max_indices with search_mask if it is freezed
                max_indices[self.binary_search_mask] = max_indices_temp[self.binary_search_mask]
            self.binary_mask = max_indices
            self.dimension_mask = None

    def init_gumbel_weights(self, epochs=100, eval_mode=False):
        arch_weights = self.arch_weights
        dimension_mask, binary_mask = None, None
        binary_search_mask, dimension_search_mask = None, None
        if self.iter_search:
            arch_weights2 = self.arch_weights2
            if self.iterative_order or self.main_forward:  # iterative_order=True, means binary search stage
                self.modify_arch_mask(binary_stage=True)
                dimension_mask, binary_mask = self.dimension_mask, None
                if self.progressive_fix:
                    binary_search_mask, dimension_search_mask = self.binary_search_mask, None
            else:
                self.modify_arch_mask(binary_stage=False)
                dimension_mask, binary_mask = None, self.binary_mask
                if self.progressive_fix:
                    binary_search_mask, dimension_search_mask = None, self.dimension_search_mask

        if self.use_search:
            temp = 5 - (5. - 1.) / 15. * epochs
            max_indices_comput = []
            gumbel_weights_all_layers = []

            if self.args.use_beta:
                if not self.iter_search:
                    beta_sample_weights = dirichlet.Dirichlet(F.elu(arch_weights.clone()) + 1).rsample()
                else:
                    if self.iterative_order or self.main_forward: #binary search stage
                        beta_sample_weights = dirichlet.Dirichlet(F.elu(arch_weights.clone()) + 1).rsample()
                    else:
                        beta_sample_weights = dirichlet.Dirichlet(F.elu(arch_weights2.clone()) + 1).rsample()

            for t_layer_i, blk in enumerate(self.t5_model.encoder.block):
                # print(arch_weights[t_layer_i].size(),"arch")
                if not eval_mode and not self.retrain:
                    if self.args.use_beta:
                        weight_layer = beta_sample_weights[t_layer_i]
                    else:
                        if not self.iter_search or self.iterative_order or self.main_forward:  # binary search stage
                            weight_layer = arch_weights[t_layer_i]
                        else:
                            weight_layer = arch_weights2[t_layer_i]
                    binary_mask_layer = binary_mask[t_layer_i] if binary_mask is not None else None
                    binary_search_mask_layer = binary_search_mask[t_layer_i] if binary_search_mask is not None else None
                    dimension_search_mask_layer = dimension_search_mask[t_layer_i] if dimension_search_mask is not None else None
                    gumbel_weights = bernoulli_sample(weight_layer, temp=temp, GumbleSoftmax=self.GumbleSoftmax,
                                                      use_beta=self.args.use_beta, binary_mask=binary_mask_layer,
                                                      binary_search_mask=binary_search_mask_layer,
                                                      dimension_search_mask=dimension_search_mask_layer)  # shape: [possible_location, candidate_dims]
                # If we only want few lora layer instead of all
                else:
                    # max_indices = torch.max(arch_weights[t_layer_i], dim=-1).indices
                    weight_layer = arch_weights[t_layer_i]
                    max_indices = torch.max(weight_layer, dim=-1).indices
                    max_indices_comput.append(torch.tensor(max_indices))
                    max_weights = torch.zeros_like(arch_weights[t_layer_i])
                    max_weights[np.arange(len(max_indices)), max_indices] = 1
                    gumbel_weights = max_weights
                gumbel_weights = gumbel_weights.cuda()

                gumbel_weights_all_layers.append(gumbel_weights)
            bia = len(self.t5_model.encoder.block)
            for t_layer_i, blk in enumerate(self.t5_model.decoder.block):
                # print(arch_weights[t_layer_i].size(),"arch")
                t_layer_i += bia
                if not eval_mode and not self.retrain:
                    if self.args.use_beta:
                        weight_layer = beta_sample_weights[t_layer_i]
                    else:
                        if not self.iter_search or self.iterative_order or self.main_forward:  # binary search stage
                            weight_layer = arch_weights[t_layer_i]
                        else:
                            weight_layer = arch_weights2[t_layer_i]
                    binary_mask_layer = binary_mask[t_layer_i] if binary_mask is not None else None
                    binary_search_mask_layer = binary_search_mask[t_layer_i] if binary_search_mask is not None else None
                    dimension_search_mask_layer = dimension_search_mask[
                        t_layer_i] if dimension_search_mask is not None else None
                    gumbel_weights = bernoulli_sample(weight_layer, temp=temp, GumbleSoftmax=self.GumbleSoftmax,
                                                      use_beta=self.args.use_beta, binary_mask=binary_mask_layer,
                                                      binary_search_mask=binary_search_mask_layer,
                                                      dimension_search_mask=dimension_search_mask_layer)  # shape: [possible_location, candidate_dims]
                # If we only want few lora layer instead of all
                else:
                    # max_indices = torch.max(arch_weights[t_layer_i], dim=-1).indices
                    #todo: add dimension mask
                    weight_layer = arch_weights[t_layer_i]
                    max_indices = torch.max(weight_layer, dim=-1).indices
                    max_indices_comput.append(torch.tensor(max_indices))
                    max_weights = torch.zeros_like(arch_weights[t_layer_i])
                    max_weights[np.arange(len(max_indices)), max_indices] = 1
                    gumbel_weights = max_weights
                gumbel_weights = gumbel_weights.cuda()

                gumbel_weights_all_layers.append(gumbel_weights)

            if (eval_mode or self.retrain) and self.print_eval:
                print("params after search: ", compute_search_size(max_indices_comput))
                self.print_eval = 0
            return gumbel_weights_all_layers if len(gumbel_weights_all_layers) else None

        else:
            print("no sampling lora")
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x, cur_epoch, eval_mode=False, main_forward=False) -> Tensor:
        self.main_forward = main_forward # main_forward: it means that this is not the forward for the "arch search"
        if eval_mode:
            self.main_forward = True
        if self.use_search and not self.retrain:
            gumbel_weights_all_layers = self.init_gumbel_weights(epochs=cur_epoch, eval_mode=eval_mode)
            # print("dimension maskk:", self.dimension_mask)
            loss = self.t5_model(**x, gumbel_weights=gumbel_weights_all_layers, dimension_mask=self.dimension_mask,
                                 iterative_order=self.iterative_order, main_forward=self.main_forward)
            if not self.main_forward:
                self.iterative_order = not self.iterative_order
        else:
            loss = self.t5_model(**x)
        return loss


def compute_search_size(max_indices):
    # print(max_indices)
    dims = [0, 1, 4, 8]
    # dims = [0, 8]
    num_params = 0
    for max_index in max_indices:
        if type(max_index) != list:
            max_index = max_index.tolist()
        for max_ in max_index[:4]:
            num_params += dims[max_] * 1024 * 2
        if len(max_index) > 4:
            for max_ in max_index[4:]:
                num_params += dims[max_] * 1024 + dims[max_] * 3072

    return num_params



if __name__ == "__main__":  # Debug
    img = torch.randn(2, 3, 224, 224)
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    lora_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10)
    pred = lora_vit(img)
    print(pred.shape)

    img = torch.randn(2*20, 3, 224, 224)
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    lora_vit = LoRA_ViT_timm(vit_model=model, r=4, num_classes=10)
    pred = lora_vit.forward3D(img)
    print(pred.shape)
