# Aofei Chang at April 22 2024
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.distributions import dirichlet

from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration
from gumbel_module import GumbleSoftmax, gumbel_sample_weight, measure_entropy, calculate_zeta_for_shifting, bernoulli_sample
from lora.peft_modules import LoRA_PEFT, Mix_PEFT, PrefixTuning, PrefixTuningSearch
from lora.forward_injection import set_lora_forward

from utils.utils import cosine_similarity, recognize_layer_id, recognize_module_weights_loc, calculate_DSI, get_top_k_modules



def weights(model: nn.Module):
    res = []
    for n, p in model.named_parameters():
        if 'arch' not in n and p.requires_grad == True:
            res.append(p)
        else:
            continue
    return res


class MoM_T5(nn.Module):
    def __init__(self, backbone:T5ForConditionalGeneration, r: int, model_config:T5Config, args=None):
        super(MoM_T5, self).__init__()

        assert r > 0
        self.r = r
        self.num_encoder_layers = model_config.num_layers
        self.num_decoder_layers = model_config.num_decoder_layers
        self.use_search = args.use_search
        self.iter_search = args.iter_search
        self.early_stop = args.early_stop
        self.use_beta = args.use_beta
        self.retrain = False
        if args is not None:
            self.retrain = args.retrain
        self.eval_mode = False
        self.search_mom = args.search_mom
        self.GumbleSoftmax = GumbleSoftmax()
        # dim = vit_model.head.in_features
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        self.w_As_final, self.w_Bs_final = [], []
        self.args=args
        self.all_epochs = args.epochs
        self.print_eval = True

        self.search_lora_dim = [0, 8]
        self.candidate_dims = [1, 4, 8]
        if args.large_sp:
            self.candidate_dims = [1, 2, 4, 8]
        if args.small_sp:
            self.candidate_dims = [1, 2]
        self.small_prefix = args.small_prefix
        self.fix_prefix_dim = args.fix_prefix_dim
        self.dim_then_binary, self.binary_then_dim = args.dim_then_binary, args.binary_then_dim #ablation study args
        self.no_gumbel = args.no_gumbel
        if not self.iter_search:
            self.candidate_dims = [0, 1, 4, 8]
        self.search_lora_dim_final = [0, 1, 4, 8]

        r = max(self.candidate_dims)
        self.r = r
        self.prefix_dim = r
        if self.fix_prefix_dim:
            self.prefix_dim = 8

        lora_possible_positions = 6
        self.num_options = lora_possible_positions
        if self.search_mom:
            self.num_options = self.num_options+2

        self.progressive_fix = args.progressive_fix
        print(f"progressively fix: {args.progressive_fix}")
        # parameters for progressively shrinking and budget control
        self.budget_abs = args.budget_abs
        print(f"budget_abs: {self.budget_abs}")
        self.use_budget = args.use_budget
        self.use_bitfit, self.use_lora, self.use_adapter, self.use_lnfit = args.use_bitfit, args.use_lora, args.use_adapter, args.use_lnfit
        self.use_PA, self.use_SA, self.use_prefix = args.use_PA, args.use_SA, args.use_prefix
        if self.use_search or self.use_prefix:
            set_lora_forward(backbone)

        self._init_arch_weight()
        self._insert_peft_modules(backbone=backbone, r=r)

        if self.use_search:
            if self.iter_search:
                self._arch_parameters = [
                    self.arch_weights_binary_encoder_matrix,
                    self.arch_weights_binary_decoder_matrix,
                    self.arch_weights_binary_encoder,
                    self.arch_weights_binary_decoder,
                    self.arch_weights_multi_encoder,
                    self.arch_weights_multi_decoder,
                    self.arch_weights_binary_final_norm
                ]
                if self.use_prefix and not self.early_stop:
                    self._arch_parameters.extend([self.arch_weights_binary_prefix, self.arch_weights_multi_prefix])
            else:
                self._arch_parameters = [
                    self.arch_weights_binary_encoder,
                    self.arch_weights_binary_decoder,
                    self.arch_weights_multi_encoder,
                    self.arch_weights_multi_decoder,
                    self.arch_weights_binary_final_norm
                ]

        # set new forward

        #those masks are for pruning
        self.encoder_matrix_binary_mask, self.encoder_vector_binary_mask, self.decoder_matrix_binary_mask, self.decoder_vector_binary_mask, self.final_norm_binary_mask = [None] * 5
        self.prefix_binary_mask = None
        if self.early_stop:
            self.encoder_matrix_binary_mask, self.encoder_vector_binary_mask, self.decoder_matrix_binary_mask, self.decoder_vector_binary_mask, self.final_norm_binary_mask \
                = (torch.ones(self.arch_weights_binary_encoder_matrix.shape[:-1], dtype=torch.int64), torch.ones(self.arch_weights_binary_encoder.shape[:-1], dtype=torch.int64),
                   torch.ones(self.arch_weights_binary_decoder_matrix.shape[:-1], dtype=torch.int64), torch.ones(self.arch_weights_binary_decoder.shape[:-1], dtype=torch.int64),
                   torch.ones(self.arch_weights_binary_final_norm.shape[:-1], dtype=torch.int64))
            if self.use_prefix:
                self.prefix_binary_mask = torch.ones(self.arch_weights_binary_prefix.shape[:-1], dtype=torch.int64)
        #shape:[layers, possible_positions]

        self.iterative_order = True
        # if self.early_stop:
        #     self.iterative_order = False
        if self.early_stop:
            self._init_early_stop_setting()

        self.max_prune_step = args.max_prune_steps
        self.beta1, self.beta2 = 0.85, 0.85
        
    def _init_early_stop_setting(self):
        self.prune_flag = False

        # num_matrix_modules_layer = self.arch_weights_binary_encoder_matrix.shape[1] + self.arch_weights_binary_decoder_matrix.shape[1]
        # num_vector_modules_layer = self.arch_weights_binary_encoder.shape[1] + self.arch_weights_binary_decoder.shape[1]
        num_matrix_modules = (
                    self.arch_weights_binary_encoder_matrix.shape[0] * self.arch_weights_binary_encoder_matrix.shape[1]
                    + self.arch_weights_binary_decoder_matrix.shape[0] * self.arch_weights_binary_decoder_matrix.shape[
                        1])
        num_vector_modules = (
                self.arch_weights_binary_encoder.shape[0] * self.arch_weights_binary_encoder.shape[1]
                + self.arch_weights_binary_decoder.shape[0] * self.arch_weights_binary_decoder.shape[1]
                + self.arch_weights_binary_final_norm.shape[0])
        # gradient storage, the sensitivity should >= 0
        self.accumulated_gradient_matrix = [torch.tensor(0, dtype=torch.float).cuda() for _ in
                                            range(num_matrix_modules)]
        self.accumulated_gradient_vector = [torch.tensor(0, dtype=torch.float).cuda() for _ in
                                            range(num_vector_modules)]
        self.PEFT_pruning_flag_matrix = [False for _ in range(num_matrix_modules)]  # matrix based modules
        self.PEFT_pruning_flag_vector = [False for _ in range(num_vector_modules)]  # vector (bias) based modules
        self.prune_dict = dict()
        self.gradient_records_dict = dict()
        self.val_gradient_records_dict = dict()
        self.train_gradient_records_dict = dict()
        self.sen_records_dict = dict()  # sensitivity records
        self.exp_avg_grad_records_dict = dict()
        self.exp_avg_unc_records_dict = dict()
        self.param_scale_dict = dict()
        self.module_id_dict = dict()
        self.id_module_dict = dict()
        module_id = 0
        # param scale list for matrix based modules
        self.matrix_based_params_mapped = [0] * self.arch_weights_binary_encoder_matrix.shape[1]
        self.vector_based_params_mapped_encoder = [0] * self.arch_weights_binary_encoder.shape[1]
        self.vector_based_params_mapped_decoder = [0] * self.arch_weights_binary_decoder.shape[1]
        self.vector_based_params_mapped_final_norm = [0] * self.arch_weights_binary_final_norm.shape[0]
        if self.use_prefix:
            self.prefix_params_mapped = [2 * 1024 * 2] * self.arch_weights_binary_prefix.shape[0]

        for name, param in self.t5_model.named_parameters():
            if param.requires_grad:
                sub_ = name.split(".")
                if 'gate' in name:
                    continue
                if 'Adapter' in sub_[-2] or 'LoRA' in sub_[-2]:
                    name = '.'.join(sub_[:-2])
                elif 'prefix' in sub_[-2] and 'up' in sub_[-2] and 'gate' not in name:
                    name = '.'.join(sub_[:-1])
                elif 'sadapter' in sub_[-3] or 'padapter' in sub_[-3]:
                    name = '.'.join(sub_[:-2])
                self.prune_dict[name] = False
                if self.param_scale_dict.__contains__(name):
                    self.param_scale_dict[name] += param.numel()  # simply contains all parameters
                else:
                    self.module_id_dict[name] = module_id
                    self.id_module_dict[module_id] = name
                    module_id += 1
                    self.param_scale_dict[name] = param.numel()
                self.gradient_records_dict[name] = torch.tensor(0, dtype=torch.float).cuda()
                self.val_gradient_records_dict[name] = []
                self.train_gradient_records_dict[name] = []
                self.sen_records_dict[name] = torch.tensor(0, dtype=torch.float).cuda()
                self.exp_avg_grad_records_dict[name] = torch.tensor(0, dtype=torch.float).cuda()
                self.exp_avg_unc_records_dict[name] = torch.tensor(0, dtype=torch.float).cuda()
        self.modules_number = module_id
        print(self.param_scale_dict)
        self.param_scale_list = [0] * self.modules_number
        self.gradient_records_list = [None] * self.modules_number
        self.prune_records_list = [False] * self.modules_number
        for (n, s) in self.param_scale_dict.items():
            module_id = self.module_id_dict[n]
            self.param_scale_list[module_id] = s
            # formulate a param scale list for param expectation calculatio
        for name in self.param_scale_dict:
            if "prefix" in name:
                continue
            layer_id, loc, encoder_flag, matrix_flag, final_norm_flag = recognize_module_weights_loc(name)
            # if layer_id >= 1:
            #     break
            if loc is not None:
                if matrix_flag:
                    self.matrix_based_params_mapped[loc] = self.param_scale_dict[name]
                else:
                    if encoder_flag:
                        self.vector_based_params_mapped_encoder[loc] = self.param_scale_dict[name]
                    else:
                        self.vector_based_params_mapped_decoder[loc] = self.param_scale_dict[name]
        for name in self.param_scale_dict:
            if "final_layer_norm" not in name:
                continue
            layer_id, loc, encoder_flag, matrix_flag, final_norm_flag = recognize_module_weights_loc(name)
            if final_norm_flag:
                self.vector_based_params_mapped_final_norm[loc] = self.param_scale_dict[name]
            # self.matrix_based_params_mapped = self.matrix_based_params_mapped.unsqueeze()
        self.module_rank_records = []
        self._init_dimension_pruning()

    def _init_dimension_pruning(self):
        self.dimension_fix_flag = [[False for _ in range(self.arch_weights_multi_encoder.shape[1])] for _ in
                                   range(self.num_encoder_layers + self.num_decoder_layers)]
        self.dimension_fix_mask = torch.zeros(len(self.dimension_fix_flag), len(self.dimension_fix_flag[0]))
        self.dimension_weight_history = [] # T items, T is the pruning interval


    def update_dimension_pruning(self):
        if self.main_forward:
            cur_weight = torch.cat((self.arch_weights_multi_encoder, self.arch_weights_multi_decoder), dim=0)
            self.dimension_weight_history.append(cur_weight)

    def fix_dimensions(self):
        fixed_indices = None
        #update the binary mask condition here
        encoder_layers, nums = self.encoder_matrix_binary_mask.shape[0], self.encoder_matrix_binary_mask.shape[1]
        for i in range(encoder_layers):
            for j in range(nums):
                prune_status = self.encoder_matrix_binary_mask[i][j] < 1
                if prune_status:
                    self.dimension_fix_flag[i][j] = True
        decoder_layers, nums = self.decoder_matrix_binary_mask.shape[0], self.decoder_matrix_binary_mask.shape[1]
        for i in range(decoder_layers):
            for j in range(nums):
                prune_status = self.decoder_matrix_binary_mask[i][j] < 1
                if prune_status:
                    self.dimension_fix_flag[i+encoder_layers][j] = True
        self.dimension_fix_mask = torch.tensor(self.dimension_fix_flag, dtype=torch.float)

        if len(self.dimension_weight_history) >= 2:
            start_dimension_dist = self.dimension_weight_history[0].argmax(dim=-1).flatten().detach().cpu()
            end_dimension_dist = self.dimension_weight_history[-1].argmax(dim=-1).flatten().detach().cpu()
            global_stability = cosine_similarity(start_dimension_dist, end_dimension_dist)
            not_fixed_module_number = sum([len(self.dimension_fix_flag[i]) - sum(self.dimension_fix_flag[i]) for i in range(len(self.dimension_fix_flag))])  # M^z
            fix_number_this_step = 0
            if self.max_prune_step:
                if self.max_prune_step <= 5:
                    fix_number_this_step = not_fixed_module_number
                else:
                    if math.isnan(global_stability) or global_stability is np.nan:
                        global_stability = 1
                    fix_number_this_step = (global_stability * not_fixed_module_number) / self.max_prune_step
                    # fix_number_this_step = not_fixed_module_number / self.max_prune_step
                    if isinstance(fix_number_this_step, torch.Tensor):
                        fix_number_this_step = fix_number_this_step.item()
            print("Fix at this step:", fix_number_this_step, "not fixed:", not_fixed_module_number, "")
            DSI_metric = calculate_DSI(torch.stack(self.dimension_weight_history, dim=0))  # in shape [layers, modules]
            masked_DSI_metric = (999 + torch.zeros_like(DSI_metric)) * self.dimension_fix_mask + (1 - self.dimension_fix_mask) * DSI_metric
            # then take the top fix_number, get the index to fix
            if math.isnan(fix_number_this_step):
                fix_number_this_step = 0
                # print("nan", global_stability, not_fixed_module_number, self.dimension_fix_flag, self.max_prune_step)
            top_k = math.floor(fix_number_this_step)
            fixed_indices = get_top_k_modules(masked_DSI_metric, top_k=top_k)
            for (i, j) in fixed_indices:
                self.dimension_fix_flag[i][j] = True

            self.dimension_fix_mask = torch.tensor(self.dimension_fix_flag, dtype=torch.float)
            #reset history
            self.dimension_weight_history = []



        return fixed_indices


    def _init_arch_weight(self):
        encoder_binary_num_per_layer = 6 + 2 + 2 + 8 - 6 - 2 #lora, adapter, layer_norm, bitfit
        encoder_multi_num_per_layer = 6 + 2 #lora, adapter
        if self.use_PA and self.use_SA:
            encoder_multi_num_per_layer = 6 + 2 + 2
        decoder_binary_num_per_layer = 6 + 2 + 3 + 9 - 6 - 2  # lora, adapter, layer_norm, bitfit
        decoder_multi_num_per_layer = 6 + 2  # lora, adapter
        if self.use_PA and self.use_SA:
            decoder_multi_num_per_layer = 6 + 2 + 2
        final_norm_binary_num = 2 + 2 #final_layer_norm, bitfit, lnnorm

        self.arch_weights_binary_prefix, self.arch_weights_multi_prefix = None, None
        if self.use_prefix:
            self.arch_weights_binary_prefix = nn.Parameter(
                1e-3 * torch.randn(self.num_encoder_layers+self.num_decoder_layers, 2, dtype=torch.float32))
            self.arch_weights_multi_prefix = torch.randn(self.num_encoder_layers + self.num_decoder_layers, len(self.candidate_dims),
                                   dtype=torch.float32)
            if self.fix_prefix_dim:
                fixed_dims = [0 for _ in range(len(self.candidate_dims)-1)] + [1]
                temp = torch.tensor(fixed_dims).unsqueeze(0).expand_as(self.arch_weights_multi_prefix).cuda()
                self.arch_weights_multi_prefix = temp.type_as(self.arch_weights_binary_prefix)
            else:
                self.arch_weights_multi_prefix = nn.Parameter(
                    1e-3 * torch.randn(self.num_encoder_layers + self.num_decoder_layers, len(self.candidate_dims),
                                       dtype=torch.float32))
        
        # num_layes = self.num_decoder_layers + self.num_encoder_layers
        if self.iter_search:
            self.arch_weights_binary_encoder_matrix = nn.Parameter(
                1e-3 * torch.randn(self.num_encoder_layers, encoder_multi_num_per_layer, 2, dtype=torch.float32))
            self.arch_weights_binary_decoder_matrix = nn.Parameter(
                1e-3 * torch.randn(self.num_decoder_layers, decoder_multi_num_per_layer, 2, dtype=torch.float32))

            self.arch_weights_binary_encoder = nn.Parameter(
                1e-3 * torch.randn(self.num_encoder_layers, encoder_binary_num_per_layer, 2, dtype=torch.float32))
            self.arch_weights_binary_decoder = nn.Parameter(
                1e-3 * torch.randn(self.num_decoder_layers, decoder_binary_num_per_layer, 2, dtype=torch.float32))

            self.arch_weights_multi_encoder = nn.Parameter(
                1e-3 * torch.randn(self.num_encoder_layers, encoder_multi_num_per_layer, len(self.candidate_dims), dtype=torch.float32))
            self.arch_weights_multi_decoder = nn.Parameter(
                1e-3 * torch.randn(self.num_decoder_layers, decoder_multi_num_per_layer, len(self.candidate_dims), dtype=torch.float32))

            self.arch_weights_binary_final_norm = nn.Parameter(
                1e-3 * torch.randn(final_norm_binary_num, 2, dtype=torch.float32))

            self.binary_search_mask, self.dimension_search_mask = None, None
            # if self.progressive_fix:
            #     self.max_records = torch.zeros_like(self.arch_weights)
            #     self.max_records2 = torch.zeros_like(self.arch_weights2)
            #     self.cur_max_arch, self.cur_max_arch2 = None, None
            #     self.binary_search_mask = torch.zeros(num_layes, self.num_options)
            #     self.dimension_search_mask = torch.zeros(num_layes, self.num_options)
        else:
            self.arch_weights_binary_encoder = nn.Parameter(
                1e-3 * torch.randn(self.num_encoder_layers, encoder_binary_num_per_layer, 2, dtype=torch.float32))
            self.arch_weights_binary_decoder = nn.Parameter(
                1e-3 * torch.randn(self.num_decoder_layers, decoder_binary_num_per_layer, 2, dtype=torch.float32))

            self.arch_weights_multi_encoder = nn.Parameter(
                1e-3 * torch.randn(self.num_encoder_layers, encoder_multi_num_per_layer, len(self.candidate_dims),
                                   dtype=torch.float32))
            self.arch_weights_multi_decoder = nn.Parameter(
                1e-3 * torch.randn(self.num_decoder_layers, decoder_multi_num_per_layer, len(self.candidate_dims),
                                   dtype=torch.float32))

            self.arch_weights_binary_final_norm = nn.Parameter(
                1e-3 * torch.randn(final_norm_binary_num, 2, dtype=torch.float32))

        self.binary_mask_encoder = torch.zeros(self.num_encoder_layers, encoder_multi_num_per_layer)  # for binary selection
        self.binary_mask_decoder = torch.zeros(self.num_decoder_layers, decoder_multi_num_per_layer)  # for binary selection
        self.binary_mask_prefix = torch.zeros(self.num_encoder_layers+self.num_decoder_layers)  # for binary selection

        self.dimension_mask_encoder = torch.zeros(self.num_encoder_layers, encoder_multi_num_per_layer)  # for binary selection
        self.dimension_mask_decoder = torch.zeros(self.num_decoder_layers, decoder_multi_num_per_layer)  # for binary selection
        self.dimension_mask_prefix = torch.zeros(self.num_encoder_layers+self.num_decoder_layers)  # for binary selection

    def _insert_peft_modules(self, backbone, r):
        for param in backbone.parameters():
            param.requires_grad = False


        # Here, we INSERT the lora modules, and other PEFT modules
        # PEFT_config = {
        #     "lora": "self.attn: [Q, K, V, O], ffn:[wi, wo], number: 48*6",
        #     "LNfit": "[self.attn, cross.attn, ffn, final_layer_norm], number: 48 + 24 + 48 + 2 = 122",
        #     "bitfit": "self.attn: [Q, K, V, O], ffn:[wi, wo], layer_norm: all LNfit. number: 48*6+122",
        #     "LR-Adapter": "after attn, after ffn. number: 48*2"
        #  }
        for t_layer_i, blk in enumerate(backbone.encoder.block):
            attn = blk.layer[0].SelfAttention
            if self.use_prefix:
                attn.prefix_gate = torch.nn.Parameter(torch.zeros(1))
            layer_norm = blk.layer[0].layer_norm
            ffn = blk.layer[-1].DenseReluDense
            d_model = attn.d_model
            inner_dim = attn.inner_dim
            ffn_dim = ffn.wi.out_features
            in_dim, out_dim = ffn.wi.in_features, ffn.wo.out_features
            w_a_linear_q, w_a_linear_v, w_a_linear_o, w_a_linear_k, w_a_linear_ffn1, w_a_linear_ffn2 = [None]*6
            w_b_linear_q, w_b_linear_v, w_b_linear_o, w_b_linear_k, w_b_linear_ffn1, w_b_linear_ffn2 = [None]*6
            #attention config
            if self.use_lora:
                w_a_linear_q = nn.Linear(d_model, r, bias=False)
                w_b_linear_q = nn.Linear(r, inner_dim, bias=False)
                w_a_linear_v = nn.Linear(d_model, r, bias=False)
                w_b_linear_v = nn.Linear(r, inner_dim, bias=False)
                w_a_linear_o = nn.Linear(inner_dim, r, bias=False)
                w_b_linear_o = nn.Linear(r, d_model, bias=False)
                w_a_linear_k = nn.Linear(d_model, r, bias=False)
                w_b_linear_k = nn.Linear(r, inner_dim, bias=False)
                w_a_linear_ffn1 = nn.Linear(in_dim, r, bias=False)
                w_b_linear_ffn1 = nn.Linear(r, ffn_dim, bias=False)
                w_a_linear_ffn2 = nn.Linear(ffn_dim, r, bias=False)
                w_b_linear_ffn2 = nn.Linear(r, out_dim, bias=False)
                self.w_As.extend([w_a_linear_q, w_a_linear_v, w_a_linear_o, w_a_linear_k, w_a_linear_ffn1, w_a_linear_ffn2])
                self.w_Bs.extend([w_b_linear_q, w_b_linear_v, w_b_linear_o, w_b_linear_k, w_b_linear_ffn1, w_b_linear_ffn2])
            # attn.q = LoRA_PEFT(attn.q, w_a_linear_q, w_b_linear_q)
            # attn.v = LoRA_PEFT(attn.v, w_a_linear_v, w_b_linear_v)
            # print(self.candidate_dims,'s')
            attn.q = Mix_PEFT(attn.q, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_q, w_b_linear_q], hidden_dim=d_model, candidate_dims=self.candidate_dims)
            attn.v = Mix_PEFT(attn.v, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_v, w_b_linear_v], hidden_dim=d_model, candidate_dims=self.candidate_dims)
            attn.o = Mix_PEFT(attn.o, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_o, w_b_linear_o], hidden_dim=d_model, candidate_dims=self.candidate_dims)
            attn.k = Mix_PEFT(attn.k, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_k, w_b_linear_k], hidden_dim=d_model, candidate_dims=self.candidate_dims)
            ffn.wi = Mix_PEFT(ffn.wi, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_ffn1, w_b_linear_ffn1], hidden_dim=ffn_dim, candidate_dims=self.candidate_dims)
            ffn.wo = Mix_PEFT(ffn.wo, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_ffn2, w_b_linear_ffn2], hidden_dim=d_model, candidate_dims=self.candidate_dims)
            blk.layer[0].SelfAttention = Mix_PEFT(attn, add_adapter=self.use_adapter, hidden_dim=d_model, name='adapter', super_rank=r,
                                                  candidate_dims=self.candidate_dims, args=self.args, is_main_module=True)
            blk.layer[-1].DenseReluDense = Mix_PEFT(ffn, add_adapter=self.use_adapter, add_SA=self.use_SA, add_PA=self.use_PA, hidden_dim=d_model, name='adapter', super_rank=r,
                                                    candidate_dims=self.candidate_dims, args=self.args, is_main_module=True)
            blk.layer[-1].layer_norm = Mix_PEFT(blk.layer[-1].layer_norm, add_bitfit=self.use_bitfit,
                                                add_lnfit=self.use_lnfit, hidden_dim=d_model, args=self.args)
            blk.layer[0].layer_norm = Mix_PEFT(layer_norm, add_bitfit=self.use_bitfit, add_lnfit=self.use_lnfit, hidden_dim=d_model, args=self.args)

        for t_layer_i, blk in enumerate(backbone.decoder.block):
            attn = blk.layer[0].SelfAttention
            if self.use_prefix:
                attn.prefix_gate = torch.nn.Parameter(torch.zeros(1))
            # cross_attn = blk.layer[1].EncDecAttention
            layer_norm = blk.layer[0].layer_norm
            ffn = blk.layer[-1].DenseReluDense
            d_model = attn.d_model
            inner_dim = attn.inner_dim
            ffn_dim = ffn.wi.out_features
            in_dim, out_dim = ffn.wi.in_features, ffn.wo.out_features
            w_a_linear_q, w_a_linear_v, w_a_linear_o, w_a_linear_k, w_a_linear_ffn1, w_a_linear_ffn2 = [None] * 6
            w_b_linear_q, w_b_linear_v, w_b_linear_o, w_b_linear_k, w_b_linear_ffn1, w_b_linear_ffn2 = [None] * 6
            # attention config
            if self.use_lora:
                w_a_linear_q = nn.Linear(d_model, r, bias=False)
                w_b_linear_q = nn.Linear(r, inner_dim, bias=False)
                w_a_linear_v = nn.Linear(d_model, r, bias=False)
                w_b_linear_v = nn.Linear(r, inner_dim, bias=False)
                w_a_linear_o = nn.Linear(inner_dim, r, bias=False)
                w_b_linear_o = nn.Linear(r, d_model, bias=False)
                w_a_linear_k = nn.Linear(d_model, r, bias=False)
                w_b_linear_k = nn.Linear(r, inner_dim, bias=False)
                w_a_linear_ffn1 = nn.Linear(in_dim, r, bias=False)
                w_b_linear_ffn1 = nn.Linear(r, ffn_dim, bias=False)
                w_a_linear_ffn2 = nn.Linear(ffn_dim, r, bias=False)
                w_b_linear_ffn2 = nn.Linear(r, out_dim, bias=False)
                self.w_As.extend(
                    [w_a_linear_q, w_a_linear_v, w_a_linear_o, w_a_linear_k, w_a_linear_ffn1, w_a_linear_ffn2])
                self.w_Bs.extend(
                    [w_b_linear_q, w_b_linear_v, w_b_linear_o, w_b_linear_k, w_b_linear_ffn1, w_b_linear_ffn2])
            # attn.q = LoRA_PEFT(attn.q, w_a_linear_q, w_b_linear_q)
            # attn.v = LoRA_PEFT(attn.v, w_a_linear_v, w_b_linear_v)
            attn.q = Mix_PEFT(attn.q, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_q, w_b_linear_q], hidden_dim=d_model, candidate_dims=self.candidate_dims)
            attn.v = Mix_PEFT(attn.v, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_v, w_b_linear_v], hidden_dim=d_model, candidate_dims=self.candidate_dims)
            attn.o = Mix_PEFT(attn.o, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_o, w_b_linear_o], hidden_dim=d_model, candidate_dims=self.candidate_dims)
            attn.k = Mix_PEFT(attn.k, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args,
                              lora_modules=[w_a_linear_k, w_b_linear_k], hidden_dim=d_model, candidate_dims=self.candidate_dims)
            ffn.wi = Mix_PEFT(ffn.wi, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args, candidate_dims=self.candidate_dims,
                              lora_modules=[w_a_linear_ffn1, w_b_linear_ffn1], hidden_dim=ffn_dim)
            ffn.wo = Mix_PEFT(ffn.wo, add_lora=self.use_lora, add_bitfit=self.use_bitfit, args=self.args, candidate_dims=self.candidate_dims,
                              lora_modules=[w_a_linear_ffn2, w_b_linear_ffn2], hidden_dim=d_model)
            blk.layer[0].SelfAttention = Mix_PEFT(attn, add_adapter=self.use_adapter, hidden_dim=d_model, name='adapter', super_rank=r,
                                                  candidate_dims=self.candidate_dims, args=self.args, is_main_module=True)
            blk.layer[-1].DenseReluDense = Mix_PEFT(ffn, add_adapter=self.use_adapter, hidden_dim=d_model, add_SA=self.use_SA, add_PA=self.use_PA, name='adapter', super_rank=r,
                                                    candidate_dims=self.candidate_dims, args=self.args, is_main_module=True)
            blk.layer[0].layer_norm = Mix_PEFT(layer_norm, add_bitfit=self.use_bitfit, add_lnfit=self.use_lnfit, hidden_dim=d_model, args=self.args)
            blk.layer[-1].layer_norm = Mix_PEFT(blk.layer[-1].layer_norm, add_bitfit=self.use_bitfit, add_lnfit=self.use_lnfit, hidden_dim=d_model, args=self.args)
            #different with encoder, we need to tune layer norm in decoder
            blk.layer[1].layer_norm = Mix_PEFT(blk.layer[1].layer_norm, add_bitfit=self.use_bitfit, add_lnfit=self.use_lnfit, hidden_dim=d_model, args=self.args)
        #finally, tune the final_layer_norm
        backbone.encoder.final_layer_norm = Mix_PEFT(backbone.encoder.final_layer_norm, add_bitfit=self.use_bitfit, add_lnfit=self.use_lnfit, hidden_dim=d_model, args=self.args)
        backbone.decoder.final_layer_norm = Mix_PEFT(backbone.decoder.final_layer_norm, add_bitfit=self.use_bitfit, add_lnfit=self.use_lnfit, hidden_dim=d_model, args=self.args)

        if self.use_prefix:
            if self.early_stop:

                backbone.prefix_module = PrefixTuningSearch(n_layers=self.num_encoder_layers + self.num_decoder_layers,
                                                      input_size=d_model, prefix_length=self.prefix_dim,
                                                      candidate_dims=self.candidate_dims, small_prefix=self.small_prefix)
            else:
                backbone.prefix_module = PrefixTuning(n_layers=self.num_encoder_layers+self.num_decoder_layers,
                                                  input_size=d_model, prefix_length=self.prefix_dim, candidate_dims=self.candidate_dims)

        self.reset_parameters()
        self.t5_model = backbone

        for name, param in self.t5_model.named_parameters():
            if "LoRA" in name or "LNfit" in name or "Adapter" in name or "BitFit" in name or "sadapter" in name or "padapter" in name or "prefix" in name:
                param.requires_grad = True
                # print(name)
            else:
                param.requires_grad = False

    def arch_parameters(self):
        return self._arch_parameters

    def map_pruning_id_to_arch(self, idx):
        module_name = self.id_module_dict[idx]
        layer_id, encoder_flag, final_layer_norm_flag, matrix_flag, loc = recognize_layer_id(module_name)
        # layer_id, loc, encoder_flag, final_layer_norm_flag
        if "prefix" in module_name:
            return "prefix_binary", layer_id, None
        if final_layer_norm_flag:
            weight_matrix_name = 'final_norm_binary'
        elif encoder_flag:
            if matrix_flag:
                weight_matrix_name = 'encoder_matrix_bianry'
            else:
                weight_matrix_name = 'encoder_bianry'
        else:
            if matrix_flag:
                weight_matrix_name = 'decoder_matrix_bianry'
            else:
                weight_matrix_name = 'decoder_bianry'

        return weight_matrix_name, layer_id, loc

    def update_prune_mask(self):
        # id -> weight location
        weight_matrix_mask_dict = {
            "encoder_matrix_bianry": self.encoder_matrix_binary_mask,
            "encoder_bianry": self.encoder_vector_binary_mask,
            "decoder_matrix_bianry": self.decoder_matrix_binary_mask,
            "decoder_bianry": self.decoder_vector_binary_mask,
            "final_norm_binary": self.final_norm_binary_mask,
            "prefix_binary": self.prefix_binary_mask
        }
        for (idx, prune_flag) in enumerate(self.prune_records_list):
            (weight_matrix_name, layer, loc) = self.map_pruning_id_to_arch(idx)
            prune_status = self.prune_records_list[idx]
            prune_status = 0 if prune_status else 1
            if "final_norm" not in weight_matrix_name and 'prefix' not in weight_matrix_name:
                weight_matrix_mask_dict[weight_matrix_name][layer][loc] = prune_status
            elif "prefix" in weight_matrix_name:
                weight_matrix_mask_dict[weight_matrix_name][layer] = prune_status
            else:
                weight_matrix_mask_dict[weight_matrix_name][loc] = prune_status

    def update_grad(self):
        #sensitivity records: self.exp_avg_grad_records_dict
        #uncertainty records: self.exp_avg_unc_records_dict
        # if self.use_prefix:
        #     prefix_grad_dict = self.t5_model.prefix_module.upgrade_sub_grad()
        for name, param in self.t5_model.named_parameters():
            # if "prefix" in name:
            #     continue
            if param.requires_grad:
                sub_ = name.split(".")
                parent_name = name
                if 'gate' in name:
                    continue
                if 'Adapter' in sub_[-2] or 'LoRA' in sub_[-2]:
                    parent_name = '.'.join(sub_[:-2])
                elif 'prefix' in sub_[-2] and 'up' in sub_[-2] and 'gate' not in name:
                    parent_name = '.'.join(sub_[:-1])
                elif len(sub_) > 3 and 'sadapter' in sub_[-3] or 'padapter' in sub_[-3]:
                    parent_name = '.'.join(sub_[:-2])
                # if 'adapter' in sub_[-2] or 'Adapter' in sub_[-2] or 'LoRA' in sub_[-2]:
                #     parent_name = '.'.join(sub_[:-2])
                if self.prune_dict[parent_name] == False:
                    new_grad = None
                    new_grad_sum = None
                    if param.grad is not None:
                        new_grad = (param * param.grad).detach()
                        if self.args.no_abs_grad:
                            new_grad_sum = (-1) * new_grad.sum()
                        else:
                            new_grad_sum = new_grad.abs().sum()

                    if new_grad is not None:
                        if not self.main_forward:
                            self.val_gradient_records_dict[parent_name].append(new_grad)
                        else:
                            self.gradient_records_dict[parent_name] += new_grad_sum
                            self.train_gradient_records_dict[parent_name].append(new_grad)
        # print('fnfk', self.gradient_records_dict)
        val_gradients = list(self.val_gradient_records_dict.items())
        train_gradients = list(self.train_gradient_records_dict.items())

        flag_update = len(train_gradients[0][1]) > 0 and len(val_gradients[0][1]) > 0
        if flag_update:
            for i, (n, s) in enumerate(self.gradient_records_dict.items()):
                # update the module gradient and prune records list
                train_grad, val_grad = train_gradients[i][1], val_gradients[i][1]
                grad_cos = []
                val_gradient = torch.tensor(0, dtype=torch.float).cuda()
                for j in range(len(train_grad)):
                    cos = F.cosine_similarity(train_grad[j].flatten(), val_grad[j].flatten(), dim=0)
                    grad_cos.append(cos)
                    if self.args.no_abs_grad:
                        val_gradient += -1 * val_grad[j].sum()
                    else:
                        val_gradient += val_grad[j].abs().sum()
                avg_cos = 0
                if len(grad_cos) > 0:
                    avg_cos = sum(grad_cos) / len(grad_cos)
                    # print(avg_cos,'cos')

                module_id = self.module_id_dict[n]
                prune_flag = self.prune_dict[n]
                if self.args.prune_criterion == "tra":
                    new_grad = self.gradient_records_dict[n] / self.param_scale_dict[n]
                elif self.args.prune_criterion == "val":
                    new_grad = val_gradient / self.param_scale_dict[n]
                elif self.args.prune_criterion == "tra_val":
                    new_grad = (self.gradient_records_dict[n] / self.param_scale_dict[n] + val_gradient / self.param_scale_dict[n])
                elif self.args.prune_criterion == "tra_val_cos1":
                    new_grad = self.gradient_records_dict[n] / self.param_scale_dict[n] + avg_cos * val_gradient / self.param_scale_dict[n]
                else:
                    new_grad = avg_cos * (self.gradient_records_dict[n] / self.param_scale_dict[n] + val_gradient / self.param_scale_dict[n])

                self.gradient_records_dict[n] = torch.tensor(0, dtype=torch.float).cuda()
                # if new_grad > 0: # consider the case that, the module is not selected, he gradient will be 0
                self.exp_avg_grad_records_dict[n] = self.beta1 * self.exp_avg_grad_records_dict[n] + (1 - self.beta1) * new_grad
                unc_step = (new_grad - self.exp_avg_grad_records_dict[n]).abs()
                self.exp_avg_unc_records_dict[n] = self.beta2 * self.exp_avg_unc_records_dict[n] + (1 - self.beta2) * unc_step

                # new_sensitivity = self.exp_avg_unc_records_dict[n] * self.exp_avg_grad_records_dict[n]
                new_sensitivity = self.exp_avg_grad_records_dict[n]
                # self.gradient_records_list[module_id] = s
                self.gradient_records_list[module_id] = new_sensitivity
                self.prune_records_list[module_id] = prune_flag
                self.sen_records_dict[n] = new_sensitivity
            # reset value
            for n in self.train_gradient_records_dict:
                self.train_gradient_records_dict[n] = []
                self.val_gradient_records_dict[n] = []

    def select_top_gradient_modules(self, budget):
        param_number_list = self.param_scale_list
        gradients = self.gradient_records_list
        if None in gradients:
            return None
        gradients_mask = [
            torch.tensor(-999, dtype=torch.float).cuda() if i else torch.tensor(0, dtype=torch.float).cuda() for i
            in self.prune_records_list]
        masked_gradients = [a + b for a, b in zip(gradients, gradients_mask)]
        # add other modules later
        param_numbers = param_number_list
        # Pair module index with gradient and number of parameters, then sort by gradient descending
        modules = sorted(enumerate(zip(masked_gradients, param_numbers)), key=lambda x: x[1][0], reverse=True)
        selected_modules = [0] * len(gradients)  # Initialize the one-hot list for selected modules
        current_budget = 0

        for idx, (gradient, param_number) in modules:
            if current_budget + param_number <= budget:
                selected_modules[idx] = 1
                current_budget += param_number
            # Stop adding if the next module exceeds the budget
            if current_budget >= budget:
                break

        return selected_modules

    def prune_trigger(self):
        selected_top_modules = self.select_top_gradient_modules(budget=self.budget_abs)
        if selected_top_modules is not None:
            cos_records = []
            if len(list(self.module_rank_records)) <= 5:
                self.prune_flag = False
            else:
                for record in list(self.module_rank_records)[-5:]:
                    cos = cosine_similarity(selected_top_modules, record)
                    cos_records.append(cos)
                avg_cos = 0
                if len(cos_records):
                    avg_cos = sum(cos_records) / len(cos_records)
                self.prune_flag = False
                if avg_cos >= self.args.prune_threshold:
                    self.prune_flag = True
            self.module_rank_records.append(selected_top_modules)

    def get_param_expectation(self):
        # first version, only consider the binary choice, without dimension expectation (max dimension)
        # all_expected_params = 0
        # for i in range(self.modules_number):
        #     if not self.prune_records_list[i]:
        #         all_expected_params += self.param_scale_list[i]
        # second version
        # final_weight = ~mask * softmax_weight + mask * max_weight
        all_matrix_weights = torch.cat((self.arch_weights_multi_encoder, self.arch_weights_multi_decoder), dim=0).detach().cpu()
        # all_binary_matrix_weights = torch.stack((self.arch_weights_binary_encoder_matrix, self.arch_weights_binary_decoder_matrix), dim=0)
        max_matrix_dimension_weight = self.get_max_weight(all_matrix_weights)
        expanded_matrix_params = torch.tensor(self.matrix_based_params_mapped).unsqueeze(-1).repeat(1, len(self.candidate_dims)) * torch.tensor(self.candidate_dims) / self.candidate_dims[-1]
        # expanded_matrix_params in shape [modules, candidate_dims]
        softmax_dimension_weight = F.softmax(all_matrix_weights, dim=-1)

        # final_matrix_weight in shape [layers, modules, candidate_dims]
        final_matrix_weight = self.dimension_fix_mask.unsqueeze(-1) * max_matrix_dimension_weight + (1 - self.dimension_fix_mask).unsqueeze(-1) * softmax_dimension_weight
        # here we also need to consider the binary weights for matrix weight, self.encoder_matrix_binary_mask in shape [layers, modules]
        matrix_binary_mask = torch.cat((self.encoder_matrix_binary_mask, self.decoder_matrix_binary_mask), dim=0)
        matrix_bianry_arch_weight = torch.cat((F.softmax(self.arch_weights_binary_encoder_matrix,dim=-1)[:,:, 1], F.softmax(self.arch_weights_binary_decoder_matrix,dim=-1)[:,:, 1]), dim=0)
        # print(matrix_bianry_arch_weight.size(), matrix_binary_mask.size())
        # softmax_encoder_multi_binary, softmax_decoder_multi_binary = F.softmax(self.arch_weights_binary_encoder_matrix,dim=-1).detach().cpu(), F.softmax(
        #     self.arch_weights_binary_decoder_matrix, dim=-1).detach().cpu()
        # in fact we do not need to multiply with the binary weight
        # softmax_multi_binary = torch.cat((softmax_encoder_multi_binary, softmax_decoder_multi_binary), dim=0)
        # matrix_param_exp = (final_matrix_weight * (matrix_binary_mask * softmax_multi_binary[:, :, 1]).unsqueeze(-1) * (expanded_matrix_params.unsqueeze(0))).sum()
        if self.max_prune_step <= 10:
            matrix_param_and_weight = final_matrix_weight * ((matrix_binary_mask).unsqueeze(-1)) * (expanded_matrix_params.unsqueeze(0))
        else:
            matrix_param_and_weight = final_matrix_weight * ((matrix_binary_mask * matrix_bianry_arch_weight.cpu()).unsqueeze(-1)) * (expanded_matrix_params.unsqueeze(0))
        matrix_param_exp = matrix_param_and_weight.sum()
        self._update_params_scale_expectation(matrix_param_and_weight)


        #vertor based params, mask in shape [layers, modules], except for final_norm_mask
        encoder_binary_mask, decoder_binary_mask, final_norm_mask = self.encoder_vector_binary_mask, self.decoder_vector_binary_mask, self.final_norm_binary_mask
        if self.max_prune_step <= 10:
            pass
        else:
            encoder_binary_mask = encoder_binary_mask * (F.softmax(self.arch_weights_binary_encoder, dim=-1)[:, :, 1].cpu())
            decoder_binary_mask = decoder_binary_mask * (F.softmax(self.arch_weights_binary_decoder, dim=-1).cpu()[:, :, 1])
            final_norm_mask = final_norm_mask * (F.softmax(self.arch_weights_binary_final_norm, dim=-1)[:, 1].cpu())
        all_expected_params = 0
        if self.use_prefix:
            prefix_mask = self.prefix_binary_mask
            if self.max_prune_step <= 10:
                prefix_mask = self.prefix_binary_mask * (F.softmax(self.arch_weights_binary_prefix, dim=-1).cpu()[..., 1])
            binary_prefix_params = (torch.tensor(self.prefix_params_mapped).unsqueeze(0) * prefix_mask).sum()
            all_expected_params += binary_prefix_params.item()
        # softmax_encoder_binary, softmax_decoder_binary = F.softmax(self.arch_weights_binary_encoder, dim=-1).detach().cpu(), F.softmax(self.arch_weights_binary_decoder, dim=-1).detach().cpu()
        # softmax_final_norm = F.softmax(self.arch_weights_binary_final_norm, dim=-1).detach().cpu()
        # binary_encoder_params = (torch.tensor(self.vector_based_params_mapped_encoder).unsqueeze(0) * encoder_binary_mask * softmax_encoder_binary[:,:,1]).sum()
        binary_encoder_params = (torch.tensor(self.vector_based_params_mapped_encoder).unsqueeze(0) * encoder_binary_mask).sum()

        # binary_decoder_params = (torch.tensor(self.vector_based_params_mapped_decoder).unsqueeze(0) * decoder_binary_mask * softmax_decoder_binary[:,:,1]).sum()
        binary_decoder_params = (torch.tensor(self.vector_based_params_mapped_decoder).unsqueeze(0) * decoder_binary_mask).sum()
        # binary_final_norm_params = (torch.tensor(self.vector_based_params_mapped_final_norm) * final_norm_mask * softmax_final_norm[:,1]).sum()
        binary_final_norm_params = (torch.tensor(self.vector_based_params_mapped_final_norm) * final_norm_mask).sum()
        all_expected_params += (matrix_param_exp + binary_encoder_params + binary_decoder_params + binary_final_norm_params).item()
        return all_expected_params

    def _update_params_scale_expectation(self, matrix_param_and_weight):
        matrix_param_and_weight = matrix_param_and_weight.sum(dim=-1)
        for (n, s) in self.param_scale_dict.items():
            if 'lora' in n or 'adapter' in n:
                layer_id, encoder_flag, final_layer_norm_flag, matrix_flag, loc = recognize_layer_id(n)
                if not encoder_flag:
                    layer_id += 24
                new_param_scale = matrix_param_and_weight[layer_id, loc]
                module_id = self.module_id_dict[n]
                self.param_scale_list[module_id] = new_param_scale.item()
        # return

    def prune_modules(self):
        all_expected_params = self.get_param_expectation()
        params_pruned = (all_expected_params - self.budget_abs) / self.max_prune_step
        self.max_prune_step -= 1
        print(f"budget: {self.budget_abs}, current expectation: {all_expected_params}, pruned: {params_pruned}")

        gradients = [i.cpu().item() for i in self.gradient_records_list]
        gradients_mask = [999 if i == True else 0 for i in self.prune_records_list]
        masked_gradients = [a + b for a, b in zip(gradients, gradients_mask)]

        # add other modules later
        param_numbers = self.param_scale_list
        # Pair module index with gradient and number of parameters, then sort by gradient descending
        modules = sorted(enumerate(zip(masked_gradients, param_numbers)), key=lambda x: x[1][0], reverse=False)
        # print("rank", modules)
        current_pruning_num = 0
        pruned_idx, pruned_names = [], []
        for idx, (gradient, param_number) in modules:
            if current_pruning_num + param_number <= params_pruned:
                self.prune_records_list[idx] = True
                module_name = self.id_module_dict[idx]
                self.prune_dict[module_name] = True
                pruned_idx.append(idx)
                pruned_names.append(module_name)
                current_pruning_num += param_number
            elif params_pruned > 0 and current_pruning_num + param_number > params_pruned and len(pruned_idx) == 0:
                self.prune_records_list[idx] = True
                module_name = self.id_module_dict[idx]
                self.prune_dict[module_name] = True
                pruned_idx.append(idx)
                pruned_names.append(module_name)
                current_pruning_num += param_number
            # Stop adding if the next module exceeds the budget
            if current_pruning_num > params_pruned:
                break
        self.update_prune_mask()
        return pruned_idx, pruned_names

    def finalize_arch(self):
        backbone = self.t5_model
        # arch_weights_binary_encoder_matrix, arch_weights_binary_decoder_matrix = None, None
        # if not self.early_stop:
        if self.iter_search:
            arch_weights_binary_encoder_matrix = self.arch_weights_binary_encoder_matrix
            arch_weights_binary_decoder_matrix = self.arch_weights_binary_decoder_matrix
        else:
            weight_all = torch.tensor([0, 1])
            ideal_shape = self.arch_weights_multi_encoder[..., :2]
            while weight_all.dim() < ideal_shape.dim():
                weight_all = weight_all.unsqueeze(0)
                weight_all = weight_all.expand_as(ideal_shape).cuda()
            arch_weights_binary_encoder_matrix = weight_all.clone()
            arch_weights_binary_decoder_matrix = weight_all.clone()
        arch_weights_binary_prefix = self.arch_weights_binary_prefix
        arch_weights_binary_encoder = self.arch_weights_binary_encoder
        arch_weights_binary_decoder = self.arch_weights_binary_decoder
        arch_weights_multi_encoder = self.arch_weights_multi_encoder
        arch_weights_multi_decoder = self.arch_weights_multi_decoder
        arch_weights_multi_prefix = self.arch_weights_multi_prefix
        arch_weights_binary_final_norm = self.arch_weights_binary_final_norm
        if self.early_stop and not self.retrain:
            arch_weights_binary_encoder_matrix = nn.Parameter(
                F.one_hot(self.encoder_matrix_binary_mask, num_classes=2), requires_grad=False)
            arch_weights_binary_decoder_matrix = nn.Parameter(
                F.one_hot(self.decoder_matrix_binary_mask, num_classes=2), requires_grad=False)
            if self.use_prefix:
                arch_weights_binary_prefix = nn.Parameter(
                    F.one_hot(self.prefix_binary_mask, num_classes=2), requires_grad=False)

            arch_weights_binary_encoder = nn.Parameter(F.one_hot(self.encoder_vector_binary_mask, num_classes=2),
                                                            requires_grad=False)
            arch_weights_binary_decoder = nn.Parameter(F.one_hot(self.decoder_vector_binary_mask, num_classes=2),
                                                            requires_grad=False)
            arch_weights_binary_final_norm = nn.Parameter(F.one_hot(self.final_norm_binary_mask, num_classes=2),
                                                           requires_grad=False)

        for param in backbone.parameters():
            param.requires_grad = False

        #make the weights to a one-hot matrix, or here we only need the vector
        if self.use_prefix:
            prefix_binary_mask = self.get_max_weight(arch_weights_binary_prefix)
            if self.early_stop:
                prefix_binary_mask = arch_weights_binary_prefix
            _, prefix_max_dim = torch.max(arch_weights_multi_prefix, dim=-1)
            prefix_dimension_mask = prefix_max_dim.cuda()
            backbone.prefix_module.freeze_arch(finalized_weight={"binary":prefix_binary_mask.cuda(), "dim":prefix_dimension_mask}
                                               , retrain_flag=self.retrain)

        for t_layer_i, blk in enumerate(backbone.encoder.block):
            attn = blk.layer[0].SelfAttention
            layer_norm = blk.layer[0].layer_norm
            ffn = blk.layer[-1].DenseReluDense
            #binary order: attn, attn_norm, ffn, ffn_norm, cross_attn_norm
            # binary_encoder_matrix_layer = None
            # if not self.early_stop:
            binary_encoder_matrix_layer = arch_weights_binary_encoder_matrix[t_layer_i]
            binary_encoder_layer = arch_weights_binary_encoder[t_layer_i]
            dimension_encoder_layer = arch_weights_multi_encoder[t_layer_i]
            q_weight = {
                'lora': {"binary": binary_encoder_matrix_layer[0] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[0]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_encoder_layer[0], "dim": None}
            }
            k_weight = {
                'lora': {"binary": binary_encoder_matrix_layer[1] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[1]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_encoder_layer[1], "dim": None}
            }
            v_weight = {
                'lora': {"binary": binary_encoder_matrix_layer[2] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[2]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_encoder_layer[2], "dim": None}
            }
            o_weight = {
                'lora': {"binary": binary_encoder_matrix_layer[3] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[3]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_encoder_layer[3], "dim": None}
            }

            attn_norm_weight = {
                'lora': None, 'adapter':None,
                'bitfit': {"binary": binary_encoder_layer[4], "dim": None},
                'lnfit': {"binary": binary_encoder_layer[5], "dim": None}
            }

            ffn1_weight = {
                'lora': {"binary": binary_encoder_matrix_layer[4] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[4]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_encoder_layer[6], "dim": None}
            }
            ffn2_weight = {
                'lora': {"binary": binary_encoder_matrix_layer[5] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[5]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_encoder_layer[7], "dim": None}
            }

            ffn_norm_weight = {
                'lora': None, 'adapter': None,
                'bitfit': {"binary": binary_encoder_layer[8], "dim": None},
                'lnfit': {"binary": binary_encoder_layer[9], "dim": None}
            }

            attn_adapter_weight = {
                'lora': None, 'lnfit': None,
                'bitfit': None,
                'adapter': {"binary": binary_encoder_matrix_layer[6] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[6]}
            }
            ffn_adapter_weight = {
                'lora': None, 'lnfit': None,
                'bitfit': None,
                'adapter': {"binary": binary_encoder_matrix_layer[7] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[7]},
            }
            if self.use_PA and self.use_SA:
                ffn_adapter_weight['sa'] = {"binary": binary_encoder_matrix_layer[8] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[8]}
                ffn_adapter_weight['pa'] = {"binary": binary_encoder_matrix_layer[9] if binary_encoder_matrix_layer is not None else None, "dim": dimension_encoder_layer[9]}


            attn.original_module.q.freeze_arch(finalized_weight=q_weight, retrain_flag=self.retrain)
            attn.original_module.k.freeze_arch(finalized_weight=k_weight, retrain_flag=self.retrain)
            attn.original_module.v.freeze_arch(finalized_weight=v_weight, retrain_flag=self.retrain)
            attn.original_module.o.freeze_arch(finalized_weight=o_weight, retrain_flag=self.retrain)

            ffn.original_module.wi.freeze_arch(finalized_weight=ffn1_weight, retrain_flag=self.retrain)
            ffn.original_module.wo.freeze_arch(finalized_weight=ffn2_weight, retrain_flag=self.retrain)

            blk.layer[0].SelfAttention.freeze_arch(finalized_weight=attn_adapter_weight, retrain_flag=self.retrain)
            blk.layer[-1].DenseReluDense.freeze_arch(finalized_weight=ffn_adapter_weight, retrain_flag=self.retrain)
            blk.layer[-1].layer_norm.freeze_arch(finalized_weight=ffn_norm_weight, retrain_flag=self.retrain)
            blk.layer[0].layer_norm.freeze_arch(finalized_weight=attn_norm_weight, retrain_flag=self.retrain)

        for t_layer_i, blk in enumerate(backbone.decoder.block):
            attn = blk.layer[0].SelfAttention
            # cross_attn = blk.layer[1].EncDecAttention
            layer_norm = blk.layer[0].layer_norm
            ffn = blk.layer[-1].DenseReluDense
            # binary_decoder_matrix_layer = None
            # if not self.early_stop:
            binary_decoder_matrix_layer = arch_weights_binary_decoder_matrix[t_layer_i]
            binary_decoder_layer = arch_weights_binary_decoder[t_layer_i]
            dimension_decoder_layer = arch_weights_multi_decoder[t_layer_i]
            q_weight = {
                'lora': {"binary": binary_decoder_matrix_layer[0] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[0]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_decoder_layer[0], "dim": None}
            }
            k_weight = {
                'lora': {"binary": binary_decoder_matrix_layer[1] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[1]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_decoder_layer[1], "dim": None}
            }
            v_weight = {
                'lora': {"binary": binary_decoder_matrix_layer[2] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[2]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_decoder_layer[2], "dim": None}
            }
            o_weight = {
                'lora': {"binary": binary_decoder_matrix_layer[3] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[3]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_decoder_layer[3], "dim": None}
            }

            attn_norm_weight = {
                'lora': None, 'adapter': None,
                'bitfit': {"binary": binary_decoder_layer[4], "dim": None},
                'lnfit': {"binary": binary_decoder_layer[5], "dim": None}
            }

            ffn1_weight = {
                'lora': {"binary": binary_decoder_matrix_layer[4] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[4]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_decoder_layer[6], "dim": None}
            }
            ffn2_weight = {
                'lora': {"binary": binary_decoder_matrix_layer[5] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[5]},
                'adapter': None, 'lnfit': None,
                'bitfit': {"binary": binary_decoder_layer[7], "dim": None}
            }

            ffn_norm_weight = {
                'lora': None, 'adapter': None,
                'bitfit': {"binary": binary_decoder_layer[8], "dim": None},
                'lnfit': {"binary": binary_decoder_layer[9], "dim": None}
            }
            cross_attn_norm_weight = {
                'lora': None, 'adapter': None,
                'bitfit': {"binary": binary_decoder_layer[10], "dim": None},
                'lnfit': {"binary": binary_decoder_layer[11], "dim": None}
            }

            attn_adapter_weight = {
                'lora': None, 'lnfit': None,
                'bitfit': None,
                'adapter': {"binary": binary_decoder_matrix_layer[6] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[6]}
            }
            ffn_adapter_weight = {
                'lora': None, 'lnfit': None,
                'bitfit': None,
                'adapter': {"binary": binary_decoder_matrix_layer[7] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[7]},
                
            }
            if self.use_PA and self.use_SA:
                ffn_adapter_weight['sa'] = {"binary": binary_decoder_matrix_layer[8] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[8]}
                ffn_adapter_weight['pa'] = {"binary": binary_decoder_matrix_layer[9] if binary_decoder_matrix_layer is not None else None, "dim": dimension_decoder_layer[9]}


            attn.original_module.q.freeze_arch(finalized_weight=q_weight, retrain_flag=self.retrain)
            attn.original_module.k.freeze_arch(finalized_weight=k_weight, retrain_flag=self.retrain)
            attn.original_module.v.freeze_arch(finalized_weight=v_weight, retrain_flag=self.retrain)
            attn.original_module.o.freeze_arch(finalized_weight=o_weight, retrain_flag=self.retrain)

            ffn.original_module.wi.freeze_arch(finalized_weight=ffn1_weight, retrain_flag=self.retrain)
            ffn.original_module.wo.freeze_arch(finalized_weight=ffn2_weight, retrain_flag=self.retrain)

            blk.layer[0].SelfAttention.freeze_arch(finalized_weight=attn_adapter_weight, retrain_flag=self.retrain)
            blk.layer[-1].DenseReluDense.freeze_arch(finalized_weight=ffn_adapter_weight, retrain_flag=self.retrain)
            blk.layer[-1].layer_norm.freeze_arch(finalized_weight=ffn_norm_weight, retrain_flag=self.retrain)
            blk.layer[0].layer_norm.freeze_arch(finalized_weight=attn_norm_weight, retrain_flag=self.retrain)
            # different with encoder, we need to tune layer norm in decoder
            blk.layer[1].layer_norm.freeze_arch(finalized_weight=cross_attn_norm_weight, retrain_flag=self.retrain)
        # finally, tune the final_layer_norm
        encoder_norm_weight = {
            'lora': None, 'adapter': None,
            'bitfit': {"binary": arch_weights_binary_final_norm[0], "dim": None},
            'lnfit': {"binary": arch_weights_binary_final_norm[1], "dim": None}
        }
        decoder_norm_weight = {
            'lora': None, 'adapter': None,
            'bitfit': {"binary": arch_weights_binary_final_norm[2], "dim": None},
            'lnfit': {"binary": arch_weights_binary_final_norm[3], "dim": None}
        }
        backbone.encoder.final_layer_norm.freeze_arch(finalized_weight=encoder_norm_weight, retrain_flag=self.retrain)
        backbone.decoder.final_layer_norm.freeze_arch(finalized_weight=decoder_norm_weight, retrain_flag=self.retrain)

        for name, param in self.t5_model.named_parameters():
            if "LoRA" in name or "LNfit" in name or "Adapter" in name or "BitFit" in name or "sadapter" in name or "padapter" in name or "prefix" in name:
                param.requires_grad = True
                if self.retrain:
                    print(name)
            else:
                param.requires_grad = False

    def modify_arch_mask(self, binary_stage=True):
        if binary_stage: #binary search stage
            dimension_weights_encoder = self.arch_weights_multi_encoder
            dimension_weights_decoder = self.arch_weights_multi_decoder
            dimension_weights_prefix = self.arch_weights_multi_prefix

            # max_indices_temp_encoder = dimension_weights_encoder.argmax(dim=-1)
            # max_indices_temp_decoder = dimension_weights_decoder.argmax(dim=-1)
            # dimension_weights = dirichlet.Dirichlet(F.elu(dimension_weights.clone()) + 1).sample()
            max_indices_encoder = dimension_weights_encoder.argmax(dim=-1)  # shape: [layers, possible_positions]
            max_indices_decoder = dimension_weights_decoder.argmax(dim=-1)  # shape: [layers, possible_positions]
            if self.use_prefix:
                max_indices_prefix = dimension_weights_prefix.argmax(dim=-1)  # shape: [layers, possible_positions]
                self.dimension_mask_prefix = max_indices_prefix
            # if self.progressive_fix: # here if we need to use early-stop
            #     # replace the max_indices with search_mask if it is freezed
            #     max_indices[self.dimension_search_mask] = max_indices_temp[self.dimension_search_mask]
            self.dimension_mask_encoder = max_indices_encoder
            self.dimension_mask_decoder = max_indices_decoder

            self.binary_mask_encoder, self.binary_mask_decoder, self.binary_mask_prefix = None, None, None
        else:
            binary_weights_encoder = self.arch_weights_binary_encoder_matrix
            binary_weights_decoder = self.arch_weights_binary_decoder_matrix
            if self.use_prefix:
                binary_weights_prefix = self.arch_weights_binary_prefix
                max_indices_prefix = binary_weights_prefix
                if not self.fix_prefix_dim:
                    max_indices_prefix = F.gumbel_softmax(binary_weights_prefix, hard=True)
                max_indices_prefix = max_indices_prefix.argmax(dim=-1)
                self.binary_mask_prefix = max_indices_prefix
            # max_indices_temp = binary_weights.argmax(dim=-1)
            # binary_weights_encoder = dirichlet.Dirichlet(F.elu(binary_weights_encoder.clone()) + 1).sample()
            # max_indices_encoder = binary_weights_encoder.argmax(dim=-1) #shape: [layers, possible_positions]
            # binary_weights_decoder = dirichlet.Dirichlet(F.elu(binary_weights_decoder.clone()) + 1).sample()
            # max_indices_decoder = binary_weights_decoder.argmax(dim=-1)
            max_indices_encoder = binary_weights_encoder
            if not self.binary_then_dim:
                # in this setting, the binary weights is already searched
                max_indices_encoder = F.gumbel_softmax(binary_weights_encoder, hard=True)
            max_indices_encoder = max_indices_encoder.argmax(dim=-1)
            max_indices_decoder = binary_weights_decoder
            if not self.binary_then_dim:
                max_indices_decoder = F.gumbel_softmax(binary_weights_decoder, hard=True)
            max_indices_decoder = max_indices_decoder.argmax(dim=-1)

            # if self.progressive_fix: # here if we need to use early-stop
            #     # replace the max_indices with search_mask if it is freezed
            #     max_indices[self.binary_search_mask] = max_indices_temp[self.binary_search_mask]
            self.binary_mask_encoder = max_indices_encoder
            self.binary_mask_decoder = max_indices_decoder
            self.dimension_mask_encoder, self.dimension_mask_decoder, self.dimension_mask_prefix = None, None, None

    def get_max_weight(self, weights):
        _, max_indices = torch.max(weights, dim=-1, keepdim=True)
        max_weights = torch.zeros_like(weights)
        max_weights.scatter_(dim=-1, index=max_indices, value=1)
        return max_weights

    def replace_binary_weights(self):
        self.arch_weights_binary_encoder_matrix = nn.Parameter(F.one_hot(self.encoder_matrix_binary_mask, num_classes=2), requires_grad=False)
        self.arch_weights_binary_decoder_matrix = nn.Parameter(F.one_hot(self.decoder_matrix_binary_mask, num_classes=2), requires_grad=False)
        self.arch_weights_binary_encoder = nn.Parameter(F.one_hot(self.encoder_vector_binary_mask, num_classes=2), requires_grad=False)
        self.arch_weights_binary_decoder = nn.Parameter(F.one_hot(self.decoder_vector_binary_mask, num_classes=2), requires_grad=False)
        self.arch_weights_binary_final_norm = nn.Parameter(F.one_hot(self.final_norm_binary_mask, num_classes=2), requires_grad=False)
        if self.use_prefix:
            self.arch_weights_binary_prefix = nn.Parameter(F.one_hot(self.prefix_binary_mask, num_classes=2), requires_grad=False)
        print("replace the binary arch weights with the pruning result")

    def init_gumbel_weights(self, epochs=100, eval_mode=False):
        if self.iter_search:
            arch_weights_binary_encoder_matrix = self.arch_weights_binary_encoder_matrix
            arch_weights_binary_decoder_matrix = self.arch_weights_binary_decoder_matrix
        arch_weights_binary_encoder = self.arch_weights_binary_encoder
        arch_weights_binary_decoder = self.arch_weights_binary_decoder
        arch_weights_multi_encoder = self.arch_weights_multi_encoder
        arch_weights_multi_decoder = self.arch_weights_multi_decoder
        if self.use_prefix:
            arch_weights_binary_prefix = self.arch_weights_binary_prefix
            arch_weights_multi_prefix = self.arch_weights_multi_prefix
        arch_weights_binary_final_norm = self.arch_weights_binary_final_norm

        dimension_mask_encoder, binary_mask_encoder, dimension_mask_decoder, binary_mask_decoder = None, None, None, None
        dimension_mask_prefix, binary_mask_prefix = None, None
        if self.iter_search:
            if self.iterative_order or self.main_forward:  # iterative_order=True, means binary search stage
                self.modify_arch_mask(binary_stage=True)
                dimension_mask_encoder, dimension_mask_decoder, binary_mask_encoder, binary_mask_decoder = self.dimension_mask_encoder, self.dimension_mask_decoder, None, None
                if self.use_prefix:
                    dimension_mask_prefix, binary_mask_prefix = self.dimension_mask_prefix, None
                # if self.progressive_fix:
                #     binary_search_mask, dimension_search_mask = self.binary_search_mask, None
            else:
                dimension_mask_encoder, dimension_mask_decoder, binary_mask_encoder, binary_mask_decoder = None, None, None, None
                # if not self.early_stop:
                self.modify_arch_mask(binary_stage=False)
                dimension_mask_encoder, dimension_mask_decoder, binary_mask_encoder, binary_mask_decoder = None, None, self.binary_mask_encoder, self.binary_mask_decoder
                if self.use_prefix:
                    dimension_mask_prefix, binary_mask_prefix = None, binary_mask_prefix
                # if self.progressive_fix:
                #     binary_search_mask, dimension_search_mask = None, self.dimension_search_mask

        if self.use_search:
            temp = 5 - (5. - 1.) / self.all_epochs * epochs

            gumbel_weights_encoder_matrix, gumbel_weights_decoder_matrix, gumbel_weights_final_norm_all = None, None, None
            gumbel_weights_encoder_binary, gumbel_weights_decoder_binary = None, None

            if self.args.use_beta:
                arch_weights_binary_final_norm = dirichlet.Dirichlet(F.elu(arch_weights_binary_final_norm.clone()) + 1).rsample()
                arch_weights_binary_encoder = dirichlet.Dirichlet(F.elu(arch_weights_binary_encoder.clone()) + 1).rsample()
                arch_weights_binary_decoder = dirichlet.Dirichlet(F.elu(arch_weights_binary_decoder.clone()) + 1).rsample()
                if not self.iter_search:
                    arch_weights_multi_encoder = dirichlet.Dirichlet(F.elu(arch_weights_multi_encoder.clone()) + 1).rsample()
                    arch_weights_multi_decoder = dirichlet.Dirichlet(F.elu(arch_weights_multi_decoder.clone()) + 1).rsample()
                else:
                    if self.iterative_order or self.main_forward: #binary search stage
                        arch_weights_binary_encoder_matrix = dirichlet.Dirichlet(F.elu(arch_weights_binary_encoder_matrix.clone()) + 1).rsample()
                        arch_weights_binary_decoder_matrix = dirichlet.Dirichlet(F.elu(arch_weights_binary_decoder_matrix.clone()) + 1).rsample()
                    else:
                        arch_weights_multi_encoder = dirichlet.Dirichlet(F.elu(arch_weights_multi_encoder.clone()) + 1).rsample()
                        arch_weights_multi_decoder = dirichlet.Dirichlet(F.elu(arch_weights_multi_decoder.clone()) + 1).rsample()
            #final norm, no layers
            if not eval_mode and not self.retrain:
                gumbel_weights_final_norm = bernoulli_sample(arch_weights_binary_final_norm, temp=temp, binary_prune_mask=self.final_norm_binary_mask, no_gumbel=self.no_gumbel)
            else:
                max_indices = torch.max(arch_weights_binary_final_norm, dim=-1).indices
                max_weights = torch.zeros_like(arch_weights_binary_final_norm)
                max_weights[np.arange(len(max_indices)), max_indices] = 1
                gumbel_weights_final_norm = max_weights
            gumbel_weights_prefix = None
            if not eval_mode and not self.retrain:
                self.freeze_dimension_mask = False
                if not self.iter_search :
                    gumbel_weights_encoder_matrix = bernoulli_sample(arch_weights_multi_encoder, temp=temp,
                                                                     binary_prune_mask=None, early_stop=False, binary_mask=None, no_gumbel=self.no_gumbel)
                    gumbel_weights_decoder_matrix = bernoulli_sample(arch_weights_multi_decoder, temp=temp,
                                                                     binary_prune_mask=None, early_stop=False, binary_mask=None, no_gumbel=self.no_gumbel)
                elif self.iterative_order or self.main_forward:  # binary search stage
                    gumbel_weights_encoder_matrix = bernoulli_sample(arch_weights_binary_encoder_matrix, temp=temp, binary_prune_mask=self.encoder_matrix_binary_mask, early_stop=self.early_stop,
                                                                     binary_mask=binary_mask_encoder, no_gumbel=self.no_gumbel)
                    gumbel_weights_decoder_matrix = bernoulli_sample(arch_weights_binary_decoder_matrix, temp=temp, binary_prune_mask=self.decoder_matrix_binary_mask, early_stop=self.early_stop,
                                                                     binary_mask=binary_mask_decoder, no_gumbel=self.no_gumbel)

                    if self.use_prefix:
                        gumbel_weights_prefix = bernoulli_sample(arch_weights_binary_prefix, temp=temp, binary_prune_mask=self.prefix_binary_mask, early_stop=self.early_stop,
                                                                     binary_mask=binary_mask_prefix, no_gumbel=self.no_gumbel)
                else:
                    gumbel_weights_encoder_matrix = bernoulli_sample(arch_weights_multi_encoder, temp=temp, early_stop=self.early_stop, dim_stage=True,
                                                                     binary_mask=binary_mask_encoder, binary_prune_mask=self.encoder_matrix_binary_mask, no_gumbel=self.no_gumbel)
                    gumbel_weights_decoder_matrix = bernoulli_sample(arch_weights_multi_decoder, temp=temp, early_stop=self.early_stop, dim_stage=True,
                                                                     binary_mask=binary_mask_decoder, binary_prune_mask=self.decoder_matrix_binary_mask, no_gumbel=self.no_gumbel)
                    if self.use_prefix and not self.fix_prefix_dim:
                        gumbel_weights_prefix = bernoulli_sample(arch_weights_multi_prefix, temp=temp, binary_prune_mask=self.prefix_binary_mask,
                                                                     early_stop=self.early_stop, dim_stage=True,
                                                                     binary_mask=binary_mask_prefix, no_gumbel=self.no_gumbel)
                    elif self.fix_prefix_dim:
                        gumbel_weights_prefix = arch_weights_multi_prefix
                                                                    # in shape [layers, opsitions, dimensions]
                    if self.early_stop:
                        max_weights_encoder_matrix = self.get_max_weight(arch_weights_multi_encoder)
                        max_weights_decoder_matrix = self.get_max_weight(arch_weights_multi_decoder) # max_weights in shape [layers, opsitions]
                        if self.dimension_fix_mask is not None:
                            inverted_dimension_mask = (1 - self.dimension_fix_mask.cuda())
                            # [layers, modules, 1] * [layers, modules, 3(candidates)]
                            gumbel_weights_encoder_matrix = self.dimension_fix_mask[:self.num_encoder_layers].unsqueeze(-1).cuda() * max_weights_encoder_matrix + inverted_dimension_mask[:self.num_encoder_layers].unsqueeze(-1) * gumbel_weights_encoder_matrix
                            gumbel_weights_decoder_matrix = self.dimension_fix_mask[self.num_encoder_layers:].unsqueeze(-1).cuda() * max_weights_decoder_matrix + inverted_dimension_mask[self.num_encoder_layers:].unsqueeze(-1) * gumbel_weights_decoder_matrix
                        encoder_matrix_binary_mask, decoder_matrix_binary_mask = self.encoder_matrix_binary_mask, self.decoder_matrix_binary_mask
                        gumbel_weights_encoder_matrix = encoder_matrix_binary_mask.unsqueeze(-1).cuda() * gumbel_weights_encoder_matrix
                        gumbel_weights_decoder_matrix = decoder_matrix_binary_mask.unsqueeze(-1).cuda() * gumbel_weights_decoder_matrix
                        if self.use_prefix:
                            gumbel_weights_prefix = self.prefix_binary_mask.unsqueeze(-1).cuda() * gumbel_weights_prefix.cuda()
                gumbel_weights_encoder_binary = bernoulli_sample(arch_weights_binary_encoder, temp=temp, early_stop=self.early_stop, binary_prune_mask=self.encoder_vector_binary_mask, no_gumbel=self.no_gumbel)
                gumbel_weights_decoder_binary = bernoulli_sample(arch_weights_binary_decoder, temp=temp, early_stop=self.early_stop, binary_prune_mask=self.decoder_vector_binary_mask, no_gumbel=self.no_gumbel)
            else:
                # if not self.iter_search or self.iterative_order or self.main_forward:
                #in eval stage, we directly use the binary weight
                gumbel_weights_encoder_matrix = self.get_max_weight(arch_weights_binary_encoder_matrix)
                gumbel_weights_decoder_matrix = self.get_max_weight(arch_weights_binary_decoder_matrix)
                if self.use_prefix:
                    gumbel_weights_prefix = self.get_max_weight(arch_weights_binary_prefix)
                # else:
                #     gumbel_weights_encoder_matrix = self.get_max_weight(arch_weights_multi_encoder)
                #     gumbel_weights_decoder_matrix = self.get_max_weight(arch_weights_multi_decoder)
                gumbel_weights_encoder_binary = self.get_max_weight(arch_weights_binary_encoder)
                gumbel_weights_decoder_binary = self.get_max_weight(arch_weights_binary_decoder)
                if self.freeze_dimension_mask:
                    self.modify_arch_mask(binary_stage=True)
                    self.freeze_dimension_mask = True


            return {"encoder": gumbel_weights_encoder_matrix if gumbel_weights_encoder_matrix is not None else None,
                    "encoder_binary": gumbel_weights_encoder_binary if gumbel_weights_encoder_binary is not None else None,
                    "decoder": gumbel_weights_decoder_matrix if gumbel_weights_decoder_matrix is not None else None,
                    "decoder_binary": gumbel_weights_decoder_binary if gumbel_weights_decoder_binary is not None else None,
                    "final_norm": gumbel_weights_final_norm, 'prefix': gumbel_weights_prefix}

        else:
            print("no sampling lora")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def prune_step(self, cur_epoch):
        # if self.main_forward and cur_epoch >= 0:
        if self.max_prune_step > 0:
            self.update_grad()
        if self.args.prune_begin_epoch <= cur_epoch and self.main_forward:
            # if not eval_mode and cur_epoch >= 0:
            # if self.early_stop:
            self.prune_flag = False
            if self.max_prune_step > 0:
                # self.update_grad()
                self.update_dimension_pruning()
                self.prune_trigger()
            if self.prune_flag:
                print("Pruning step: ", self.max_prune_step)
                print("start pruning")
                pruned_idx, pruned_names = self.prune_modules()
                fix_indices = self.fix_dimensions()
                # print("Fixed modules at this round", fix_indices)
                print("Pruned modules at this round", pruned_names)
                print("Pruned gradients: ", [self.gradient_records_list[id_] for id_ in pruned_idx])
                self.prune_flag = False

        # stop pruning and finalize the arch
        if self.early_stop and self.main_forward and self.max_prune_step == 0:
            print("Finally selected modules", [self.id_module_dict[idw] for idw in range(self.modules_number) if
                                               self.prune_records_list[idw] == False])
            self.replace_binary_weights()
            print("gradients records", self.sen_records_dict)

    def forward(self, x, cur_epoch, eval_mode=False, main_forward=False) -> Tensor:
        self.main_forward = main_forward # main_forward: it means that this is not the forward for the "arch search"
        if eval_mode:
            self.main_forward = True
        # if self.use_prefix:
        #     prefix = self.t5_model.prefix_module.eject()  # # [layers, len(qv), prefix, dim]
        if self.use_search and not self.retrain:
            # print("self.iter_oder", self.iterative_order)
            gumbel_weights_all_dict = self.init_gumbel_weights(epochs=cur_epoch, eval_mode=eval_mode)
            dimension_mask = {
                "encoder_dimension_mask": self.dimension_mask_encoder,
                "decoder_dimension_mask": self.dimension_mask_decoder,
                "prefix_dimension_mask": self.dimension_mask_prefix
            }
            if not self.iter_search:
                self.iterative_order = None
            # if self.no_gumbel:
            #     print(gumbel_weights_all_dict)
            loss = self.t5_model(**x, gumbel_weights=gumbel_weights_all_dict, dimension_mask=dimension_mask,
                                 iterative_order=self.iterative_order, main_forward=self.main_forward)
            if not self.main_forward and self.iter_search:
                self.iterative_order = not self.iterative_order
            if self.dim_then_binary or self.binary_then_dim:
                all_epochs = self.all_epochs
                if cur_epoch <= all_epochs // 2:
                    self.iterative_order = True if self.binary_then_dim else False
                else:
                    self.iterative_order = False if self.binary_then_dim else True
                # if self.early_stop: #remove this case: final version
                #     self.iterative_order = False
        else:
            loss = self.t5_model(**x)
        return loss





if __name__ == "__main__":  # Debug

    print("MoM for S3Delta")
