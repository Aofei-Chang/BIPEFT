# Sheng Wang at Feb 22 2023
import copy
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from safetensors import safe_open
# from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter
import numpy as np

# from base_vit import ViT
from vit_backbone import VisionTransformer
from gumbel_module import GumbleSoftmax, gumbel_sample_weight, gumbel_multiple_weight, measure_entropy, calculate_zeta_for_shifting


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, wa: nn.Module, wb: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = wa
        self.w_b = wb

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        # x = self.w(x) + F.linear(input=F.linear(x, weight=self.wa), weight=self.wb)
        return x


class _LoRALayer_search(nn.Module):
    def __init__(self, w: nn.Module, w_a: torch.tensor, w_b: torch.tensor):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + F.linear(input=F.linear(x, weight=self.w_a), weight=self.w_b)
        return x


class LA_ViT(nn.Module):

    def __init__(self, vit_model: VisionTransformer, r: int, num_classes: int = 0, lora_layer=None,
                 classifier_name="head", use_search=False, retrain=False, progressive_fix=False, use_sparse=False,
                 args=None):
        super(LA_ViT, self).__init__()

        assert r > 0
        # base_vit_dim = vit_model.encoder.layer[0].attn.query.weight.shape[0]
        base_vit_dim = 768
        dim = base_vit_dim
        self.search_lora_dim = [0, 1, 4, 8]
        self.search_adapter_dim = [0, 1, 4, 8]
        # self.search_lora_dim = [0, 8]
        # self.fixed_layers = [0, 3, 6, 9]
        self.fixed_layers = None
        self.num_layers = len(vit_model.transformer.encoder.layer)
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(self.num_layers))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        self.GumbleSoftmax = GumbleSoftmax()
        self.use_search = use_search
        self.retrain = retrain
        self.num_options = 4
        self.use_sparse = use_sparse
        self.is_LoRA = args.is_LoRA
        self.is_adapter = args.is_adapter

        self.args = args
        self.progressive_fix = progressive_fix
        self.fix_lora = args.fix_lora
        self.search_all_lora = args.search_all_lora
        self.fix_adapter = args.fix_adapter
        print(f"fix lora: {self.fix_lora}")
        print(f"search all lora: {self.search_all_lora}")
        print(f"fix adapter: {self.fix_adapter}")
        print(f"use search: {use_search}")
        print(f"use sparse: {use_sparse}")
        print(f"use progressive_fix: {progressive_fix}")
        self.budget_abs = args.budget_abs
        self.budget_anneal = args.budget_anneal
        self.budget_high = 0
        if self.budget_anneal:
            # num_params = sum(p.numel() for p in search_model.parameters() if p.requires_grad)
            self.budget_high = 685348
            print(f"highest budget: {self.budget_high}")
        self.use_budget = args.use_budget
        self.last_zeta = 0
        if args is not None:
            self.last_zeta = args.last_zeta
        print(self.last_zeta, "last_zeta")
        self.print_lora_eval = 1
        self.print_adapter_eval = 1
        self.print_zeta = [True for _ in range(121)]
        if self.use_budget:
            print(f"use budget_abs: {self.budget_abs}")
        self.lora_num = 4
        if self.search_all_lora:
            self.lora_num = 6
        self.arch_fix_mask = torch.zeros(self.num_layers, self.lora_num)
        self.adapter_arch_fix_mask = torch.zeros(self.num_layers, 2)
        self.search_multiple = (not self.fix_adapter) and (not self.fix_lora) and (args.is_LoRA) and (args.is_adapter)
        if self.search_multiple:
            self.delta_arch_fix_mask = torch.zeros(self.num_layers, self.lora_num+2)
        if progressive_fix:
            self.prune_epoch_flag = dict(zip(list(range(121)), [True for _ in range(121)]))
        else:
            self.prune_epoch_flag = dict(zip(list(range(121)), [False for _ in range(121)]))

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # init_super_lora weights
        if self.is_LoRA:
            for t_layer_i, blk in enumerate(vit_model.transformer.encoder.layer):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer:
                    continue
                w_q_linear = blk.attn.query
                w_v_linear = blk.attn.value
                w_k_linear = blk.attn.key
                w_o_linear = blk.attn.out

                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                w_a_linear_o = nn.Linear(dim, r, bias=False)
                w_b_linear_o = nn.Linear(r, dim, bias=False)
                w_a_linear_k = nn.Linear(dim, r, bias=False)
                w_b_linear_k = nn.Linear(r, dim, bias=False)
                if self.search_all_lora:
                    w_a_linear_ffn1 = nn.Linear(dim, r, bias=False)
                    w_b_linear_ffn1 = nn.Linear(r, 3072, bias=False)
                    w_a_linear_ffn2 = nn.Linear(3072, r, bias=False)
                    w_b_linear_ffn2 = nn.Linear(r, dim, bias=False)
                    self.w_As.append(w_a_linear_ffn1)
                    self.w_Bs.append(w_b_linear_ffn1)
                    self.w_As.append(w_a_linear_ffn2)
                    self.w_Bs.append(w_b_linear_ffn2)
                    blk.ffn.w_a_ffn1 = w_a_linear_ffn1
                    blk.ffn.w_a_ffn2 = w_a_linear_ffn2
                    blk.ffn.w_b_ffn1 = w_b_linear_ffn1
                    blk.ffn.w_b_ffn2 = w_b_linear_ffn2

                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                self.w_As.append(w_a_linear_o)
                self.w_Bs.append(w_b_linear_o)

                blk.attn.query = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
                blk.attn.value = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)
                blk.attn.key = _LoRALayer(w_k_linear, w_a_linear_k, w_b_linear_k)
                blk.attn.out = _LoRALayer(w_o_linear, w_a_linear_o, w_b_linear_o)


            self.reset_parameters()
            # for name, param in vit_model.named_parameters():
            #     if classifier_name in name or "w_a" in name or "w_b" in name:
            #         # print(name,"dnnd")
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False
        self.lora_vit = vit_model
        # if num_classes > 0:
        #     self.lora_vit.head = nn.Linear(vit_model.head.in_features, num_classes)

    # given initial super lora weights, return sampled weights
    def sample_lora(self, w_a_q, w_b_q, w_a_k, w_b_k, w_a_v, w_b_v, w_a_o, w_b_o, w_a_ffn1, w_b_ffn1, w_a_ffn2, w_b_ffn2,
                    gumbel_weights=None,
                    gumbel_mask=None):
        stacked_samples = self.sample_weights(w_a_q, w_b_q, w_a_k, w_b_k, w_a_v, w_b_v, w_a_o, w_b_o, w_a_ffn1, w_b_ffn1, w_a_ffn2, w_b_ffn2)
        (sampled_w_a_q, sampled_w_b_q, sampled_w_a_k, sampled_w_b_k,
         sampled_w_a_v, sampled_w_b_v, sampled_w_a_o, sampled_w_b_o,
         sampled_w_a_ffn1, sampled_w_b_ffn1, sampled_w_a_ffn2, sampled_w_b_ffn2) = stacked_samples
        # print("sampled size: ", sampled_w_a_q.size(), sampled_w_b_q.size())
        gumbel_q, gumbel_k, gumbel_v, gumbel_o = (
        gumbel_weights[0].unsqueeze(-1).unsqueeze(-1), gumbel_weights[1].unsqueeze(-1).unsqueeze(-1),
        gumbel_weights[2].unsqueeze(-1).unsqueeze(-1), gumbel_weights[3].unsqueeze(-1).unsqueeze(-1))

        sum_gumbel_ffn1, sum_gumbel_ffn2, sum_gumbel_ffn1_b, sum_gumbel_ffn2_b = None, None, None, None
        if self.search_all_lora:
            gumbel_ffn1, gumbel_ffn2 = (
                gumbel_weights[4].unsqueeze(-1).unsqueeze(-1), gumbel_weights[5].unsqueeze(-1).unsqueeze(-1),)
            sum_gumbel_ffn1, sum_gumbel_ffn2 = (torch.sum(gumbel_ffn1 * sampled_w_a_ffn1, dim=0),
                                                                      torch.sum(gumbel_ffn2 * sampled_w_a_ffn2, dim=0))
            sum_gumbel_ffn1_b, sum_gumbel_ffn2_b = (torch.sum(gumbel_ffn1.detach() * sampled_w_b_ffn1, dim=0),
                                                torch.sum(gumbel_ffn2.detach() * sampled_w_b_ffn2, dim=0))

        sum_gumbel_q, sum_gumbel_k, sum_gumbel_v, sum_gumbel_o = (torch.sum(gumbel_q * sampled_w_a_q, dim=0),
                                                                  torch.sum(gumbel_k * sampled_w_a_k, dim=0),
                                                                  torch.sum(gumbel_v * sampled_w_a_v, dim=0),
                                                                  torch.sum(gumbel_o * sampled_w_a_o, dim=0))

        w_a_q_sampled, w_a_k_sampled, w_a_v_sampled, w_a_o_sampled, w_a_ffn1_sampled, w_a_ffn2_sampled, \
            w_b_q_sampled, w_b_k_sampled, w_b_v_sampled, w_b_o_sampled, w_b_ffn1_sampled, w_b_ffn2_sampled = (
        sum_gumbel_q, sum_gumbel_k, sum_gumbel_v, sum_gumbel_o, sum_gumbel_ffn1, sum_gumbel_ffn2,
        torch.sum(gumbel_q.detach() * sampled_w_b_q, dim=0), torch.sum(gumbel_k.detach() * sampled_w_b_k, dim=0),
        torch.sum(gumbel_v.detach() * sampled_w_b_v, dim=0),
        torch.sum(gumbel_o.detach() * sampled_w_b_o, dim=0), sum_gumbel_ffn1_b, sum_gumbel_ffn2_b)


        return w_a_q_sampled, w_a_k_sampled, w_a_v_sampled, w_a_o_sampled, w_a_ffn1_sampled, w_a_ffn2_sampled,\
            w_b_q_sampled, w_b_k_sampled, w_b_v_sampled, w_b_o_sampled, w_b_ffn1_sampled, w_b_ffn2_sampled

    def modify_arch_mask(self, epoch):
        if self.search_multiple:
            arch_weights = self.lora_vit.transformer.encoder.delta_arch_weights
        else:
            arch_weights = self.lora_vit.transformer.encoder.lora_arch_weights
        # if epoch % 10 == 0:
        arch_entropy = measure_entropy(arch_weights)  # in shape [12, num_options]
        if self.search_multiple:
            arch_entropy = arch_entropy + 999 * torch.ones_like(
                arch_entropy).cuda() * self.delta_arch_fix_mask.cuda()  # for those already been masked, add 999 to entropy matrix
        else:
            arch_entropy = arch_entropy + 999 * torch.ones_like(
                arch_entropy).cuda() * self.arch_fix_mask.cuda()  # for those already been masked, add 999 to entropy matrix

        x_flat = arch_entropy.flatten()
        if self.search_all_lora:
            self.num_options = 6
        if self.search_multiple:
            num_options = self.num_options+2
        else:
            num_options = self.num_options
        values, flat_indices = torch.topk(-x_flat, num_options)
        rows, cols = torch.div(flat_indices, arch_entropy.size(1),
                               rounding_mode='trunc'), flat_indices % arch_entropy.size(1)
        if self.search_multiple:
            self.delta_arch_fix_mask[rows, cols] = 1
            print("lora_and_adapter", self.delta_arch_fix_mask)
        else:
            self.arch_fix_mask[rows, cols] = 1
            print(self.arch_fix_mask)
        print(f"Epoch {epoch}: Masking tensors at indices {rows, cols}")


    def prune_weights(self, gumbel_weights, arch_layer_weights, layer_id):
        # to fix some layers gradually
        # gumbel_weights shape: [possible_location, candidate_dims]
        options_to_mask_original = self.arch_fix_mask[layer_id].cuda()  # 1d: [num_options]
        options_to_mask = options_to_mask_original.unsqueeze(0).T

        max_indices = torch.max(arch_layer_weights, dim=-1).indices  # shape: [num_options]
        max_weights = torch.zeros_like(arch_layer_weights)
        max_weights[np.arange(len(max_indices)), max_indices] = 1

        gumbel_mask = options_to_mask * torch.ones_like(gumbel_weights).cuda()
        # print(gumbel_mask, "gumbel_mask")
        # print(gumbel_weights * (1 - gumbel_mask))
        # change the masked gumbel_weights to the fixed index, according to the arch_layer_weights
        gumbel_weights = gumbel_weights * (1 - gumbel_mask) + max_weights * gumbel_mask
        # print(gumbel_weights.requires_grad, "df")
        # gumbel_weights = gumbel_weights.detach()
        # print(gumbel_weights.requires_grad, "df")

        return gumbel_weights, options_to_mask_original

    def init_lora_samples(self, arch_weights=None, epochs=0, eval_mode=False):
        multiple_sign = arch_weights.shape[1] > 6

        if not self.retrain and (epochs + 1) % 10 == 0 and self.prune_epoch_flag[epochs]:
            self.modify_arch_mask(epochs)
            self.prune_epoch_flag[epochs] = False
        if self.use_budget and not self.retrain:
            if multiple_sign:
                fix_mask = self.delta_arch_fix_mask
            else:
                fix_mask = self.arch_fix_mask
            anneal_epoch=None
            if self.budget_anneal:
                # anneal_epoch=epochs
                if epochs >= 40:
                    self.last_zeta = calculate_zeta_for_shifting(arch_weights, self.budget_abs, ranks=self.search_lora_dim, epoch=anneal_epoch,
                                                         last_zeta=self.last_zeta, fix_mask=fix_mask, budget_high=self.budget_high)
            else:
                self.last_zeta = calculate_zeta_for_shifting(arch_weights, self.budget_abs, ranks=self.search_lora_dim,
                                                             epoch=anneal_epoch,
                                                             last_zeta=self.last_zeta, fix_mask=fix_mask,
                                                             budget_high=self.budget_high)
            if epochs % 1 == 0 and self.print_zeta[epochs]:
                print(self.last_zeta, "zeta at epoch ", epochs)
                self.print_zeta[epochs] = False
        if self.use_search:
            min_temp = 1
            temp = 4 - (4. - min_temp) / 120. * epochs
            max_indices_comput = []
            adapter_layer_gumbel_weights = []
            w_a_qs, w_a_ks, w_a_vs, w_a_os, w_b_qs, w_b_ks, w_b_vs, w_b_os, w_a_ffn1s, w_a_ffn2s, w_b_ffn1s, w_b_ffn2s = [], [], [], [], [], [], [], [], [], [], [], []
            for t_layer_i, blk in enumerate(self.lora_vit.transformer.encoder.layer):
                # print(arch_weights[t_layer_i].size(),"arch")
                manual_weights = None
                gumbel_mask = None
                if self.fixed_layers is not None:
                    if t_layer_i in self.fixed_layers:
                        max_indices = torch.tensor([2, 2, 2, 2])
                        manual_weights = torch.zeros_like(arch_weights[t_layer_i])
                        manual_weights[np.arange(len(max_indices)), max_indices] = 1

                if not eval_mode and not self.retrain:
                    # prob = arch_weights[t_layer_i]
                    if manual_weights is not None:
                        gumbel_weights = manual_weights
                    else:
                        # print(arch_weights[t_layer_i], "layer", t_layer_i)
                        gumbel_mask = self.arch_fix_mask[t_layer_i].cuda()
                        if multiple_sign:
                            gumbel_mask = self.delta_arch_fix_mask[t_layer_i].cuda()
                            gumbel_weights = gumbel_multiple_weight(arch_weights[t_layer_i],
                                                                  gumbel_mask=gumbel_mask,
                                                                  sample_time=1, temp=temp,
                                                                  GumbleSoftmax=self.GumbleSoftmax, zeta=self.last_zeta,
                                                                  use_sparse=self.use_sparse, args=self.args,
                                                                  ranks=self.search_lora_dim)  # shape: [possible_location, candidate_dims]
                        else:
                            gumbel_weights = gumbel_sample_weight(arch_weights[t_layer_i],
                                                              gumbel_mask=gumbel_mask,
                                                              sample_time=1, temp=temp,
                                                              GumbleSoftmax=self.GumbleSoftmax, zeta=self.last_zeta,
                                                              use_sparse=self.use_sparse, args=self.args,
                                                              ranks=self.search_lora_dim)  # shape: [possible_location, candidate_dims]
                        # print(gumbel_weights, "axsx")
                        # gumbel_weights, gumbel_mask = self.prune_weights(gumbel_weights, arch_layer_weights=arch_weights[t_layer_i], layer_id=t_layer_i)
                # If we only want few lora layer instead of all
                else:
                    weight_layer = arch_weights[t_layer_i]
                    if self.use_budget:
                        # self.last_zeta = -0.01560147723751139
                        bia = self.last_zeta * (torch.tensor(self.search_lora_dim)) / 2
                        bia_expanded = bia.unsqueeze(0).repeat(weight_layer.shape[0], 1).cuda()
                        weight_layer = (weight_layer - bia_expanded)
                        # weight_layer = torch.clamp(x, min=-200)  # Clamping
                        # weight_layer = F.softmax(x, dim=-1)
                    max_indices = torch.max(weight_layer, dim=-1).indices
                    if self.fixed_layers is not None:
                        if t_layer_i in self.fixed_layers:
                            max_indices = torch.tensor([2, 2, 2, 2])
                    max_indices_comput.append(max_indices)
                    # max_indices_comput.append(manual_index)
                    max_weights = torch.zeros_like(arch_weights[t_layer_i])
                    max_weights[np.arange(len(max_indices)), max_indices] = 1
                    gumbel_weights = max_weights
                # print(gumbel_weights, "gumbel_weights")
                if t_layer_i not in self.lora_layer:
                    continue
                gumbel_weights = gumbel_weights.cuda()
                adapter_gumbel_weights = None

                if multiple_sign:
                    gumbel_weights_lora = gumbel_weights[:self.lora_num]
                    adapter_gumbel_weights = gumbel_weights[self.lora_num:]
                    adapter_layer_gumbel_weights.append(adapter_gumbel_weights)
                else:
                    gumbel_weights_lora = gumbel_weights
                w_a_q, w_a_k, w_a_v, w_a_o = blk.attn.query.w_a.weight, blk.attn.key.w_a.weight, blk.attn.value.w_a.weight, blk.attn.out.w_a.weight
                w_b_q, w_b_k, w_b_v, w_b_o = blk.attn.query.w_b.weight, blk.attn.key.w_b.weight, blk.attn.value.w_b.weight, blk.attn.out.w_b.weight
                w_a_ffn1, w_b_ffn1, w_a_ffn2, w_b_ffn2 = [None] * 4
                if self.search_all_lora:
                    # w_a_ffn1, w_a_ffn2 = self.w_As[-2:]
                    # w_b_ffn1, w_b_ffn2 = self.w_Bs[-2:]
                    w_a_ffn1, w_a_ffn2, w_b_ffn1, w_b_ffn2 = blk.ffn.w_a_ffn1, blk.ffn.w_a_ffn2, blk.ffn.w_b_ffn1, blk.ffn.w_b_ffn2
                    w_a_ffn1, w_a_ffn2, w_b_ffn1, w_b_ffn2 = w_a_ffn1.weight, w_a_ffn2.weight, w_b_ffn1.weight, w_b_ffn2.weight
                w_a_q_sampled, w_a_k_sampled, w_a_v_sampled, w_a_o_sampled, w_a_ffn1_sampled, w_a_ffn2_sampled,\
                    w_b_q_sampled, w_b_k_sampled, w_b_v_sampled, w_b_o_sampled,\
                     w_b_ffn1_sampled, w_b_ffn2_sampled = (
                    self.sample_lora(w_a_q, w_b_q, w_a_k, w_b_k, w_a_v, w_b_v, w_a_o, w_b_o, w_a_ffn1, w_b_ffn1, w_a_ffn2, w_b_ffn2,
                                     gumbel_weights=gumbel_weights_lora, gumbel_mask=gumbel_mask))
                w_a_qs.append(w_a_q_sampled)
                w_a_ks.append(w_a_k_sampled)
                w_a_vs.append(w_a_v_sampled)
                w_a_os.append(w_a_o_sampled)
                w_a_ffn1s.append(w_a_ffn1_sampled)
                w_a_ffn2s.append(w_a_ffn2_sampled)
                w_b_qs.append(w_b_q_sampled)
                w_b_ks.append(w_b_k_sampled)
                w_b_vs.append(w_b_v_sampled)
                w_b_os.append(w_b_o_sampled)
                w_b_ffn1s.append(w_b_ffn1_sampled)
                w_b_ffn2s.append(w_b_ffn2_sampled)

            if self.retrain and self.print_lora_eval:
                if self.search_multiple:
                    print("lora params after search: ", compute_search_size([i[:self.lora_num] for i in max_indices_comput]))
                    print("adapter params after search: ", compute_adapter_search_size([i[self.lora_num:] for i in max_indices_comput]))
                else:
                    print("lora params after search: ", compute_search_size(max_indices_comput))
                self.print_lora_eval = 0
            if self.search_all_lora:
                return w_a_qs, w_a_ks, w_a_vs, w_a_os, w_a_ffn1s, w_a_ffn2s, w_b_qs, w_b_ks, w_b_vs, w_b_os, w_b_ffn1s, w_b_ffn2s, adapter_layer_gumbel_weights
            return w_a_qs, w_a_ks, w_a_vs, w_a_os, w_b_qs, w_b_ks, w_b_vs, w_b_os, adapter_layer_gumbel_weights
        else:
            print("not sampling lora")

    # given initial super lora weights, return stacked all possible sampled weights
    def sample_weights(self, w_a_q, w_b_q, w_a_k, w_b_k, w_a_v, w_b_v, w_a_o, w_b_o, w_a_ffn1, w_b_ffn1, w_a_ffn2, w_b_ffn2):
        (sampled_w_a_q, sampled_w_b_q, sampled_w_a_k, sampled_w_b_k,
         sampled_w_a_v, sampled_w_b_v, sampled_w_a_o, sampled_w_b_o,
         sampled_w_a_ffn1, sampled_w_b_ffn1, sampled_w_a_ffn2, sampled_w_b_ffn2) = (
            self.sample_weights_single(w_a_q, "a"), self.sample_weights_single(w_b_q, "b"),
            self.sample_weights_single(w_a_k, "a"), self.sample_weights_single(w_b_k, "b"),
            self.sample_weights_single(w_a_v, "a"), self.sample_weights_single(w_b_v, "b"),
            self.sample_weights_single(w_a_o, "a"), self.sample_weights_single(w_b_o, "b"),
            self.sample_weights_single(w_a_ffn1, "a"), self.sample_weights_single(w_b_ffn1, "b"),
            self.sample_weights_single(w_a_ffn2, "a"), self.sample_weights_single(w_b_ffn2, "b"),
        )
        return (sampled_w_a_q.cuda(), sampled_w_b_q.cuda(), sampled_w_a_k.cuda(), sampled_w_b_k.cuda(),
                sampled_w_a_v.cuda(), sampled_w_b_v.cuda(), sampled_w_a_o.cuda(), sampled_w_b_o.cuda(),
                sampled_w_a_ffn1.cuda(), sampled_w_b_ffn1.cuda(), sampled_w_a_ffn2.cuda(), sampled_w_b_ffn2.cuda())

    def sample_weights_single(self, w, type="a"):
        ws = []
        for sample_dim in self.search_lora_dim:
            # Set non-sampled weights to zero
            mask_weight = torch.zeros_like(w)
            if type == "a":
                mask_weight[:sample_dim, :] = 1
            elif type == "b":
                mask_weight[:, :sample_dim] = 1
            sampled_weight = mask_weight * w
            ws.append(sampled_weight)

        return torch.stack(ws, dim=0)

    def init_adapter_gumbel_weight(self, adapter_weights, epochs=100,):
        min_temp = 1
        temp = 4 - (4. - min_temp) / 120. * epochs
        max_indices_comput = []
        adapter_layer_gumbel_weights = []
        for t_layer_i, blk in enumerate(self.lora_vit.transformer.encoder.layer):
            if not self.retrain and not self.fix_adapter:
                gumbel_weights = gumbel_sample_weight(adapter_weights[t_layer_i],
                                                      gumbel_mask=self.adapter_arch_fix_mask[t_layer_i].cuda(),
                                                      sample_time=1, temp=temp,
                                                      GumbleSoftmax=self.GumbleSoftmax, zeta=self.last_zeta,
                                                      use_sparse=self.use_sparse, args=self.args,
                                                      ranks=self.search_adapter_dim)  # shape: [possible_location, candidate_dims]
                adapter_layer_gumbel_weights.append(gumbel_weights)
            else:
                weight_layer = adapter_weights[t_layer_i]
                if self.use_budget:
                    # self.last_zeta = -0.01560147723751139
                    bia = self.last_zeta * (torch.tensor(self.search_adapter_dim)) / 2
                    bia_expanded = bia.unsqueeze(0).repeat(weight_layer.shape[0], 1).cuda()
                    weight_layer = (weight_layer - bia_expanded)
                max_indices = torch.max(weight_layer, dim=-1).indices
                if self.fixed_layers is not None:
                    if t_layer_i in self.fixed_layers:
                        max_indices = torch.tensor([2, 2, 2, 2])
                max_indices_comput.append(max_indices)
                max_weights = torch.zeros_like(adapter_weights[t_layer_i])
                max_weights[np.arange(len(max_indices)), max_indices] = 1
                gumbel_weights = max_weights
                adapter_layer_gumbel_weights.append(gumbel_weights)
        if self.retrain and self.print_adapter_eval:
            print(max_indices_comput)
            print("adapter params after search: ", compute_adapter_search_size(max_indices_comput))
            self.print_adapter_eval = 0
        return adapter_layer_gumbel_weights

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor, epochs=0, eval_mode=False) -> Tensor:
        lora_weights = None
        delta_arch_weights = None
        if self.search_multiple:
            delta_arch_weights = self.lora_vit.transformer.encoder.delta_arch_weights
        if self.use_search:
            adapter_layer_gumbel_weights = []
            if self.is_LoRA and not self.fix_lora:
                w_a_ffn1, w_a_ffn2, w_b_ffn1, w_b_ffn2 = None, None, None, None

                if not self.search_multiple:
                    arch_weights = self.lora_vit.transformer.encoder.lora_arch_weights
                    if self.search_all_lora:
                        (w_a_qs, w_a_ks, w_a_vs, w_a_os, w_a_ffn1, w_a_ffn2,
                         w_b_qs, w_b_ks, w_b_vs, w_b_os, w_b_ffn1, w_b_ffn2, adapter_layer_gumbel_weights) = self.init_lora_samples(
                            arch_weights=arch_weights, epochs=epochs, eval_mode=eval_mode)
                    else:
                        w_a_qs, w_a_ks, w_a_vs, w_a_os, w_b_qs, w_b_ks, w_b_vs, w_b_os, adapter_layer_gumbel_weights = self.init_lora_samples(
                            arch_weights=arch_weights, epochs=epochs, eval_mode=eval_mode)
                else:
                    # in this case, we need to sample the gumbel weights of both lora and adapter together
                    if self.search_all_lora:
                        (w_a_qs, w_a_ks, w_a_vs, w_a_os, w_a_ffn1, w_a_ffn2,
                         w_b_qs, w_b_ks, w_b_vs, w_b_os, w_b_ffn1, w_b_ffn2, adapter_layer_gumbel_weights) = self.init_lora_samples(
                            arch_weights=delta_arch_weights, epochs=epochs, eval_mode=eval_mode)
                    else:
                        w_a_qs, w_a_ks, w_a_vs, w_a_os, w_b_qs, w_b_ks, w_b_vs, w_b_os, adapter_layer_gumbel_weights = self.init_lora_samples(
                            arch_weights=delta_arch_weights, epochs=epochs, eval_mode=eval_mode)
                if not self.search_all_lora:
                    lora_weights = [w_a_qs, w_a_ks, w_a_vs, w_a_os, w_b_qs, w_b_ks, w_b_vs, w_b_os]
                else:
                    lora_weights = [w_a_qs, w_a_ks, w_a_vs, w_a_os, w_a_ffn1, w_a_ffn2, w_b_qs, w_b_ks, w_b_vs, w_b_os, w_b_ffn1, w_b_ffn2]

            if len(adapter_layer_gumbel_weights)==0 and self.is_adapter:
                print("single sample adapter weights")
                adapter_arch_weights = self.lora_vit.transformer.encoder.adapter_arch_weights
                adapter_layer_gumbel_weights = self.init_adapter_gumbel_weight(adapter_weights=adapter_arch_weights, epochs=epochs)
            return self.lora_vit(x, lora_weights=lora_weights, adapter_gumbel_weights=adapter_layer_gumbel_weights)
        return self.lora_vit(x, lora_weights=None)


def compute_search_size(max_indices):
    dims = [0, 1, 4, 8]
    # dims = [0, 8]
    num_params = 0
    for max_index in max_indices:
        max_index = max_index.tolist()
        for max_ in max_index[:4]:
            num_params += dims[max_] * 768 * 2
        if len(max_index) > 4:
            for max_ in max_index[4:]:
                num_params += dims[max_] * 768 + dims[max_] * 3072

    return num_params

def compute_adapter_search_size(max_indices):
    dims = [0, 1, 4, 8]
    # dims = [0, 8]
    num_params = 0
    for max_index in max_indices:
        for max_ in max_index.tolist():
            num_params += dims[max_] * 768 * 2 + dims[max_] + 768
    return num_params

