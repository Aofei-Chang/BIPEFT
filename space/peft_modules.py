import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

from .peft_layers import Activations, LowRankLinear


class Mix_PEFT(nn.Module):
    def __init__(self, original_module, hidden_dim, super_rank=8, add_lora=False, add_bitfit=False, add_lnfit=False, add_adapter=False,
                                    lora_modules=None, bitfit_modules=None, lnfit_modules=None, adapter_modules=None, name=None, add_SA=False, add_PA=False,
                                    candidate_dims=[1, 4, 8], args=None, is_main_module=False):
        super().__init__()

        self.original_module = original_module
        self.add_lora = add_lora
        self.add_bitfit = add_bitfit
        self.add_lnfit = add_lnfit
        self.add_adapter = add_adapter
        self.add_SA = add_SA
        self.add_PA = add_PA

        self.lora, self.bitfit, self.lnfit, self.adapter, self.sadapter, self.padapter = [None] * 6
        self.lora_modules = lora_modules
        self.bitfit_modules = bitfit_modules
        self.lnfit_modules = lnfit_modules
        self.adapter_modules = adapter_modules
        self.candidate_dims = candidate_dims
        self.super_rank= super_rank

        self.hidden_dim = hidden_dim
        # parameters for evaluation and retraining
        self.args = args
        self.add_peft_modules()
        self.name=name

        self.is_main_module = is_main_module


    def freeze_arch(self, finalized_weight=None, retrain_flag=False):

        self.finalized_weight = finalized_weight
        self.retrain_flag = retrain_flag
        if finalized_weight is not None:
            for name, module in zip(['lora', 'bitfit', 'lnfit', 'adapter', 'sa', 'pa'], [self.lora, self.bitfit, self.lnfit, self.adapter, self.sadapter, self.padapter]):
                if module is not None:
                    module.freeze_arch(finalized_weight=self.finalized_weight[name], retrain_flag=retrain_flag)

    def add_peft_modules(self):
        if self.add_lora:
            self.lora = LoRA_ParallelLayer(LoRA_a=self.lora_modules[0], LoRA_b=self.lora_modules[1], candidate_dims=self.candidate_dims,
                                                        LoRA_dim=min(self.lora_modules[0].weight.shape[0], self.lora_modules[0].weight.shape[1]))
        if self.add_bitfit:
            self.bitfit = BitFitParallelLayer(hidden_dim=self.hidden_dim)

        if self.add_lnfit:
            self.lnfit = T5LayerNormParalleyLayer(hidden_dim=self.hidden_dim)

        if self.add_adapter:
            self.adapter = LowRankAdapterSequentialLayer(hidden_dim=self.hidden_dim, low_rank_rank=self.super_rank, candidate_dims=self.candidate_dims, zero_init=self.args.zero_lr_adapter)

        if self.add_SA:
            self.sadapter = SAdapterLayer(hidden_dim=self.hidden_dim, candidate_dims=self.candidate_dims)

        if self.add_PA:
            self.padapter = PAdapterLayer(hidden_dim=self.hidden_dim, candidate_dims=self.candidate_dims)

    def forward(self, x, gumbel_weights=None, dimension_mask=None, iterative_order=None, main_forward=None, *args, **kwargs):
        # we expect that the gumbel_weights and dimension_mask are all in dict version
        self.iterative_order = iterative_order
        self.main_forward = main_forward
        gumbel_weights_lora, gumbel_weights_adapter, gumbel_weights_bitfit, gumbel_weights_lnfit, gumbel_weights_sa, gumbel_weights_pa = [None]*6
        dimension_mask_lora, dimension_mask_adapter, dimension_mask_sa, dimension_mask_pa = None, None, None, None
        if gumbel_weights is not None:
            gumbel_weights_lora = gumbel_weights['lora']
            gumbel_weights_adapter = gumbel_weights['adapter']
            gumbel_weights_bitfit = gumbel_weights['bitfit']
            gumbel_weights_lnfit = gumbel_weights['lnfit']
            #sapa
            gumbel_weights_sa, gumbel_weights_pa = None, None
            if gumbel_weights.__contains__('sa'):
                gumbel_weights_sa = gumbel_weights['sa']
                gumbel_weights_pa = gumbel_weights['pa']

        if dimension_mask is not None:
            dimension_mask_lora = dimension_mask['lora']
            dimension_mask_adapter = dimension_mask['adapter']
            # sapa
            dimension_mask_sa, dimension_mask_pa = None, None
            if dimension_mask.__contains__('sa'):
                dimension_mask_sa = dimension_mask['sa']
                dimension_mask_pa = dimension_mask['pa']

        #forward-order: lora, lnfit, bitfit, adapter
        if self.adapter is not None and gumbel_weights is not None:
            hidden_flow = self.original_module(x, iterative_order=iterative_order, main_forward=main_forward, *args, **kwargs)
        else:
            #this case: for self-attn module forward
            if self.is_main_module and self.args is not None and self.args.use_search and not self.args.retrain:
                hidden_flow = self.original_module(x, iterative_order=iterative_order, main_forward=main_forward, *args,
                                                   **kwargs)
            else:
                hidden_flow = self.original_module(x, *args, **kwargs)
        #parallel
        if self.add_lora:
            lora_output = self.lora(x, gumbel_weights=gumbel_weights_lora, dimension_mask=dimension_mask_lora, iterative_order=iterative_order, main_forward=main_forward)
            hidden_flow = hidden_flow + lora_output
        if self.add_lnfit:
            lnfit_out = self.lnfit(x, gumbel_weights=gumbel_weights_lnfit)
            hidden_flow = hidden_flow + lnfit_out
        #sequential
        if self.add_bitfit:
            bitfit_out = self.bitfit(hidden_flow, gumbel_weights=gumbel_weights_bitfit)
            hidden_flow = hidden_flow + bitfit_out
        if self.add_adapter:
            adapter_out = self.adapter(hidden_flow, gumbel_weights=gumbel_weights_adapter, dimension_mask=dimension_mask_adapter, iterative_order=iterative_order, main_forward=main_forward, **kwargs)
            if isinstance(adapter_out, torch.Tensor) and isinstance(hidden_flow, torch.Tensor):
                hidden_flow = hidden_flow + adapter_out
            elif isinstance(hidden_flow, tuple):
                a = list(hidden_flow)
                a[0] = a[0] + adapter_out
                hidden_flow = tuple(a)
            else:
                hidden_flow = hidden_flow

        if self.add_SA:
            SA_adapter_out = self.sadapter(hidden_flow, gumbel_weights=gumbel_weights_sa, dimension_mask=dimension_mask_sa, iterative_order=iterative_order, main_forward=main_forward, **kwargs)
            if not self.add_PA:
                hidden_flow = hidden_flow + SA_adapter_out
            else:
                SA_adapter_out = hidden_flow + SA_adapter_out
        if self.add_PA:
            #different with SA: use x instead of hidden_flow
            PA_adapter_out = self.padapter(x, gumbel_weights=gumbel_weights_pa, dimension_mask=dimension_mask_pa, iterative_order=iterative_order, main_forward=main_forward, **kwargs)
            if self.add_SA:
                PA_adapter_out = PA_adapter_out + SA_adapter_out
            else:
                PA_adapter_out = PA_adapter_out + hidden_flow

            if isinstance(PA_adapter_out, torch.Tensor) and isinstance(hidden_flow, torch.Tensor):
                hidden_flow = PA_adapter_out
            elif isinstance(hidden_flow, tuple):
                a = list(hidden_flow)
                a[0] = PA_adapter_out
                hidden_flow = tuple(a)
            else:
                hidden_flow = hidden_flow

        return hidden_flow


class LoRA_ParallelLayer(nn.Module):
    def __init__(self, LoRA_a:nn.Linear, LoRA_b:nn.Linear, LoRA_dim=8, candidate_dims=[1, 4, 8], dropout=0):
        super().__init__()

        self.samples = {}

        self.super_LoRA_dim = LoRA_dim
        self.candidate_dims = candidate_dims
        self.candidate_dims_binary = [0, max(candidate_dims)]
        self.LoRA_a = LoRA_a
        self.LoRA_b = LoRA_b
        self.LoRA_dropout = nn.Dropout(dropout)
        # self.LoRA_a = nn.Parameter(torch.zeros(in_dim, LoRA_dim))
        # nn.init.kaiming_uniform_(self.LoRA_a, a=math.sqrt(5))
        # self.LoRA_b = nn.Parameter(torch.zeros(LoRA_dim, out_dim))
        self.retrain_flag = False
        self.fix_weight = None

    def freeze_arch(self, finalized_weight=None, retrain_flag=False):
        self.fix_weight = finalized_weight
        self.retrain_flag = retrain_flag

        if self.fix_weight is not None:
            binary_weight = self.fix_weight['binary'] #shape: [2]
            dim_weight = self.fix_weight['dim'] # shape: [3]
            if binary_weight is not None:
                self.binary_choice = torch.argmax(binary_weight).item() # 0 or 1
            else:
                self.binary_choice = 1
            dim_choice = torch.argmax(dim_weight).item() # dims
            self.dim_choice = self.candidate_dims[dim_choice]
            # print("lora dim: ", self.dim_choice)
        if self.retrain_flag:
            self.finalized_arch()

    def finalized_arch(self):
        if self.binary_choice == 1:
            w_a_sampled, w_b_sampled = self.LoRA_a.weight[:self.dim_choice, :], self.LoRA_b.weight[:, :self.dim_choice]
            self.LoRA_a.weight = nn.Parameter(w_a_sampled)
            # nn.init.kaiming_uniform_(self.LoRA_a.weight, a=math.sqrt(5))
            self.LoRA_b.weight = nn.Parameter(w_b_sampled)
            if self.dim_choice == 0:
                del self.LoRA_b
                del self.LoRA_a
        else:
            del self.LoRA_b
            del self.LoRA_a

    def sample_lora(self, w_a, w_b, gumbel_weights=None, dimension_mask=None):
        stacked_samples = self.sample_weights(w_a, w_b, dimension_mask=dimension_mask)
        (sampled_w_a, sampled_w_b) = stacked_samples
        gumbel_weights = gumbel_weights.unsqueeze(-1).unsqueeze(-1)
        # print(self.iterative_order, gumbel_weights, sampled_w_a,'jj')
        w_a_sampled = torch.sum(gumbel_weights * sampled_w_a, dim=0)
        w_b_sampled = torch.sum(gumbel_weights.detach() * sampled_w_b, dim=0)

        return w_a_sampled, w_b_sampled

    def forward(self, x, gumbel_weights=None, dimension_mask=None, iterative_order=None, main_forward=None):
        self.iterative_order = iterative_order
        self.main_forward = main_forward
        if self.retrain_flag and gumbel_weights is None:
            if self.binary_choice == 1 and self.dim_choice != 0:
                x = self.LoRA_b(self.LoRA_a(self.LoRA_dropout(x)))
            else:
                return 0
            return x
        if gumbel_weights is not None:
            w_a_sampled, w_b_sampled = (
                self.sample_lora(self.LoRA_a.weight, self.LoRA_b.weight, gumbel_weights=gumbel_weights, dimension_mask=dimension_mask))
            x = F.linear(input=F.linear(self.LoRA_dropout(x), weight=w_a_sampled), weight=w_b_sampled)
        elif self.fix_weight is not None:
            if self.binary_choice == 1:
                w_a_sampled, w_b_sampled = self.LoRA_a.weight[:self.dim_choice, :], self.LoRA_b.weight[:, :self.dim_choice]
                x = F.linear(input=F.linear(self.LoRA_dropout(x), weight=w_a_sampled), weight=w_b_sampled)
            else:
                return 0
        else:
            x = self.LoRA_b(self.LoRA_a(self.LoRA_dropout(x)))
        return x

    def sample_weights(self, w_a, w_b, dimension_mask=None):
        (sampled_w_a, sampled_w_b) = (
            self.sample_weights_single(w_a, "a", given_max_rank_id=dimension_mask), self.sample_weights_single(w_b, "b", given_max_rank_id=dimension_mask)
        )
        return sampled_w_a.cuda(), sampled_w_b.cuda()

    def sample_weights_single(self, w, type="a", given_max_rank_id=None):
        if w is None:
            return None
        ws = []
        # print(self.iterative_order, self.main_forward)
        if self.iterative_order is not None and (self.iterative_order or self.main_forward):
            search_lora_dim = self.candidate_dims_binary
            if given_max_rank_id is not None:
                search_lora_dim[1] = self.candidate_dims[given_max_rank_id]
        else:
            search_lora_dim = self.candidate_dims

        for sample_dim in search_lora_dim:
            # Set non-sampled weights to zero
            mask_weight = torch.zeros_like(w)
            if type == "a":
                mask_weight[:sample_dim, :] = 1
            elif type == "b":
                mask_weight[:, :sample_dim] = 1
            sampled_weight = mask_weight * w
            ws.append(sampled_weight)

        return torch.stack(ws, dim=0)
    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length *  np.prod(self.samples['weight'].size())
        return total_flops


class BitFitParallelLayer(nn.Module):
    def __init__(self, hidden_dim, init_method="zero"):
        super().__init__()
        self.init_method = init_method
        self.instantiated = False
        self.instantiate(hidden_dim=hidden_dim)

        self.retrain_flag = False
        self.binary_choice = 1

    def freeze_arch(self, finalized_weight=None, retrain_flag=False):
        self.fix_weight = finalized_weight
        self.retrain_flag = retrain_flag
        if self.fix_weight is not None:
            binary_weight = self.fix_weight['binary'] #shape: [2]
            self.binary_choice = torch.argmax(binary_weight).item() # 0 or 1
        if self.retrain_flag:
            self.finalized_arch()

    def finalized_arch(self):
        if self.binary_choice == 0:
            del self.BitFit_bias

    def instantiate(self, hidden_dim):
        if self.init_method == "zero":
            self.BitFit_bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise NotImplementedError
        self.instantiated = True

    def forward(self, output, gumbel_weights=None):
        if gumbel_weights is None and self.binary_choice == 0:
            return 0

        #here, gumbel weights are like [1, 0]
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError


        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            # print(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hidden_dim=self.hidden_dim)
        if gumbel_weights is not None:
            modified_output = torch.zeros_like(hiddens) + torch.sum(gumbel_weights.unsqueeze(-1) * torch.stack([torch.zeros_like(self.BitFit_bias).cuda(), self.BitFit_bias]), dim=0)
        else:
            modified_output = torch.zeros_like(hiddens) + self.BitFit_bias

        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output


class LowRankAdapterSequentialLayer(nn.Module):
    def __init__(self,
                 hidden_dim,
                 reduction_factor=32,
                 non_linearity="gelu_new",
                 low_rank_w_init="glorot-uniform",
                 low_rank_rank=8, zero_init=False,
                 candidate_dims=[1, 4, 8],
                 device=None,):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.zero_init = zero_init
        self.low_rank_w_init = low_rank_w_init
        self.low_rank_rank = low_rank_rank
        self.candidate_dims = candidate_dims
        self.device = device
        self.instantiated = False
        self.instantiate(hidden_dim=hidden_dim)

        self.retrain_flag = False
        self.fix_weight = None


    def freeze_arch(self, finalized_weight=None, retrain_flag=False):
        self.fix_weight = finalized_weight
        self.retrain_flag = retrain_flag
        if self.fix_weight is not None:
            binary_weight = self.fix_weight['binary'] #shape: [2]
            dim_weight = self.fix_weight['dim'] # shape: [3]
            if binary_weight is not None:
                self.binary_choice = torch.argmax(binary_weight).item() # 0 or 1
            else:
                self.binary_choice = 1
            self.dim_choice = torch.argmax(dim_weight).item() # dims
            self.dim_choice = self.candidate_dims[self.dim_choice]
        if self.retrain_flag:
            if self.binary_choice == 0:
                del self.Adapter_down_sampler
                del self.Adapter_up_sampler
            else:
                self.Adapter_down_sampler.finalized_arch(binary_choice=self.binary_choice, dim_choice=self.dim_choice)
                self.Adapter_up_sampler.finalized_arch(binary_choice=self.binary_choice, dim_choice=self.dim_choice)


    def instantiate(self, hidden_dim):
        self.down_sample_size = hidden_dim // self.reduction_factor
        self.activation = Activations(self.non_linearity.lower()).to(self.device)
        self.Adapter_down_sampler = LowRankLinear(hidden_dim, self.candidate_dims, self.down_sample_size,
                                          w_init=self.low_rank_w_init, zero_init=self.zero_init,
                                          rank=self.low_rank_rank).to(self.device)
        self.Adapter_up_sampler = LowRankLinear(self.down_sample_size, self.candidate_dims, hidden_dim,
                                        w_init=self.low_rank_w_init, zero_init=self.zero_init,
                                        rank=self.low_rank_rank).to(self.device)

        self.instantiated = True

    def forward(self, output, gumbel_weights=None, iterative_order=None, main_forward=None, **kwargs):
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)
        if self.retrain_flag:
            if self.binary_choice == 0 or self.dim_choice == 0:
                return 0
            else:
                z = self.Adapter_down_sampler(hiddens)
                z = self.activation(z)
                adapter_output = self.Adapter_up_sampler(z)
                modified_output = adapter_output
                # if isinstance(output, tuple):
                #     output = (modified_output,) + output[1:]
                # elif isinstance(output, torch.Tensor):
                #     output = modified_output
                # else:
                #     raise TypeError
                return modified_output

        if gumbel_weights is not None:
            # print("gumbel weights for adapter:", gumbel_weights)
            z = self.Adapter_down_sampler(hiddens, gumbel_weights=gumbel_weights, iterative_order=iterative_order, main_forward=main_forward)
            z = self.activation(z)
            adapter_output = self.Adapter_up_sampler(z, gumbel_weights=gumbel_weights, iterative_order=iterative_order, main_forward=main_forward)
            modified_output = adapter_output
        elif self.fix_weight is not None:
            if self.binary_choice == 1 and self.dim_choice != 0:
                z = self.Adapter_down_sampler(hiddens, dim_choice=self.dim_choice)
                z = self.activation(z)
                adapter_output = self.Adapter_up_sampler(z, dim_choice=self.dim_choice)
                modified_output = adapter_output
            else:
                return 0
        else:
            modified_output = 0
            if self.dim_choice != 0:
                z = self.Adapter_down_sampler(hiddens)
                z = self.activation(z)
                adapter_output = self.Adapter_up_sampler(z)
                modified_output = adapter_output
        # if isinstance(output, tuple):
        #     output = (modified_output,) + output[1:]
        # elif isinstance(output, torch.Tensor):
        #     output = modified_output
        # else:
        #     raise TypeError
        return modified_output

class T5LayerNormParalleyLayer(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6, init_method="zero"):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.variance_epsilon = eps
        self.instantiated = False
        self.init_method = init_method
        self.instantiate(hidden_dim=hidden_dim)

        self.retrain_flag = False
        self.binary_choice = 1

    def freeze_arch(self, finalized_weight=None, retrain_flag=False):
        self.fix_weight = finalized_weight
        self.retrain_flag = retrain_flag
        if self.fix_weight is not None:
            binary_weight = self.fix_weight['binary'] #shape: [2]
            self.binary_choice = torch.argmax(binary_weight).item() # 0 or 1
        if self.retrain_flag:
            self.finalized_arch()

    def finalized_arch(self):
        if self.binary_choice == 0:
            del self.LNfit_weight

    def instantiate(self, hidden_dim):
        if self.init_method == "zero":
            self.LNfit_weight = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise NotImplementedError
        self.instantiated = True

    def forward(self, hidden_states, gumbel_weights=None):

        if gumbel_weights is None and self.binary_choice == 0:
            return 0

        if not self.instantiated:
            self.hidden_dim = hidden_states.shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.LNfit_weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.LNfit_weight.dtype)

        if gumbel_weights is not None:
            weighted_LNfit_weight = torch.sum(gumbel_weights.unsqueeze(-1) * torch.stack([torch.zeros_like(self.LNfit_weight).cuda(), self.LNfit_weight]), dim=0)
            return weighted_LNfit_weight * hidden_states
        return self.LNfit_weight * hidden_states


class SAdapterLayer(nn.Module):
    r"""A layer of adapter tuning module.
    """
    def __init__(self, hidden_dim=1024, bottleneck_dim=8, non_linearity='gelu_new', candidate_dims=[1, 4, 8]):
        super().__init__()
        self.bottleneck_dim = max(candidate_dims)
        self.hidden_dim = hidden_dim
        self.instantiated = False
        self.non_linearity = non_linearity
        self.candidate_dims = candidate_dims
        self.candidate_dims_binary = [0, self.bottleneck_dim]

        self.instantiated = False
        self.instantiate(hidden_dim=hidden_dim)

        self.retrain_flag = False
        self.fix_weight = None

    def freeze_arch(self, finalized_weight=None, retrain_flag=False):
        self.fix_weight = finalized_weight
        self.retrain_flag = retrain_flag

        if self.fix_weight is not None:
            binary_weight = self.fix_weight['binary'] #shape: [2]
            dim_weight = self.fix_weight['dim'] # shape: [3]
            if binary_weight is not None:
                self.binary_choice = torch.argmax(binary_weight).item() # 0 or 1
            else:
                self.binary_choice = 1
            dim_choice = torch.argmax(dim_weight).item() # dims
            self.dim_choice = self.candidate_dims[dim_choice]
            # print("lora dim: ", self.dim_choice)
        if self.retrain_flag:
            self.finalized_arch()

    def finalized_arch(self):
        if self.binary_choice == 1:
            print("select SA")
            w_a_sampled, w_b_sampled = self.down_proj.weight[:self.dim_choice, :], self.up_proj.weight[:, :self.dim_choice]
            w_a_bias = self.down_proj.bias[:self.dim_choice]
            self.down_proj.weight = nn.Parameter(w_a_sampled)
            self.down_proj.bias = nn.Parameter(w_a_bias)
            # nn.init.kaiming_uniform_(self.LoRA_a.weight, a=math.sqrt(5))
            self.up_proj.weight = nn.Parameter(w_b_sampled)
        else:
            del self.down_proj
            del self.up_proj

    def instantiate(self, hidden_dim):
        self.down_proj = nn.Linear(hidden_dim, self.bottleneck_dim)
        # select non-linearity
        self.non_linear = Activations(self.non_linearity.lower())
        self.up_proj = nn.Linear(self.bottleneck_dim, self.hidden_dim)

        self.instantiated = True
        # initialize the weight, which is important for fast convergence and better performance.
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def sample_adapter(self, w_a, w_b, bias, gumbel_weights=None, dimension_mask=None):
        stacked_samples = self.sample_weights(w_a, w_b, bias, dimension_mask=dimension_mask)
        (sampled_w_a, sampled_w_b, sampled_bias) = stacked_samples
        gumbel_weights = gumbel_weights.unsqueeze(-1).unsqueeze(-1)
        # print(self.iterative_order, gumbel_weights, sampled_w_a,'jj')
        w_a_sampled = torch.sum(gumbel_weights * sampled_w_a, dim=0)
        w_b_sampled = torch.sum(gumbel_weights.detach() * sampled_w_b, dim=0)
        bias_sampled = torch.sum(gumbel_weights.squeeze(-1) * sampled_bias, dim=0)

        return w_a_sampled, w_b_sampled, bias_sampled

    def forward(self, output, gumbel_weights=None, dimension_mask=None, iterative_order=None, main_forward=None, **kwargs):
        r""" Get the hidden_states from the PLM's layer output, pass it into the adapter,
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        """
        self.iterative_order = iterative_order
        self.main_forward = main_forward
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError

        if self.retrain_flag and gumbel_weights is None:
            if self.binary_choice == 1:
                x = self.down_proj(hiddens)
                x = self.non_linear(x)
                x = self.up_proj(x)
            else:
                return 0
            return x

        if gumbel_weights is not None:
            # t1 = time.time()
            w_a_sampled, w_b_sampled, bias_sampled = (
                self.sample_adapter(self.down_proj.weight, self.up_proj.weight, bias=self.down_proj.bias, gumbel_weights=gumbel_weights,
                                    dimension_mask=dimension_mask))
            x = F.linear(hiddens, weight=w_a_sampled, bias=bias_sampled)
            x = self.non_linear(x)
            adapter_output = F.linear(x, weight=w_b_sampled, bias=self.up_proj.bias)
        elif self.fix_weight is not None:
            if self.binary_choice == 1:
                w_a_sampled, w_b_sampled = self.down_proj.weight[:self.dim_choice, :], self.up_proj.weight[:,
                                                                                       :self.dim_choice]
                bias_sampled = self.down_proj.bias[:self.dim_choice]
                x = F.linear(hiddens, weight=w_a_sampled, bias=bias_sampled)
                x = self.non_linear(x)
                adapter_output = F.linear(x, weight=w_b_sampled, bias=self.up_proj.bias)
            else:
                return 0
        else:
            x = self.down_proj(hiddens)
            x = self.non_linear(x)
            adapter_output = self.up_proj(x)

        return adapter_output

    def sample_weights(self, w_a, w_b, bias, dimension_mask=None):
        (sampled_w_a, sampled_w_b) = (
            self.sample_weights_single(w_a, "a", given_max_rank_id=dimension_mask), self.sample_weights_single(w_b, "b", given_max_rank_id=dimension_mask)
        )
        sampled_bias = self.sample_bias(bias, given_max_rank_id=dimension_mask)
        return sampled_w_a.cuda(), sampled_w_b.cuda(), sampled_bias.cuda()

    def sample_bias(self, bias, given_max_rank_id=None):
        ws = []
        if self.iterative_order is not None and (self.iterative_order or self.main_forward):
            search_lora_dim = self.candidate_dims_binary
            if given_max_rank_id is not None:
                search_lora_dim[1] = self.candidate_dims[given_max_rank_id]
        else:
            search_lora_dim = self.candidate_dims

        for sample_dim in search_lora_dim:
            # Set non-sampled weights to zero
            mask_weight = torch.zeros_like(bias)
            mask_weight[:sample_dim] = 1
            sampled_weight = mask_weight * bias
            ws.append(sampled_weight)

        return torch.stack(ws, dim=0)

    def sample_weights_single(self, w, type="a", given_max_rank_id=None):
        if w is None:
            return None
        ws = []
        if self.iterative_order is not None and (self.iterative_order or self.main_forward):
            search_lora_dim = self.candidate_dims_binary
            if given_max_rank_id is not None:
                search_lora_dim[1] = self.candidate_dims[given_max_rank_id]
        else:
            search_lora_dim = self.candidate_dims

        for sample_dim in search_lora_dim:
            # Set non-sampled weights to zero
            mask_weight = torch.zeros_like(w)
            if type == "a":
                mask_weight[:sample_dim, :] = 1
            elif type == "b":
                mask_weight[:, :sample_dim] = 1
            sampled_weight = mask_weight * w
            ws.append(sampled_weight)

        return torch.stack(ws, dim=0)


class PAdapterLayer(nn.Module):
    r"""A layer of adapter tuning module.
    """
    def __init__(self, hidden_dim=1024, bottleneck_dim=8, non_linearity='gelu_new', candidate_dims=[1, 4, 8]):
        super().__init__()
        self.bottleneck_dim = max(candidate_dims)
        self.hidden_dim = hidden_dim
        self.instantiated = False
        self.non_linearity = non_linearity
        self.candidate_dims = candidate_dims
        self.candidate_dims_binary = [0, self.bottleneck_dim]

        self.instantiated = False
        self.instantiate(hidden_dim=hidden_dim)
        self.retrain_flag = False
        self.fix_weight = None

    def freeze_arch(self, finalized_weight=None, retrain_flag=False):
        self.fix_weight = finalized_weight
        self.retrain_flag = retrain_flag

        if self.fix_weight is not None:
            binary_weight = self.fix_weight['binary'] #shape: [2]
            dim_weight = self.fix_weight['dim'] # shape: [3]
            if binary_weight is not None:
                self.binary_choice = torch.argmax(binary_weight).item() # 0 or 1
            else:
                self.binary_choice = 1
            dim_choice = torch.argmax(dim_weight).item() # dims
            self.dim_choice = self.candidate_dims[dim_choice]
            # print("lora dim: ", self.dim_choice)
        if self.retrain_flag:
            self.finalized_arch()

    def finalized_arch(self):
        if self.binary_choice == 1:
            print("select PA")
            w_a_sampled, w_b_sampled = self.down_proj.weight[:self.dim_choice, :], self.up_proj.weight[:, :self.dim_choice]
            w_a_bias = self.down_proj.bias[:self.dim_choice]
            self.down_proj.weight = nn.Parameter(w_a_sampled)
            self.down_proj.bias = nn.Parameter(w_a_bias)
            # nn.init.kaiming_uniform_(self.LoRA_a.weight, a=math.sqrt(5))
            self.up_proj.weight = nn.Parameter(w_b_sampled)
        else:
            del self.down_proj
            del self.up_proj

    def instantiate(self, hidden_dim):
        self.down_proj = nn.Linear(hidden_dim, self.bottleneck_dim)
        # select non-linearity
        self.non_linear = Activations(self.non_linearity.lower())
        self.up_proj = nn.Linear(self.bottleneck_dim, self.hidden_dim)

        self.instantiated = True
        # initialize the weight, which is important for fast convergence and better performance.
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def sample_adapter(self, w_a, w_b, bias, gumbel_weights=None, dimension_mask=None):
        stacked_samples = self.sample_weights(w_a, w_b, bias, dimension_mask=dimension_mask)
        (sampled_w_a, sampled_w_b, sampled_bias) = stacked_samples
        gumbel_weights = gumbel_weights.unsqueeze(-1).unsqueeze(-1)
        # print(self.iterative_order, gumbel_weights, sampled_w_a,'jj')
        w_a_sampled = torch.sum(gumbel_weights * sampled_w_a, dim=0)
        w_b_sampled = torch.sum(gumbel_weights.detach() * sampled_w_b, dim=0)
        bias_sampled = torch.sum(gumbel_weights.squeeze(-1) * sampled_bias, dim=0)

        return w_a_sampled, w_b_sampled, bias_sampled

    def forward(self, output, gumbel_weights=None, dimension_mask=None, iterative_order=None, main_forward=None, **kwargs):
        r""" Get the hidden_states from the PLM's layer output, pass it into the adapter,
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        """
        self.iterative_order = iterative_order
        self.main_forward = main_forward
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError

        if self.retrain_flag and gumbel_weights is None:
            if self.binary_choice == 1:
                x = self.down_proj(hiddens)
                x = self.non_linear(x)
                x = self.up_proj(x)
            else:
                return 0
            return x

        if gumbel_weights is not None:
            # t1 = time.time()
            w_a_sampled, w_b_sampled, bias_sampled = (
                self.sample_adapter(self.down_proj.weight, self.up_proj.weight, bias=self.down_proj.bias, gumbel_weights=gumbel_weights,
                                    dimension_mask=dimension_mask))
            x = F.linear(hiddens, weight=w_a_sampled, bias=bias_sampled)
            x = self.non_linear(x)
            adapter_output = F.linear(x, weight=w_b_sampled, bias=self.up_proj.bias)
        elif self.fix_weight is not None:
            if self.binary_choice == 1:
                w_a_sampled, w_b_sampled = self.down_proj.weight[:self.dim_choice, :], self.up_proj.weight[:,
                                                                                       :self.dim_choice]
                bias_sampled = self.down_proj.bias[:self.dim_choice]
                x = F.linear(hiddens, weight=w_a_sampled, bias=bias_sampled)
                x = self.non_linear(x)
                adapter_output = F.linear(x, weight=w_b_sampled, bias=self.up_proj.bias)
            else:
                return 0
        else:
            x = self.down_proj(hiddens)
            x = self.non_linear(x)
            adapter_output = self.up_proj(x)

        return adapter_output

    def sample_weights(self, w_a, w_b, bias, dimension_mask=None):
        (sampled_w_a, sampled_w_b) = (
            self.sample_weights_single(w_a, "a", given_max_rank_id=dimension_mask),
            self.sample_weights_single(w_b, "b", given_max_rank_id=dimension_mask)
        )
        sampled_bias = self.sample_bias(bias, given_max_rank_id=dimension_mask)
        return sampled_w_a.cuda(), sampled_w_b.cuda(), sampled_bias.cuda()

    def sample_bias(self, bias, given_max_rank_id=None):
        ws = []
        if self.iterative_order is not None and (self.iterative_order or self.main_forward):
            search_lora_dim = self.candidate_dims_binary
            if given_max_rank_id is not None:
                search_lora_dim[1] = self.candidate_dims[given_max_rank_id]
        else:
            search_lora_dim = self.candidate_dims

        for sample_dim in search_lora_dim:
            # Set non-sampled weights to zero
            mask_weight = torch.zeros_like(bias)
            mask_weight[:sample_dim] = 1
            sampled_weight = mask_weight * bias
            ws.append(sampled_weight)

        return torch.stack(ws, dim=0)

    def sample_weights_single(self, w, type="a", given_max_rank_id=None):
        if w is None:
            return None
        ws = []
        if self.iterative_order is not None and (self.iterative_order or self.main_forward):
            search_lora_dim = self.candidate_dims_binary
            if given_max_rank_id is not None:
                search_lora_dim[1] = self.candidate_dims[given_max_rank_id]
        else:
            search_lora_dim = self.candidate_dims

        for sample_dim in search_lora_dim:
            # Set non-sampled weights to zero
            mask_weight = torch.zeros_like(w)
            if type == "a":
                mask_weight[:sample_dim, :] = 1
            elif type == "b":
                mask_weight[:, :sample_dim] = 1
            sampled_weight = mask_weight * w
            ws.append(sampled_weight)

        return torch.stack(ws, dim=0)


class PrefixTuningSearch(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_size: int,
        prefix_length: int,
        candidate_dims=[1, 4, 8],
        small_prefix = False
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.prefix_length = prefix_length

        self.prefix_wte = nn.Embedding(self.prefix_length, self.input_size)
        self.bottle_dim = 2
        if small_prefix:
            self.bottle_dim = 1
        self.prefix_down = nn.Linear(self.input_size, self.bottle_dim)
        self.init_prefix_up()

        self.candidate_dims = candidate_dims
        self.candidate_dims_binary = [0, prefix_length]
        # self.prefix_control_trans = nn.Sequential(
        #     nn.Linear(self.input_size, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, self.n_layers * 2 * self.input_size),
        # )
        # self.relu = F.relu()
        self.dropout = nn.Dropout(0.1)
        self.binary_mask, self.dimension_mask = None, None
        self.retrain_flag = False
        self.subprefix_grad_dict = dict()
        # self.prefix_module_list = nn.ModuleList()

    def init_prefix_up(self):
        for i in range(self.n_layers):
            setattr(self, f"prefix_{i}_up", nn.Linear(self.bottle_dim, 2 * self.input_size))
            # self.subprefix_grad_dict[] = prefix_weight[:, i, :, :]
            # self.subprefix_grad_dict[f"layer_{i}_down"] = self.prefix_down.weight


    def freeze_arch(self, finalized_weight, retrain_flag):
        self.binary_mask, dimension_mask = finalized_weight['binary'], finalized_weight['dim']
        if dimension_mask is not None:
            dimension_mask = dimension_mask.tolist()
            dimension_mask = [self.candidate_dims[int(i)] for i in dimension_mask]
            dimension_mask = torch.tensor(dimension_mask)
        self.dimension_mask = dimension_mask
        self.retrain_flag = retrain_flag


    def eject(self, gumbel_weights=None, dimension_mask=None, iterative_order=None, main_forward=None):
        self.iterative_order = iterative_order
        self.main_forward = main_forward

        input_tokens = torch.arange(self.prefix_length).long().cuda()
        # input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1)
        embs = self.prefix_wte(input_tokens)
        key_values = self.prefix_down(embs)
        key_values = F.relu(key_values)
        # key_values = self.prefix_up(key_values)
        ups = []
        for i in range(self.n_layers):
            prefix_layer_i = getattr(self, f"prefix_{i}_up")
            up_temp = prefix_layer_i(key_values)
            ups.append(up_temp)
        key_values = torch.stack(ups, dim=0)
        # key_values = self.prefix_control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        # key_values = key_values.view(
        #     self.config.prefix_length * self.n_layers * 2 * self.input_size
        # )  # *2 for key and value
        # [layers, len(qv), prefix, dim]
        key_values = key_values.view(
                self.n_layers, 2, self.prefix_length, self.input_size
            )  # *2 for key and value

        if gumbel_weights is not None:
            # print(gumbel_weights,'bwhe')
            if dimension_mask is not None:
                dimension_mask = dimension_mask.tolist()
                dimension_mask = [self.candidate_dims[int(i)] for i in dimension_mask]
                dimension_mask = torch.tensor(dimension_mask)
            key_values = self.sample_prefix(key_values, gumbel_weights=gumbel_weights, dimension_mask=dimension_mask)
        elif self.binary_mask is not None:
            # print(self.dimension_mask, 'dniu')

            key_values = self.sample_prefix(key_values, gumbel_weights=self.binary_mask, dimension_mask=self.dimension_mask)

        return key_values

    def sample_prefix(self, prefix, gumbel_weights=None, dimension_mask=None):
        # dimension_mask: shape [layers]
        stacked_samples = self.sample_weights(prefix, dimension_mask=dimension_mask)
        sampled_w = stacked_samples
        #gumbel weights in shape [layers, candidates], sampled_w in shape [candidates, layers, len(qv), prefix, dim]
        gumbel_weights = gumbel_weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print(self.iterative_order, gumbel_weights, sampled_w_a,'jj')
        w_sampled = torch.sum(gumbel_weights * sampled_w, dim=0)

        return w_sampled

    def sample_weights(self, w_a, dimension_mask=None):
        sampled_w_a = self.sample_weights_single(w_a, dimension_mask)
        return sampled_w_a.cuda()

    def sample_weights_single(self, w, dimension_mask):
        if w is None:
            return None
        ws = []
        length = self.prefix_length
        dim = w.shape[-1]
        layers = w.shape[0]
        if dimension_mask is not None:  # here, adapt for the dimension mask in binary search stage
            range_tensor = torch.arange(w.size(2))[None, None, :, None]  # shape [1, 1, length, 1]
            range_tensor = range_tensor.expand(layers, 2, length, dim)  # Expand to match the shape of w
            M_expanded = dimension_mask[:, None, None, None]  # shape [layers, 1, 1, 1]
            M_expanded = M_expanded.expand(layers, 2, length, dim)  # Expand to match the shape of A
            mask = range_tensor.cuda() < M_expanded.cuda()
            w = w * mask.cuda()
        if self.iterative_order is not None and (self.iterative_order or self.main_forward):
            ws = [torch.zeros_like(w).cuda(), w]  # w will be dimension-masked if dimension mask is not None
        else:
            search_lora_dim = self.candidate_dims
            for sample_dim in search_lora_dim:
                # Set non-sampled weights to zero
                mask_weight = torch.zeros_like(w).cuda()
                mask_weight[:, :, :sample_dim, :] = 1
                sampled_weight = mask_weight * w
                ws.append(sampled_weight)

        return torch.stack(ws, dim=0)



    # def forward(self, batch_size):
    #     input_tokens = torch.arange(self.config.prefix_length).long()
    #     input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
    #     embs = self.wte(input_tokens)
    #     key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
    #     key_values = key_values.view(
    #         batch_size, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
    #     )  # *2 for key and value
    #     key_values = self.dropout(key_values)
    #     # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
    #     key_values = key_values.permute(2, 0, 3, 1, 4).split(2)
    #
    #     return key_values

class PrefixTuning(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_size: int,
        prefix_length: int,
        candidate_dims=[1, 4, 8]
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.prefix_length = prefix_length

        self.prefix_wte = nn.Embedding(self.prefix_length, self.input_size)
        self.prefix_down = nn.Linear(self.input_size, 32)
        self.prefix_up = nn.Linear(32, self.n_layers * 2 * self.input_size)

        self.candidate_dims = candidate_dims
        self.candidate_dims_binary = [0, prefix_length]
        # self.prefix_control_trans = nn.Sequential(
        #     nn.Linear(self.input_size, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, self.n_layers * 2 * self.input_size),
        # )
        # self.relu = F.relu()
        self.dropout = nn.Dropout(0.1)
        self.binary_mask, self.dimension_mask = None, None
        self.retrain_flag = False
        self.subprefix_grad_dict = dict()

    def upgrade_sub_grad(self):
        prefix_weight = self.prefix_up.grad.view(
            32, self.n_layers, 2, self.input_size
        )
        for i in range(self.n_layers):
            self.subprefix_grad_dict[f"layer_{i}_up"] = prefix_weight[:, i, :, :]
            self.subprefix_grad_dict[f"layer_{i}_down"] = self.prefix_down.weight


    # def name_subprefix(self):
    #     prefix_weight = self.prefix_up.weight.view(
    #             32, self.n_layers, 2, self.input_size
    #         )
    #     for i in range(self.n_layers):
    #         self.subprefix_dict[f"layer_{i}_up"] = prefix_weight[:, i, :, :]
    #         self.subprefix_dict[f"layer_{i}_down"] = self.prefix_down.weight

    def freeze_arch(self, finalized_weight, retrain_flag):
        self.binary_mask, self.dimension_mask = finalized_weight['binary'], finalized_weight['dim']
        self.retrain_flag = retrain_flag


    def eject(self, gumbel_weights=None, dimension_mask=None, iterative_order=None, main_forward=None):
        self.iterative_order = iterative_order
        self.main_forward = main_forward

        input_tokens = torch.arange(self.prefix_length).long().cuda()
        # input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1)
        embs = self.prefix_wte(input_tokens)
        key_values = self.prefix_down(embs)
        key_values = F.relu(key_values)
        key_values = self.prefix_up(key_values)
        # key_values = self.prefix_control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        # key_values = key_values.view(
        #     self.config.prefix_length * self.n_layers * 2 * self.input_size
        # )  # *2 for key and value
        # [layers, len(qv), prefix, dim]
        key_values = key_values.view(
                self.n_layers, 2, self.prefix_length, self.input_size
            )  # *2 for key and value

        if gumbel_weights is not None:
            key_values = self.sample_prefix(key_values, gumbel_weights=gumbel_weights, dimension_mask=dimension_mask)
        elif self.binary_mask is not None:
            key_values = self.sample_prefix(key_values, gumbel_weights=self.binary_mask, dimension_mask=self.dimension_mask)

        return key_values

    def sample_prefix(self, prefix, gumbel_weights=None, dimension_mask=None):
        # dimension_mask: shape [layers]
        stacked_samples = self.sample_weights(prefix, dimension_mask=dimension_mask)
        sampled_w = stacked_samples
        #gumbel weights in shape [layers, candidates], sampled_w in shape [candidates, layers, len(qv), prefix, dim]
        gumbel_weights = gumbel_weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print(self.iterative_order, gumbel_weights, sampled_w_a,'jj')
        w_sampled = torch.sum(gumbel_weights * sampled_w, dim=0)

        return w_sampled

    def sample_weights(self, w_a, dimension_mask=None):
        sampled_w_a = self.sample_weights_single(w_a, dimension_mask)
        return sampled_w_a.cuda()

    def sample_weights_single(self, w, dimension_mask):
        if w is None:
            return None
        ws = []
        length = self.prefix_length
        dim = w.shape[-1]
        layers = w.shape[0]
        if dimension_mask is not None:  # here, adapt for the dimension mask in binary search stage
            range_tensor = torch.arange(w.size(2))[None, None, :, None]  # shape [1, 1, length, 1]
            range_tensor = range_tensor.expand(layers, 2, length, dim)  # Expand to match the shape of w
            M_expanded = dimension_mask[:, None, None, None]  # shape [layers, 1, 1, 1]
            M_expanded = M_expanded.expand(layers, 2, length, dim)  # Expand to match the shape of A
            mask = range_tensor.cuda() < M_expanded.cuda()
            w = w * mask.cuda()
        if (self.iterative_order is not None and self.iterative_order) or self.main_forward:
            ws = [torch.zeros_like(w).cuda(), w]  # w will be dimension-masked if dimension mask is not None
        else:
            search_lora_dim = self.candidate_dims
            for sample_dim in search_lora_dim:
                # Set non-sampled weights to zero
                mask_weight = torch.zeros_like(w).cuda()
                mask_weight[:, :, :sample_dim, :] = 1
                sampled_weight = mask_weight * w
                ws.append(sampled_weight)

        return torch.stack(ws, dim=0)



    # def forward(self, batch_size):
    #     input_tokens = torch.arange(self.config.prefix_length).long()
    #     input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
    #     embs = self.wte(input_tokens)
    #     key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
    #     key_values = key_values.view(
    #         batch_size, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
    #     )  # *2 for key and value
    #     key_values = self.dropout(key_values)
    #     # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
    #     key_values = key_values.permute(2, 0, 3, 1, 4).split(2)
    #
    #     return key_values