
from torch.distributions import beta, bernoulli, gamma, dirichlet
import torch
import torch.nn.functional as F
import numpy as np
# torch.autograd.set_detect_anomaly(True)


def beta_sample_weight(str_weights, args=None):
    # in this case, the first weight will represent the selection of this module
    # since the beta sampling will only directly generate one value of the first concentration method
    # beta_sample = dirichlet.Dirichlet(F.elu(str_weights)+1).rsample()
    # beta_sample = dirichlet.Dirichlet(str_weights).rsample()
    # return beta_sample

    beta_sample = beta.Beta(str_weights[:, 0], str_weights[:, 1]).rsample().unsqueeze(-1)
    new_dim = 1 - beta_sample
    expanded_beta_sample = torch.cat([beta_sample, new_dim], dim=-1)
    return expanded_beta_sample



def bernoulli_sample(weights, temp=1, binary_mask=None, binary_prune_mask=None, dimension_search_mask=None, early_stop=False, dim_stage=False, no_gumbel=False):
    return gumbel_sample_weight(weights, temp=temp, binary_mask=binary_mask, binary_prune_mask=binary_prune_mask, dimension_search_mask=dimension_search_mask, early_stop=early_stop, dim_stage=dim_stage, no_gumbel=no_gumbel)


def gumbel_sample_weight(str_weights, temp=1., binary_mask=None, binary_prune_mask=None, dimension_search_mask=None, early_stop=False, dim_stage=False, no_gumbel=False):

    # if gumbel_mask is not None:
    #     possible_pos = gumbel_mask.shape[0]
    #     mask_dict = dict(zip(list(range(possible_pos)), gumbel_mask.tolist()))
    # if str_weights.shape[-1] == 2 and early_stop and not dim_stage:
    #     weight_all = torch.tensor([0, 1])
    #     while weight_all.dim() < str_weights.dim():
    #         weight_all = weight_all.unsqueeze(0)
    #     weight_all = weight_all.expand_as(str_weights).cuda()
    # else:
    if no_gumbel:
        weight_all = F.softmax(str_weights, dim=-1)
    else:
        weight_all = F.gumbel_softmax(str_weights, tau=temp, hard=True)
    if binary_mask is not None or binary_prune_mask is not None:
        if binary_prune_mask is not None and binary_mask is not None:
            binary_mask = binary_mask.cuda() | binary_prune_mask.cuda()
        elif binary_prune_mask is not None:
            binary_mask = binary_prune_mask
        # print(binary_mask.size(),"nfviw", weight_all.size())
        weight_all = weight_all * binary_mask.unsqueeze(-1).cuda()
    return weight_all



#old version, manual Gumbel-Softmax
def gumbel_sample(str_weights, sample_time=1, temp=1., flops_param=None, GumbleSoftmax=None, use_beta=False):
    weight_size = str_weights.size()
    str_weights = str_weights.view(1, -1)
    if not use_beta:
        str_weights = F.softmax(str_weights, dim=-1)
    if flops_param is not None:
        # flops_weights = F.softmax(flops_param.view(1, -1), dim=-1)
        flops_param = flops_param.view(1, -1)
        flops_weights = flops_param / (torch.sum(flops_param, dim=-1).view(1, -1) + 1e-7)
        str_weights = 0.5 * str_weights + 0.5 * flops_weights
    weight_output = GumbleSoftmax(str_weights, temp=temp, force_hard=True)
    for i in range(sample_time - 1):
        weights_t0 = GumbleSoftmax(str_weights, temp=temp, force_hard=True)
        weight_output = torch.cat([weight_output, weights_t0], 0)
    weight_output = torch.max(weight_output, 0)[0]
    weight_output = weight_output.view(weight_size)
    return weight_output