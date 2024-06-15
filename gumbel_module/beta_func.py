
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



def bernoulli_sample(weights, temp=1, GumbleSoftmax=None, use_beta=False, binary_mask=None, binary_search_mask=None, dimension_search_mask=None):
    return gumbel_sample_weight(weights, temp=temp, GumbleSoftmax=GumbleSoftmax, use_beta=use_beta, binary_mask=binary_mask, binary_search_mask=binary_search_mask, dimension_search_mask=dimension_search_mask)


def gumbel_sample_weight(str_weights, sample_time=1, temp=1., flops_param=None, GumbleSoftmax=None, use_beta=False, binary_mask=None, binary_search_mask=None, dimension_search_mask=None):

    # binary_search_mask: freeze some positions in the binary stage progressively, dimension_search_mask in the dimension search stage
    gumbel_mask = binary_mask

    max_indices = torch.max(str_weights, dim=-1).indices  # shape: [num_options]
    max_weights = torch.zeros_like(str_weights)
    max_weights[np.arange(len(max_indices)), max_indices] = 1

    mask_dict, binary_search_dict, dimension_search_dict = None, None, None #mask dict is for binary mask, in the dimension search stage
    if gumbel_mask is not None:
        possible_pos = gumbel_mask.shape[0]
        mask_dict = dict(zip(list(range(possible_pos)), gumbel_mask.tolist()))


    if binary_search_mask is not None:
        possible_pos = binary_search_mask.shape[0]
        if possible_pos >= 6:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask, gumbel_ffn1_mask, gumbel_ffn2_mask = binary_search_mask.tolist()
            binary_search_dict = {
                0: gumbel_q_mask,
                1: gumbel_k_mask,
                2: gumbel_v_mask,
                3: gumbel_o_mask,
                4: gumbel_ffn1_mask,
                5: gumbel_ffn2_mask
            }
        elif possible_pos == 4:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask = binary_search_mask.tolist()
            binary_search_dict = {
                0: gumbel_q_mask,
                1: gumbel_k_mask,
                2: gumbel_v_mask,
                3: gumbel_o_mask
            }
        elif possible_pos == 2:
            gumbel_q_mask, gumbel_v_mask = binary_search_mask.tolist()
            binary_search_dict = {
                0: gumbel_q_mask,
                1: gumbel_v_mask,
            }
    if dimension_search_mask is not None:
        possible_pos = dimension_search_mask.shape[0]
        if possible_pos >= 6:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask, gumbel_ffn1_mask, gumbel_ffn2_mask = dimension_search_mask.tolist()
            dimension_search_dict = {
                0: gumbel_q_mask,
                1: gumbel_k_mask,
                2: gumbel_v_mask,
                3: gumbel_o_mask,
                4: gumbel_ffn1_mask,
                5: gumbel_ffn2_mask
            }
        elif possible_pos == 4:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask = dimension_search_mask.tolist()
            dimension_search_dict = {
                0: gumbel_q_mask,
                1: gumbel_k_mask,
                2: gumbel_v_mask,
                3: gumbel_o_mask
            }
        elif possible_pos == 2:
            gumbel_q_mask, gumbel_v_mask = dimension_search_mask.tolist()
            dimension_search_dict = {
                0: gumbel_q_mask,
                1: gumbel_v_mask,
            }

    search_mask_dict = binary_search_dict if binary_search_dict is not None else dimension_search_dict
    if mask_dict is not None and mask_dict[0] == 0:
        weights = torch.zeros_like(str_weights[0, :].view(1, -1))
    elif search_mask_dict is not None and search_mask_dict[0]:
        weights = max_weights[0, :].view(1, -1)
    else:
        weights = gumbel_sample(str_weights[0, :].view(1, -1), sample_time, temp=temp,
                                flops_param=flops_param, GumbleSoftmax=GumbleSoftmax, use_beta=use_beta)

    for j in range(1, str_weights.shape[0]):
        if mask_dict is not None and mask_dict[j] == 0:
            weight_op = torch.zeros_like(str_weights[j, :].view(1, -1))
        elif search_mask_dict is not None and search_mask_dict[j]:
            weight_op = max_weights[j, :].view(1, -1)
        else:
            weight_op = gumbel_sample(str_weights[j, :].view(1, -1), sample_time,
                                           temp=temp, flops_param=flops_param, GumbleSoftmax=GumbleSoftmax, use_beta=use_beta)

        weights = torch.cat([weights, weight_op], 1)
    weights = weights.view(str_weights.size())
    return weights  # weights: shape [k, L]


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