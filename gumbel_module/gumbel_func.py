import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import fsolve

def global_comp(weights, M, ranks):
    updated_weights = weights[0].unsqueeze(dim=0)
    for j in range(1, weights.size()[0]):
        # Clone ranks to avoid inplace modification
        ranks_clone = ranks.clone()
        r_j = ranks[j]
        ranks_clone[j] = 0
        updated_weight = (M * ((weights * ranks).sum()) - (weights*ranks_clone).sum()) / r_j
        updated_weights = torch.cat((updated_weights, updated_weight.unsqueeze(dim=0)), dim=0)
    return updated_weights


def calculate_zeta_for_shifting(weights, max_num_param, ranks, adapter_dims=None, prompt_lens=None, last_zeta=0, fix_mask=None, freezed_weight=None,
                                budget_high=0, epoch=None, all_epochs=120):
    # Ensure weights is a PyTorch tensor
    # fix_mask: shape [layers, candidates]
    # weights:  shape [layers, candidates, search_dims]
    weights = torch.tensor(weights, dtype=torch.float32).cuda()
    beta = 1.0
    EPS = 1e-7
    # Convert ranks to a PyTorch tensor if not already
    ranks = torch.tensor(ranks, dtype=torch.float32).cuda()

    # Calculate params_for_cal as before, ensuring operations are on PyTorch tensors
    param_nums = torch.tensor([rank * 768 * 2 for rank in ranks], dtype=torch.float32)
    params_for_cal = param_nums.unsqueeze(0).unsqueeze(0).expand_as(weights[:, :4]).cuda()
    ffn_param_nums = torch.tensor([rank * (768+3072) for rank in ranks], dtype=torch.float32)
    ffn_params_for_cal = ffn_param_nums.unsqueeze(0).unsqueeze(0).expand_as(weights[:, 4:6]).cuda()
    adapter_params_for_cal = None
    prompt_params_for_cal = None
    if adapter_dims is not None:
        adapter_param_nums = torch.tensor([(rank * 768 * 2 + rank + 768) for rank in adapter_dims], dtype=torch.float32)
        adapter_params_for_cal = adapter_param_nums.unsqueeze(0).unsqueeze(0).expand_as(weights[:, 6:-1]).cuda()
    if prompt_lens is not None:
        prompt_param_nums = torch.tensor([rank * 768 for rank in prompt_lens], dtype=torch.float32)
        prompt_params_for_cal = prompt_param_nums.unsqueeze(0).unsqueeze(0).expand_as(weights[:, -1]).cuda()


    # if epoch is not None and epoch<=all_epochs:
    #     max_num_param = max_num_param + ((all_epochs-epoch)/all_epochs) * (budget_high/2 - max_num_param)
    # Define the function for fsolve, converting numpy arrays to PyTorch tensors
    def func(bia_np):
        # Convert numpy input to torch tensor
        bia = torch.tensor(bia_np, dtype=torch.float32, device='cuda') * ranks / 2
        bia_expanded = bia.unsqueeze(0).repeat(weights.shape[0], weights.shape[1], 1)

        x = (weights - bia_expanded) / beta
        x = torch.clamp(x, min=-200)  # Clamping instead of direct assignment

        softmax_shifted_weights = F.softmax(x, dim=-1)
        if fix_mask is not None:
            fix_mask2 = fix_mask.cuda()
            max_weights = freezed_weight
            # max_indices = torch.max(softmax_shifted_weights, dim=-1).indices
            # max_weights = torch.zeros_like(softmax_shifted_weights)
            # max_weights = max_weights.scatter_(dim=-1, index=max_indices.unsqueeze(-1), value=1)
            softmax_shifted_weights = softmax_shifted_weights * (1 - fix_mask2.unsqueeze(-1)) + max_weights.cuda() * fix_mask2.unsqueeze(-1)
        sum_params = (softmax_shifted_weights[:, :4] * params_for_cal).sum()
        sum_params += (softmax_shifted_weights[:, 4:6] * ffn_params_for_cal).sum()
        if adapter_params_for_cal is not None:
            sum_params += (softmax_shifted_weights[:, 6:-1] * adapter_params_for_cal).sum()
        if prompt_params_for_cal is not None:
            sum_params += (softmax_shifted_weights[:, -1] * prompt_params_for_cal).sum()
        # ret = (softmax_shifted_weights * params_for_cal).sum() - max_num_param * (1 - EPS)
        ret = sum_params - max_num_param * (1 - EPS)

        # Return as numpy array for fsolve
        return ret.cpu().detach().numpy()

    # Initial guess for fsolve, ensuring it's a numpy array
    if last_zeta == 0:
        last_zeta = np.array([last_zeta], dtype=np.float32)

    # Solve using fsolve
    bia_res_np = fsolve(func, last_zeta)[0]

    return bia_res_np

def calculate_softmax_under_budget(weights, zeta, use_budget=True, beta=1, ranks=None):
    # in this case, the weights are 1-D: [candidates], ranks: torch.tensor(candidates)
    if use_budget:
        bia = zeta * (ranks / 2)
        x = (weights - bia) / beta
        # x = (weights - zeta) / beta
        x[x < -200] = -200
        weights = F.softmax(x, dim=-1)
    else:
        weights = F.softmax(weights, dim=-1)
    return weights

def gumbel_multiple_weight(str_weights, gumbel_mask=None, sample_time=1, temp=1., flops_param=None, GumbleSoftmax=None, use_sparse=False, zeta=0, ranks=None, args=None):
    # str_weights: shape [k, L]
    #gumbel_mask: shape # 1d: [num_options]
    use_budget = args.use_budget
    possible_pos = gumbel_mask.shape[0]

    max_indices = torch.max(str_weights, dim=-1).indices  # shape: [num_options]
    max_weights = torch.zeros_like(str_weights)
    max_weights[np.arange(len(max_indices)), max_indices] = 1

    new_weights = []
    expectation = 0
    expectation_detach = 0
    expectations = []
    ranks = torch.tensor(ranks).cuda()
    # ranks = torch.tensor([0, 8]).cuda()
    mask_dict = dict(zip(list(range(possible_pos)), [0]*possible_pos))
    if gumbel_mask is not None:
        # gumbel_ffn1_mask, gumbel_ffn2_mask, gumbel_prompt_mask, gumbel_adapter1_mask, gumbel_adapter2_mask = [None] * 5
        gumbel_ffn1_mask, gumbel_ffn2_mask, gumbel_prompt_mask, gumbel_adapter1_mask, gumbel_adapter2_mask = [0] * 5
        gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask = gumbel_mask.tolist()[:4]
        if args.is_adapter and args.search_all_lora:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask, gumbel_ffn1_mask, gumbel_ffn2_mask, gumbel_adapter1_mask, gumbel_adapter2_mask = gumbel_mask.tolist()[:8]
        elif args.is_adapter:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask, gumbel_adapter1_mask, gumbel_adapter2_mask = gumbel_mask.tolist()[:6]
        if args.is_prompt:
            gumbel_prompt_mask = gumbel_mask.tolist()[-1]
        mask_dict = {
            0: gumbel_q_mask,
            1: gumbel_k_mask,
            2: gumbel_v_mask,
            3: gumbel_o_mask, 4: gumbel_ffn1_mask, 5: gumbel_ffn2_mask,
            6: gumbel_adapter1_mask, 7: gumbel_adapter2_mask, 8: gumbel_prompt_mask
        }

    # ranks = torch.tensor([0, 8]).cuda()
    if use_sparse:
        for j in range(0, str_weights.size()[0]):
            str_weight = str_weights[j, :]
            str_weight = str_weight.view(-1)
            # str_weight = F.softmax(str_weight, dim=-1)
            str_weight = calculate_softmax_under_budget(str_weight, zeta=zeta, use_budget=use_budget, ranks=ranks)
            new_weights.append(str_weight)
            element_exp = str_weight * ranks
            # element_exp = str_weight
            str_expectation = (element_exp).sum()
            if mask_dict[0] > 0:
                expectation += str_expectation.detach()
            else:
                expectation += str_expectation
            expectations.append(element_exp)
            expectation_detach += str_expectation.detach()
        weights_sparse = []
        M = expectation_detach / expectation
        for j in range(len(new_weights)):
            new_weight = new_weights[j]
            # print(new_weight, "weight before")
            # element_exp = expectations[j]
            # updated_elements = element_exp[1:] * (expectation_detach / expectation)
            # new_weight_updated = torch.cat((new_weight[:1], updated_elements), dim=0)
            # weights_sparse.append(new_weight_updated)
            new_weight_updated = global_comp(new_weight, M, ranks)
            weights_sparse.append(new_weight_updated)
        # print(weights_sparse,"weight")
        # print(new_weights,"weigwwht")
    if use_sparse:
        if mask_dict[0] > 0:
            weights = max_weights[0, :].view(1, -1)
        else:
            weights = gumbel_sample(weights_sparse[0].view(1, -1), sample_time, temp=temp,
                                     flops_param=flops_param, GumbleSoftmax=GumbleSoftmax)
    else:
        if mask_dict[0] > 0:
            weights = max_weights[0, :].view(1, -1)
            # print("use_manual", weights)
        else:
            weights = gumbel_sample(str_weights[0, :].view(1, -1), sample_time, temp=temp,
                                flops_param=flops_param, GumbleSoftmax=GumbleSoftmax)
            # print(weights, "sample 0")
    if use_sparse:
        for j in range(1, len(weights_sparse)):
            if mask_dict[j] > 0:
                weight_op = max_weights[j, :].view(1, -1)
            else:
                weight_op = gumbel_sample(weights_sparse[j].view(1, -1), sample_time,
                                           temp=temp, flops_param=flops_param, GumbleSoftmax=GumbleSoftmax, use_sparse=use_sparse)
            weights = torch.cat([weights, weight_op], 1)
    else:
        for j in range(1, str_weights.shape[0]):
            if mask_dict[j] > 0:
                weight_op = max_weights[j, :].view(1, -1)
                # print("use_manualdf", weights)
            else:
                weight_op = gumbel_sample(str_weights[j, :].view(1, -1), sample_time,
                                               temp=temp, flops_param=flops_param, GumbleSoftmax=GumbleSoftmax, use_sparse=use_sparse)
            # print(weight_op, "sample 1")
            weights = torch.cat([weights, weight_op], 1)
    weights = weights.view(str_weights.size())
    return weights  # weights: shape [k, L]



def gumbel_sample_weight(str_weights, gumbel_mask=None, sample_time=1, temp=1., flops_param=None, GumbleSoftmax=None, use_sparse=False, zeta=0, ranks=None, args=None):
    # str_weights: shape [k, L]
    #gumbel_mask: shape # 1d: [num_options]
    use_budget = args.use_budget
    possible_pos = gumbel_mask.shape[0]

    max_indices = torch.max(str_weights, dim=-1).indices  # shape: [num_options]
    max_weights = torch.zeros_like(str_weights)
    max_weights[np.arange(len(max_indices)), max_indices] = 1

    new_weights = []
    expectation = 0
    expectation_detach = 0
    expectations = []
    ranks = torch.tensor(ranks).cuda()
    # ranks = torch.tensor([0, 8]).cuda()
    mask_dict = dict(zip(list(range(possible_pos)), [0]*possible_pos))
    if gumbel_mask is not None:
        if possible_pos >=6:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask, gumbel_ffn1_mask, gumbel_ffn2_mask = gumbel_mask.tolist()
            mask_dict = {
                0: gumbel_q_mask,
                1: gumbel_k_mask,
                2: gumbel_v_mask,
                3: gumbel_o_mask,
                4: gumbel_ffn1_mask,
                5: gumbel_ffn2_mask
            }
        elif possible_pos >= 4:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask = gumbel_mask.tolist()
            mask_dict = {
                0: gumbel_q_mask,
                1: gumbel_k_mask,
                2: gumbel_v_mask,
                3: gumbel_o_mask
            }
        elif possible_pos == 2:
            gumbel_adapter1_mask, gumbel_adapter2_mask = gumbel_mask.tolist()
            mask_dict = {
                0: gumbel_adapter1_mask,
                1: gumbel_adapter2_mask,
            }

    # ranks = torch.tensor([0, 8]).cuda()
    if use_sparse:
        # print("use+sss")
        for j in range(0, str_weights.size()[0]):
            str_weight = str_weights[j, :]
            str_weight = str_weight.view(-1)
            # str_weight = F.softmax(str_weight, dim=-1)
            str_weight = calculate_softmax_under_budget(str_weight, zeta=zeta, use_budget=use_budget, ranks=ranks)
            new_weights.append(str_weight)
            element_exp = str_weight * ranks
            # element_exp = str_weight
            str_expectation = (element_exp).sum()
            if mask_dict[0] > 0:
                expectation += str_expectation.detach()
            else:
                expectation += str_expectation
            expectations.append(element_exp)
            expectation_detach += str_expectation.detach()
        weights_sparse = []
        M = expectation_detach / expectation
        for j in range(len(new_weights)):
            new_weight = new_weights[j]
            # print(new_weight, "weight before")
            # element_exp = expectations[j]
            # updated_elements = element_exp[1:] * (expectation_detach / expectation)
            # new_weight_updated = torch.cat((new_weight[:1], updated_elements), dim=0)
            # weights_sparse.append(new_weight_updated)
            new_weight_updated = global_comp(new_weight, M, ranks)
            weights_sparse.append(new_weight_updated)
        # print(weights_sparse,"weight")
        # print(new_weights,"weigwwht")
    if use_sparse:
        if mask_dict[0] > 0:
            weights = max_weights[0, :].view(1, -1)
        else:
            weights = gumbel_sample(weights_sparse[0].view(1, -1), sample_time, temp=temp,
                                     flops_param=flops_param, GumbleSoftmax=GumbleSoftmax)
    else:
        if mask_dict[0] > 0:
            weights = max_weights[0, :].view(1, -1)
            # print("use_manual", weights)
        else:
            weights = gumbel_sample(str_weights[0, :].view(1, -1), sample_time, temp=temp,
                                flops_param=flops_param, GumbleSoftmax=GumbleSoftmax)
            # print(weights, "sample 0")
    if use_sparse:
        for j in range(1, len(weights_sparse)):
            if mask_dict[j] > 0:
                weight_op = max_weights[j, :].view(1, -1)
            else:
                weight_op = gumbel_sample(weights_sparse[j].view(1, -1), sample_time,
                                           temp=temp, flops_param=flops_param, GumbleSoftmax=GumbleSoftmax, use_sparse=use_sparse)
            weights = torch.cat([weights, weight_op], 1)
    else:
        for j in range(1, str_weights.shape[0]):
            if mask_dict[j] > 0:
                weight_op = max_weights[j, :].view(1, -1)
                # print("use_manualdf", weights)
            else:
                weight_op = gumbel_sample(str_weights[j, :].view(1, -1), sample_time,
                                               temp=temp, flops_param=flops_param, GumbleSoftmax=GumbleSoftmax, use_sparse=use_sparse)
            # print(weight_op, "sample 1")
            weights = torch.cat([weights, weight_op], 1)
    weights = weights.view(str_weights.size())
    return weights  # weights: shape [k, L]


def gumbel_sample(str_weights, sample_time=1, temp=1., flops_param=None, GumbleSoftmax=None, use_sparse=False):
    weight_size = str_weights.size()
    if not use_sparse:
        str_weights = str_weights.view(1, -1)
        str_weights = F.softmax(str_weights, dim=-1)
        # str_weights = F.relu(str_weights)
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


def measure_entropy(weights):
    # Apply softmax to convert tensor values to probabilities along the last dimension
    # This assumes your tensor contains logits or unnormalized scores.
    probabilities = F.softmax(weights, dim=-1)

    # Calculate the log of probabilities, setting elements where probabilities are 0 to 0 in log_probs to avoid NaNs
    log_probs = torch.where(probabilities > 0, probabilities.log(), torch.zeros_like(probabilities))

    # Calculate the entropy using the formula -sum(p * log(p)) along the last dimension
    entropy = -torch.sum(probabilities * log_probs, dim=-1)

    return entropy

def progressive_prune(modules_entropy, weight_fix_mask, prune_num=4, epochs_num=120, current_epoch=0):
    # for the weight_mask matrix, mask the selected index with a high value.

    return weight_fix_mask