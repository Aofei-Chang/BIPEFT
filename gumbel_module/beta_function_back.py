
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
        if possible_pos >= 6:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask, gumbel_ffn1_mask, gumbel_ffn2_mask = gumbel_mask.tolist()
            mask_dict = {
                0: gumbel_q_mask,
                1: gumbel_k_mask,
                2: gumbel_v_mask,
                3: gumbel_o_mask,
                4: gumbel_ffn1_mask,
                5: gumbel_ffn2_mask
            }
        elif possible_pos == 4:
            gumbel_q_mask, gumbel_k_mask, gumbel_v_mask, gumbel_o_mask = gumbel_mask.tolist()
            mask_dict = {
                0: gumbel_q_mask,
                1: gumbel_k_mask,
                2: gumbel_v_mask,
                3: gumbel_o_mask
            }
        elif possible_pos == 2:
            gumbel_q_mask, gumbel_v_mask = gumbel_mask.tolist()
            mask_dict = {
                0: gumbel_q_mask,
                1: gumbel_v_mask,
            }

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

def init_gumbel_weights(self, epochs=100, eval_mode=False):
    arch_weights_bianry_encoder_matrix = self.arch_weights_bianry_encoder_matrix
    arch_weights_bianry_decoder_matrix = self.arch_weights_bianry_decoder_matrix
    arch_weights_bianry_encoder = self.arch_weights_bianry_encoder
    arch_weights_bianry_decoder = self.arch_weights_bianry_decoder
    arch_weights_multi_encoder = self.arch_weights_multi_encoder
    arch_weights_multi_decoder = self.arch_weights_multi_decoder
    arch_weights_bianry_final_norm = self.arch_weights_bianry_final_norm

    dimension_mask_encoder, binary_mask_encoder, dimension_mask_decoder, binary_mask_decoder = None, None, None, None
    binary_search_mask, dimension_search_mask = None, None
    if self.iter_search:
        if self.iterative_order or self.main_forward:  # iterative_order=True, means binary search stage
            self.modify_arch_mask(binary_stage=True)
            dimension_mask_encoder, dimension_mask_decoder, binary_mask_encoder, binary_mask_decoder = self.dimension_mask_encoder, self.dimension_mask_decoder, None, None
            # if self.progressive_fix:
            #     binary_search_mask, dimension_search_mask = self.binary_search_mask, None
        else:
            self.modify_arch_mask(binary_stage=False)
            dimension_mask_encoder, dimension_mask_decoder, binary_mask_encoder, binary_mask_decoder = None, None, self.binary_mask_encoder, self.binary_mask_decoder
            # if self.progressive_fix:
            #     binary_search_mask, dimension_search_mask = None, self.dimension_search_mask

    if self.use_search:
        temp = 5 - (5. - 1.) / self.all_epochs * epochs
        max_indices_comput = []
        gumbel_weights_encoder_layers, gumbel_weights_decoder_layers, gumbel_weights_final_norm_all = [], [], []
        gumbel_weights_encoder_binary_layers, gumbel_weights_decoder_binary_layers = [], []

        if self.args.use_beta:
            arch_weights_bianry_final_norm = dirichlet.Dirichlet(F.elu(arch_weights_bianry_final_norm.clone()) + 1).rsample()
            beta_arch_weights_bianry_encoder = dirichlet.Dirichlet(F.elu(arch_weights_bianry_encoder.clone()) + 1).rsample()
            beta_arch_weights_bianry_decoder = dirichlet.Dirichlet(F.elu(arch_weights_bianry_decoder.clone()) + 1).rsample()
            if not self.iter_search:
                beta_sample_weights_encoder = dirichlet.Dirichlet(F.elu(arch_weights_multi_encoder.clone()) + 1).rsample()
                beta_sample_weights_decoder = dirichlet.Dirichlet(F.elu(arch_weights_multi_decoder.clone()) + 1).rsample()
            else:
                if self.iterative_order or self.main_forward: #binary search stage
                    beta_sample_weights_encoder = dirichlet.Dirichlet(F.elu(arch_weights_bianry_encoder_matrix.clone()) + 1).rsample()
                    beta_sample_weights_decoder = dirichlet.Dirichlet(F.elu(arch_weights_bianry_decoder_matrix.clone()) + 1).rsample()
                else:
                    beta_sample_weights_encoder = dirichlet.Dirichlet(F.elu(arch_weights_multi_encoder.clone()) + 1).rsample()
                    beta_sample_weights_decoder = dirichlet.Dirichlet(F.elu(arch_weights_multi_decoder.clone()) + 1).rsample()
        #final norm, no layers
        if not eval_mode and not self.retrain:
            gumbel_weights_final_norm = bernoulli_sample(arch_weights_bianry_final_norm, temp = temp, GumbleSoftmax = self.GumbleSoftmax, use_beta = self.args.use_beta)
        else:
            max_indices = torch.max(arch_weights_bianry_final_norm, dim=-1).indices
            max_weights = torch.zeros_like(arch_weights_bianry_final_norm)
            max_weights[np.arange(len(max_indices)), max_indices] = 1
            gumbel_weights_final_norm = max_weights

        for t_layer_i, blk in enumerate(self.t5_model.encoder.block):
            # print(arch_weights[t_layer_i].size(),"arch")
            if not eval_mode and not self.retrain:
                if self.args.use_beta:
                    weight_layer_binary_encoder = beta_arch_weights_bianry_encoder[t_layer_i]
                    weight_layer_binary_decoder = beta_arch_weights_bianry_decoder[t_layer_i]
                    weight_layer_encoder = beta_sample_weights_encoder[t_layer_i]
                    weight_layer_decoder = beta_sample_weights_decoder[t_layer_i]
                else:
                    weight_layer_binary_encoder = arch_weights_bianry_encoder[t_layer_i]
                    weight_layer_binary_decoder = arch_weights_bianry_decoder[t_layer_i]
                    if not self.iter_search or self.iterative_order or self.main_forward:  # binary search stage
                        weight_layer_encoder = arch_weights_bianry_encoder_matrix[t_layer_i]
                        weight_layer_decoder = arch_weights_bianry_decoder_matrix[t_layer_i]
                    else:
                        weight_layer_encoder = arch_weights_multi_encoder[t_layer_i]
                        weight_layer_decoder = arch_weights_multi_decoder[t_layer_i]
                binary_mask_layer_encoder = binary_mask_encoder[t_layer_i] if binary_mask_encoder is not None else None
                binary_mask_layer_decoder = binary_mask_decoder[t_layer_i] if binary_mask_decoder is not None else None

                # binary_search_mask_layer = binary_search_mask[t_layer_i] if binary_search_mask is not None else None
                # dimension_search_mask_layer = dimension_search_mask[t_layer_i] if dimension_search_mask is not None else None
                gumbel_weights_encoder_matrix = bernoulli_sample(weight_layer_encoder, temp=temp, GumbleSoftmax=self.GumbleSoftmax,
                                                  use_beta=self.args.use_beta, binary_mask=binary_mask_layer_encoder)
                gumbel_weights_decoder_matrix = bernoulli_sample(weight_layer_decoder, temp=temp, GumbleSoftmax=self.GumbleSoftmax,
                                                  use_beta=self.args.use_beta, binary_mask=binary_mask_layer_decoder)
                                                  # binary_search_mask=binary_search_mask_layer,
                                                  # dimension_search_mask=dimension_search_mask_layer)  # shape: [possible_location, candidate_dims]
                gumbel_weights_encoder_binary = bernoulli_sample(weight_layer_binary_encoder, temp=temp,
                                                                 GumbleSoftmax=self.GumbleSoftmax,
                                                                 use_beta=self.args.use_beta)
                gumbel_weights_decoder_binary = bernoulli_sample(weight_layer_binary_decoder, temp=temp,
                                                                 GumbleSoftmax=self.GumbleSoftmax,
                                                                 use_beta=self.args.use_beta)


            # If we only want few lora layer instead of all
            else:
                # bianry modules
                weight_layer_binary_encoder = arch_weights_bianry_encoder[t_layer_i]
                weight_layer_binary_decoder = arch_weights_bianry_decoder[t_layer_i]
                max_indices_binary_encoder = torch.max(weight_layer_binary_encoder, dim=-1).indices
                max_indices_binary_decoder = torch.max(weight_layer_binary_decoder, dim=-1).indices
                gumbel_weights_encoder_binary = torch.zeros_like(weight_layer_binary_encoder)
                gumbel_weights_decoder_binary = torch.zeros_like(weight_layer_binary_decoder)
                gumbel_weights_encoder_binary[np.arange(len(max_indices_binary_encoder)), max_indices_binary_encoder] = 1
                gumbel_weights_decoder_binary[np.arange(len(max_indices_binary_decoder)), max_indices_binary_decoder] = 1
                # matrix modules
                weight_layer_binary_encoder_matrix = arch_weights_bianry_encoder_matrix[t_layer_i]
                weight_layer_binary_decoder_matrix = arch_weights_bianry_decoder_matrix[t_layer_i]
                max_indices_binary_encoder_matrix = torch.max(weight_layer_binary_encoder_matrix, dim=-1).indices
                max_indices_binary_decoder_matrix = torch.max(weight_layer_binary_decoder_matrix, dim=-1).indices
                gumbel_weights_encoder_matrix = torch.zeros_like(weight_layer_binary_encoder_matrix)
                gumbel_weights_decoder_matrix = torch.zeros_like(weight_layer_binary_decoder_matrix)
                gumbel_weights_encoder_matrix[
                    np.arange(len(max_indices_binary_encoder_matrix)), max_indices_binary_encoder_matrix] = 1
                gumbel_weights_decoder_matrix[
                    np.arange(len(max_indices_binary_decoder_matrix)), max_indices_binary_decoder_matrix] = 1
                # then update dimension mask for matrix based modules
                self.modify_arch_mask(binary_stage=True)

            gumbel_weights_encoder_layers.append(gumbel_weights_encoder_matrix.cuda())
            gumbel_weights_encoder_binary_layers.append(gumbel_weights_encoder_binary.cuda())
            gumbel_weights_decoder_layers.append(gumbel_weights_decoder_matrix.cuda())
            gumbel_weights_decoder_binary_layers.append(gumbel_weights_decoder_binary.cuda())
        # if (eval_mode or self.retrain) and self.print_eval:
        #     print("params after search: ", compute_search_size(max_indices_comput))
        #     self.print_eval = 0
        return {"encoder": gumbel_weights_encoder_layers if len(gumbel_weights_encoder_layers) else None,
                "encoder_binary": gumbel_weights_encoder_binary_layers if len(gumbel_weights_encoder_binary_layers) else None,
                "decoder": gumbel_weights_decoder_layers if len(gumbel_weights_decoder_layers) else None,
                "decoder_binary": gumbel_weights_decoder_binary_layers if len(gumbel_weights_decoder_binary_layers) else None,
                "final_norm": gumbel_weights_final_norm}

    else:
        print("no sampling lora")