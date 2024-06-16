import torch
import math
import torch.nn as nn
import numpy as np
# import torch.nn.functional as F

from transformers.activations import get_activation

def glorot_normal(tensor: torch.Tensor):
    return torch.nn.init.xavier_normal_(tensor, gain=math.sqrt(2))

def glorot_uniform(tensor: torch.Tensor):
    return torch.nn.init.xavier_uniform_(tensor, gain=math.sqrt(2))

class Activations(nn.Module):
    """
    Implementation of various activation function. Copied from open-source project AdapterHub
    """

    def __init__(self, activation_type):
        self.activation_type = activation_type
        if activation_type.lower() == "relu":
            self.f = nn.functional.relu
        elif activation_type.lower() == "tanh":
            self.f = torch.tanh
        elif activation_type.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif activation_type.lower() == "gelu_new":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif activation_type.lower() == "gelu_orig":
            self.f = nn.functional.gelu
        elif activation_type.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        else:
            self.f = get_activation(activation_type)

        super().__init__()

    def forward(self, x):
        return self.f(x)

    def __repr__(self):
        return self.activation_type


class LowRankLinear(torch.nn.Module):
    def __init__(self, input_dim: int, candidate_dims: list, output_dim: int, rank: int = 1,
                 bias: bool = True, w_init: str = "glorot-uniform", zero_init:bool = False):
        super(LowRankLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.bias = bias
        self.zero_init = zero_init
        self.w_init = w_init
        self.candidate_dims = candidate_dims
        self.candidate_dims_binary = [0, rank]
        self.W_left = nn.Parameter(torch.Tensor(size=(input_dim, rank)), requires_grad=True)
        self.W_right = nn.Parameter(torch.Tensor(size=(rank, output_dim)), requires_grad=True)
        self.zero_left = False
        if input_dim > output_dim:
            self.zero_left = True
        if bias:
            self.b = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()


    def finalized_arch(self, binary_choice, dim_choice):
        if binary_choice == 1:
            w_a_sampled, w_b_sampled = self.W_left[:, :dim_choice], self.W_right[:dim_choice, :]
            self.W_left = nn.Parameter(w_a_sampled)
            self.W_right = nn.Parameter(w_b_sampled)
            if dim_choice == 0:
                del self.W_left
                del self.W_right
                if self.bias:
                    del self.b
            # self.reset_parameters()
        else:
            del self.W_left
            del self.W_right
            if self.bias:
                del self.b

    def reset_parameters(self):
        if self.bias:
            self.b.data = torch.zeros_like(self.b.data)
        if self.w_init == "glorot-uniform":
            if self.zero_left:
                if self.zero_init:
                    self.W_left.data = torch.zeros_like(self.W_left.data)
                else:
                    self.W_left.data = glorot_uniform(self.W_left.data)
                self.W_right.data = glorot_uniform(self.W_right.data)
            else:
                if self.zero_init:
                    self.W_left.data = torch.zeros_like(self.W_left.data)
                else:
                    self.W_left.data = glorot_uniform(self.W_left.data)
                self.W_right.data = torch.zeros_like(self.W_right.data)
        elif self.w_init == "glorot-normal":
            if self.zero_left:
                if self.zero_init:
                    self.W_left.data = torch.zeros_like(self.W_left.data)
                else:
                # self.W_left.data = glorot_uniform(self.W_left.data)
                    self.W_right.data = glorot_normal(self.W_right.data)
            else:
                if self.zero_init:
                    self.W_right.data = torch.zeros_like(self.W_right.data)
                else:
                    self.W_right.data = glorot_normal(self.W_right.data)
                self.W_left.data = glorot_normal(self.W_left.data)

        else:
            raise ValueError

    def forward(self, x: torch.Tensor, dim_choice=None, gumbel_weights=None, dimension_mask=None, iterative_order=None, main_forward=None) -> torch.Tensor:
        # similar to lora
        self.iterative_order = iterative_order
        self.main_forward = main_forward

        if gumbel_weights is not None:
            w_a_sampled, w_b_sampled = (
                self.sample_lora(self.W_left, self.W_right, gumbel_weights=gumbel_weights,
                                 dimension_mask=dimension_mask))
            W = w_a_sampled @ w_b_sampled
        else:
            if dim_choice is not None:
                W = self.W_left[:, :dim_choice] @ self.W_right[:dim_choice, :]
            else:
                W = self.W_left @ self.W_right

        output = torch.matmul(input=x, other=W)
        if self.bias:
            if gumbel_weights is not None and gumbel_weights.shape[0] == 2:
                output += self.b * gumbel_weights[1] #  if selected at binary stage
            else:
                output += self.b
        return output

    def sample_lora(self, w_a, w_b, gumbel_weights=None, dimension_mask=None):
        stacked_samples = self.sample_weights(w_a, w_b, dimension_mask=dimension_mask)
        (sampled_w_a, sampled_w_b) = stacked_samples
        gumbel_weights = gumbel_weights.unsqueeze(-1).unsqueeze(-1)
        # print(self.iterative_order, gumbel_weights, sampled_w_a,'jj')
        w_a_sampled = torch.sum(gumbel_weights * sampled_w_a, dim=0)
        w_b_sampled = torch.sum(gumbel_weights.detach() * sampled_w_b, dim=0)

        return w_a_sampled, w_b_sampled

    def sample_weights(self, w_a, w_b, dimension_mask=None):
        (sampled_w_a, sampled_w_b) = (
            self.sample_weights_single(w_a, "a", given_max_rank_id=dimension_mask), self.sample_weights_single(w_b, "b", given_max_rank_id=dimension_mask)
        )
        return sampled_w_a.cuda(), sampled_w_b.cuda()

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
            # Set non-sampled weights to zero, different with lora
            mask_weight = torch.zeros_like(w)
            if type == "a":
                mask_weight[:, :sample_dim] = 1
            elif type == "b":
                mask_weight[:sample_dim, :] = 1
            sampled_weight = mask_weight * w
            ws.append(sampled_weight)

        return torch.stack(ws, dim=0)
