import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.cuda.amp.autocast_mode import autocast

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

class Architect(object):

    def __init__(self, model, args):
        # self.network_momentum = args.momentum
        # self.network_weight_decay = args.weight_decay
        self.model = model
        self.args = args
        val_pas = []
        names = []
        # val_params = self.model.module.arch_parameters()
        for name, param in model.named_parameters():
            if "arch" in name:
                param.requires_grad = True
                val_pas.append(param)
                names.append(name)
        print("trainable arch params:", names)
        self.val_params = val_pas
        self.optimizer = torch.optim.Adam(val_pas,
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        if self.args.use_beta:
            self.anchor_arch = Dirichlet(torch.ones_like(self.model.arch_weights).cuda())
            self.anchor_arch2 = Dirichlet(torch.ones_like(self.model.arch_weights2).cuda())

    def step(self, examples, unrolled=False, epochs=100, data_iter_step=1, accum_iter=2, epoch_step=0, search_step=0):
        # self.optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss = self._backward_step(examples, epochs=epochs, epoch_step=epoch_step, search_step=search_step)
        self.optimizer.step()

        # if epochs >= self.args.prune_begin_epoch and self.model.early_stop:
        if self.model.early_stop:
            self.model.prune_step(epochs)
        return loss

    def _backward_step(self, examples, epochs, epoch_step=0, search_step=0):

        loss = self.model(examples, cur_epoch=epochs)[0]
        if self.args.arch_reg and not self.args.use_beta:
            loss_l1 = 0
            # lora_arch_weights = self.model.arch_weights
            # if self.args.iter_search:
            for arch_weight in self.model._arch_parameters:
                arch_layer_norm = F.softmax(arch_weight, dim=-1)
                loss_l1 += F.l1_loss(arch_layer_norm, arch_weight)

            loss = loss + 0.01 * loss_l1
        if self.args.use_beta:
            arch_weights = self.model.arch_weights
            arch_weights2 = self.model.arch_weights2

            cons_arch = (F.elu(arch_weights) + 1)
            cons_arch2 = (F.elu(arch_weights2) + 1)
            q_arch = Dirichlet(cons_arch)
            q_arch2 = Dirichlet(cons_arch2)
            p_arch = self.anchor_arch
            p_arch2 = self.anchor_arch2
            kl_reg = 0.01 * (torch.sum(kl_divergence(q_arch, p_arch)) + torch.sum(kl_divergence(q_arch2, p_arch2)))

            loss = loss + 0.001 * kl_reg

        loss.backward()
        # print(arch_weights2.grad)
        return loss

