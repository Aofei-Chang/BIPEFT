

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from torch.nn import Parameter

import torch
import torch.distributed as dist
from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        # print("incin", args.rank, args.world_size, args.gpu)
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)

    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm



def save_model(args, epoch, model, model_without_ddp, optimizer, arch_optimizer, loss_scaler, save_best_flag=False):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    model_without_ddp.eval()
    trainable = {}
    # trainable_names = ['fc', 'adapter', "prompt", "lora"]
    # trainable_names = ['lora_vit.fc', 'adapter', "prompt", "w_a", 'w_b']
    # trainable_names = ['adapter', "prompt", 'gate', 'final_fc', 'head', "arch"]
    for n, p in model.named_parameters():
        if "arch" in n or p.requires_grad:
            trainable[n] = p.data
    # if args.is_LoRA:
    #     lora_paras = save_lora_parameters(model)
    #     trainable = {**trainable, **lora_paras}

    # if loss_scaler is not None:
    # checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    checkpoint_paths = [output_dir / (f'checkpoint-{epoch}.pth')]
    if save_best_flag:
        checkpoint_paths.append(output_dir / ('checkpoint.pth'))
    arch_optimizer_state = None
    if arch_optimizer is not None:
        arch_optimizer_state = arch_optimizer.state_dict()

    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': trainable,
            'optimizer': optimizer.state_dict(),
            # "arch_optimizer": arch_optimizer_state,
            'epoch': epoch,
            'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
            'args': args,
        }
        if arch_optimizer_state is not None:
            to_save["arch_optimizer"] = arch_optimizer_state
        save_on_master(to_save, checkpoint_path)

def save_lora_parameters(model):
    # print(model)
    if len(model.w_As_final) > 0 and model.retrain:
        num_layer = len(model.w_As_final)  # actually, it is half
        print(f"num of wabs: {num_layer}")
        a_tensors = {f"w_a_{i:03d}": model.w_As_final[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": model.w_Bs_final[i].weight for i in range(num_layer)}
    else:
        num_layer = len(model.w_As)  # actually, it is half
        print(f"num of wabs: {num_layer}")
        a_tensors = {f"w_a_{i:03d}": model.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": model.w_Bs[i].weight for i in range(num_layer)}

    # _in = model.lora_vit.fc.in_features
    # _out = model.lora_vit.fc.out_features
    # fc_tensors = {f"fc_{_in}in_{_out}out": model.lora_vit.fc.weight}

    merged_dict = {**a_tensors, **b_tensors}
    return merged_dict

def load_lora_parameters(model, state_dict):
    if model.retrain:
        for i, w_A_linear in enumerate(model.w_As_final):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            if saved_tensor is not None:
                w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(model.w_Bs_final):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            if saved_tensor is not None:
                w_B_linear.weight = Parameter(saved_tensor)
    else:
        for i, w_A_linear in enumerate(model.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(model.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

    return model

def load_model(args, model_without_ddp, arch_optimizer=None, optimizer=None, loss_scaler=None):
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        new_state_dict = {}
        # for key, value in checkpoint['model'].items():
        #     if args.retrain_all:
        #         skip_name = "head"
        #         if skip_name in key:
        #             print(f"skip head weights!{key}")
        #             continue
        #     new_key = key.replace('module.', '')  # Remove the "module." prefix
        #     new_state_dict[new_key] = value
        for key, value in checkpoint['model'].items():
            if args.retrain_all:
                if "arch" not in key:
                    continue
            new_state_dict[key] = value
        # if args.retrain:
        #     max_indices = torch.max(new_state_dict['adapter_arch_weights'], dim=-1).indices
        model_without_ddp.load_state_dict(new_state_dict, strict=False)
        # if args.is_LoRA and not args.retrain_all:
        #     model_without_ddp = load_lora_parameters(model_without_ddp, new_state_dict)
        #     print("load lora weights!")
        # else:
        #     print("not loading lora!!")

        # model_without_ddp.last_zeta = checkpoint['zeta']
        # print(f"load zeta: {checkpoint['zeta']}")

        print("Resume checkpoint %s" % args.resume)
        if 'arch_optimizer' in checkpoint and not args.retrain:
            if arch_optimizer is not None:
                arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
                print('with arch optim')
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            if optimizer is not None and not args.retrain_all and args.load_optim:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            else:
                "not loading optim"
    return model_without_ddp, arch_optimizer, optimizer, loss_scaler

def save_prune_model(args, epoch, model, model_without_ddp, optimizer, arch_optimizer, loss_scaler, save_best_flag=False):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    model_without_ddp.eval()
    trainable = {}
    # trainable_names = ['fc', 'adapter', "prompt", "lora"]
    # trainable_names = ['lora_vit.fc', 'adapter', "prompt", "w_a", 'w_b']
    # trainable_names = ['adapter', "prompt", 'gate', 'final_fc', 'head', "arch"]
    for n, p in model.named_parameters():
        if "mask" in n or p.requires_grad:
            trainable[n] = p.data
    # if args.is_LoRA:
    #     lora_paras = save_lora_parameters(model)
    #     trainable = {**trainable, **lora_paras}

    # if loss_scaler is not None:
    # checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    checkpoint_paths = [output_dir / (f'checkpoint-{epoch}.pth')]
    if save_best_flag:
        checkpoint_paths.append(output_dir / ('checkpoint.pth'))
    arch_optimizer_state = None
    if arch_optimizer is not None:
        arch_optimizer_state = arch_optimizer.state_dict()

    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': trainable,
            'epoch': epoch,
            'args': args,
        }
        save_on_master(to_save, checkpoint_path)

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
