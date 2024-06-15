import argparse
from cgi import test
import logging
import time
import datetime
import os
import json
from pathlib import Path

from torchvision import models
import torch.distributed as dist
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# import timm
from lora.adapter import Adapter_ViT
from base_vit import ViT
from lora.lora import LoRA_ViT, LoRA_ViT_timm
# from lorautils.dataloader_mimic import mimicDataloader
from lora.utils.result import ResultCLS, ResultMLS
from lora.utils.utils import init, save

import utils.misc as misc

from data.datasets import build_dataset

from engine import evaluate

weightInfo={
            # "small":"WinKawaks/vit-small-patch16-224",
            "base(384)":"hf_hub:timm/vit_base_patch16_384.orig_in21k_ft_in1k",
            "base":"vit_base_patch16_224.orig_in21k_ft_in1k",
            "base_dino":"vit_base_patch16_224.dino", # 21k -> 1k
            "base_sam":"vit_base_patch16_224.sam", # 1k
            "base_mill":"vit_base_patch16_224_miil.in21k_ft_in1k", # 1k
            "base_beit":"beitv2_base_patch16_224.in1k_ft_in22k_in1k",
            "base_clip":"vit_base_patch16_clip_224.laion2b_ft_in1k", # 1k
            "base_deit":"deit_base_distilled_patch16_224", # 1k
            # "large":"google/vit-large-patch16-224",
            "large_clip":"vit_large_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
            "large_beit":"beitv2_large_patch16_224.in1k_ft_in22k_in1k",
            "huge_clip":"vit_huge_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
            "giant_eva":"eva_giant_patch14_224.clip_ft_in1k", # laion-> 1k
            "giant_clip":"vit_giant_patch14_clip_224.laion2b",
            "giga_clip":"vit_gigantic_patch14_clip_224.laion2b"
            }

def get_args_parser():
    parser = argparse.ArgumentParser('lora search training', add_help=False)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str, default='/data/')
    parser.add_argument('--data_set', default='IMNET', type=str, help='Image Net dataset path')
    parser.add_argument("--vit_path", type=str, default='/data/aofei/vit/ViT-B_16.npz')
    parser.add_argument("--log_dir", type=str, default='/data/aofei/vit')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                 "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--inception',action='store_true')
    parser.add_argument('--direct_resize',action='store_true')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--clip_grad', type=float, default=None, metavar='clip gradient',
                        help='clips gradient norm of an iterable of parameters')

    # parser.add_argument("-data_path",type=str, default='../data/NIH_X-ray/')
    parser.add_argument("--data_info", type=str, default='nih_split_712.json')
    parser.add_argument("--annotation", type=str, default='Data_Entry_2017_jpg.csv')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument("--num_classes", "-nc", type=int, default=12)
    parser.add_argument("--backbone", type=str, default='base(384)')
    parser.add_argument("--train_type", "-tt", type=str, default="lora",
                        help="lora: only train lora, full: finetune on all, linear: finetune only on linear layer")

    parser.add_argument("--lora_rank", "-r", type=int, default=8)

    parser.add_argument('--is_adapter', action='store_true')
    parser.add_argument('--is_LoRA', action='store_true')
    parser.add_argument('--is_prompt', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # AutoFormer config
    parser.add_argument('--mode', type=str, default='super', choices=['super', 'vp', 'retrain', 'search'],
                        help='mode of AutoFormer')
    # search
    parser.add_argument('--use_search', action='store_true', help='whether use NAS for Adapter')
    parser.add_argument('--use_visual_search', action='store_true', help='whether use NAS for visual Adapter')
    parser.add_argument('--arch_reg', type=bool, default=True, help='whether to apply architecture regulation')
    parser.add_argument('--arch_learning_rate', type=float, default=1e-3, metavar='LR',
                        help='arch learning rate')
    parser.add_argument('--min_arch_lr', type=float, default=1e-5, metavar='LR',
                        help='arch learning rate')
    parser.add_argument('--arch_weight_decay', type=float, default=0.01,
                        help='arch weight decay (default: 0.01)')

    parser.add_argument('--retrain', action='store_true',
                        help='whether to use retrain mode with freezing the search parameters')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')

    return parser

def extractBackbone(state_dict,prefix: str)->callable:
    if prefix==None:
        for k in list(state_dict.keys()):
            if k.startswith('fc'):
                del state_dict[k]
        return state_dict

    for k in list(state_dict.keys()):
        if k.startswith(f'{prefix}.'):
            # print(k)
            if k.startswith('') and not k.startswith(f'{prefix}.fc'):
                # remove prefix
                state_dict[k[len(f"{prefix}."):]] = state_dict[k]
        # del掉不是backbone的部分
        del state_dict[k]
    return state_dict


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    ckpt_path = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(args)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # scaler = GradScaler()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    model = ViT('B_16_imagenet1k', pretrained=True, image_size=224, weight_path=args.vit_path)
    # model.load_state_dict(torch.load('../preTrain/B_16_imagenet1k.pth'))
    # model.load_state_dict(torch.load(args.vit_path))
    dataset_val, num_classes = build_dataset(is_train=False, args=args, is_individual_prompt=(
            args.is_adapter or args.is_LoRA or args.is_prefix))
    args.num_classes = num_classes
    print(dataset_val, args.num_classes)

    if args.train_type == "lora":
        # lora_model = LoRA_ViT_timm(model, r=args.rank, num_classes=args.num_classes)
        lora_model = LoRA_ViT(model, r=args.lora_rank, num_classes=args.num_classes)
        # weight = torch.load('./results/cxp_2.pt')
        # extractBackbone(weight, 'module')
        # lora_model.load_state_dict(weight)
        num_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        logging.info(f"trainable parameters: {num_params / 2 ** 20:.4f}M")
        model = lora_model.to(device)
    elif args.train_type == 'adapter':
        adapter_model = Adapter_ViT(model, num_classes=args.num_classes)
        num_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
        logging.info(f"trainable parameters: {num_params / 2 ** 20:.4f}M")
        model = adapter_model.to(device)
    elif args.train_type == "full":
        model.fc = nn.Linear(768, args.num_classes)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"trainable parameters: {num_params / 2 ** 20:.4f}M")
        net = model.to(device)
    elif args.train_type == "linear":
        model.fc = nn.Linear(768, args.num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        num_params = sum(p.numel() for p in model.fc.parameters())
        logging.info(f"trainable parameters: {num_params / 2 ** 20:.4f}M")
        net = model.to(device)
    else:
        logging.info("Wrong training type")
        exit()

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_val = %s" % str(sampler_val))

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    ckpts = misc.load_model(args=args, model_without_ddp=model)
    model = ckpts[0]

    model.to(device)

    # print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    accuracy = 0.0
    test_stats = evaluate(data_loader_val, model, device, amp=args.amp)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    accuracy = test_stats["acc1"]
    print(f'Accuracy: {accuracy:.2f}%')
    if args.output_dir:
        with (args.output_dir + "eval_log.txt").open("a") as f:
            f.write(json.dumps(test_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Eval time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


