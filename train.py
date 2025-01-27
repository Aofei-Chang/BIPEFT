import argparse
import logging
import time
import datetime
import os
import json
import random
import functools
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from examples_seq2seq.data_processors import AutoPostProcessor, AutoTask, TaskDataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import AutoTokenizer, set_seed
from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup

from space.t5_search_space import MoM_T5, weights

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from engine import train_one_epoch, evaluate
from architect import Architect

TASK_TO_METRICS = {"mrpc": ["accuracy", "f1"],
                   "cola": ['matthews_correlation'],
                   "stsb": ['pearson', 'spearmanr'],
                   'sst2': ['accuracy'],
                   "mnli": ["accuracy"],
                   "mnli_mismatched": ["accuracy"],
                   "mnli_matched": ["accuracy"],
                   "qnli": ["accuracy"],
                   "rte": ["accuracy"],
                   "wnli": ["accuracy"],
                   "qqp": ["accuracy", "f1"],
                   "superglue-boolq": ["accuracy"],
                   "superglue-rte": ["accuracy"],
                   "superglue-cb": ["f1_multiclass", "accuracy"],
                   "superglue-copa": ["accuracy"],
                   "superglue-multirc": ["f1", "em"],
                   "superglue-wic": ["accuracy"],
                   "superglue-wsc.fixed": ["accuracy"],
                   "superglue-record": ["f1", "em"]
                   }
addable_delta_type = ['None', 'Adapter', 'BitFit',
                      'Compacter', 'Lora', 'LowRankAdapter', 'Prefix']

def get_args_parser():
    parser = argparse.ArgumentParser('lora search training', add_help=False)

    parser.add_argument("--model_name_or_path", type=str, default="t5-large", help="")
    parser.add_argument("--task_name", type=str, help="")
    # parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument("--data_path", type=str, default='/data/')
    parser.add_argument('--data_set', default='IMNET', type=str, help='Image Net dataset path')
    parser.add_argument("--log_dir", type=str, default='/data/user_name')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay (default: 0.02)')
    parser.add_argument('--clip_grad_norm', type=float, default=2.0, help='')


    parser.add_argument("--lora_rank", "-r", type=int, default=8)

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_retrain', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--warmup_epochs', type=float, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--val_interval', default=10, type=int, help='validataion interval')
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

    # search
    parser.add_argument('--use_search', action='store_true', help='whether use NAS')
    parser.add_argument('--arch_reg', type=bool, default=True, help='whether to apply architecture regulation')
    parser.add_argument('--arch_learning_rate', type=float, default=1e-3, metavar='LR',
                        help='arch learning rate')
    parser.add_argument('--prompt_learning_rate', type=float, default=1e-2, metavar='LR',
                        help='prompt learning rate')
    parser.add_argument('--min_arch_lr', type=float, default=1e-5, metavar='LR',
                        help='arch learning rate')
    parser.add_argument('--arch_weight_decay', type=float, default=0.01,
                        help='arch weight decay (default: 0.01)')

    parser.add_argument('--retrain', action='store_true',
                        help='whether to use retrain mode with freezing the search parameters')
    parser.add_argument('--retrain_all', action='store_true',
                        help='whether to use retrain mode and re-initialze the PEFT parameters')

    parser.add_argument('--use_budget', action='store_true', help='whether to use params budget')
    parser.add_argument('--budget_abs', default=150000, type=int, help='whether to use params budget')
    parser.add_argument('--early_stop', action='store_true', help='whether to use early stopping')
    parser.add_argument("--max_prune_steps", type=int, default=100)
    parser.add_argument("--prune_begin_epoch", type=int, default=0)
    parser.add_argument('--prune_criterion', type=str, default='tra_val_cos1', choices=['tra', 'val', 'tra_val', 'tra_val_cos1', 'tra_val_cos2'],
                        help='criterion for pruning')
    parser.add_argument('--prune_threshold', type=float, default=0.85,
                        help='stability-based pruning threshold (default: 0.85)')
    parser.add_argument('--no_abs_grad', action='store_true', help='the other choice for gradient')
    parser.add_argument('--split_train_data', action='store_true', help='whether to split training data')

    parser.add_argument('--iter_search', action='store_true', help='whether to use iterative search')
    parser.add_argument('--use_beta', action='store_true', help='whether to use beta distribution')
    parser.add_argument('--use_lora', action='store_true', help='whether using lora')
    parser.add_argument('--use_bitfit', action='store_true', help='whether using bitfit')
    parser.add_argument('--use_lnfit', action='store_true', help='whether using lnfit')
    parser.add_argument('--use_adapter', action='store_true', help='whether using adapter')
    parser.add_argument('--use_prefix', action='store_true', help='whether using prefix tuning')
    parser.add_argument('--use_SA', action='store_true', help='whether using serial adapter')
    parser.add_argument('--use_PA', action='store_true', help='whether using parallel adapter')

    parser.add_argument('--large_sp', action='store_true', help='whether using larger search space')
    parser.add_argument('--small_sp', action='store_true', help='whether using smaller search space')
    parser.add_argument('--small_prefix', action='store_true', help='whether using smaller prefix search space')
    parser.add_argument('--zero_lr_adapter', action='store_true', help='whether using zero-initialization for low-rank adapter')
    parser.add_argument('--fix_prefix_dim', action='store_true', help='whether to fix prefix dim, because the parameters of prefix is decided by the MLP(locations), not the length of prefix')

    # ablation
    parser.add_argument('--binary_then_dim', action='store_true', help="ablation study: binary search then dimension search")
    parser.add_argument('--dim_then_binary', action='store_true')
    parser.add_argument('--no_gumbel', action='store_true', help="not using gumbel-softmax for ablation study")

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.add_argument('--test_module', action='store_true')

    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(args)

    # Set seed before initializing model.
    set_seed(args.seed)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # config, tokenizer, model
    config = T5Config.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    print("load pretrained T5 model success!")
    # scaler = GradScaler()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    args.eval_super = False

    if 'superglue' in args.task_name:
        if 'superglue-record' == args.task_name:
            args.max_source_length = 512
        else:
            args.max_source_length = 256
    else:
        if "rte" == args.task_name:
            args.max_source_length = 256
        elif "web_nlg" == args.task_name:
            args.max_source_length = 300
        else:
            args.max_source_length = 128
    print(f"max_source_length: {args.max_source_length}")

    # here, do the modification to the model for PEFT
    # peft_model = LoRA_T5(model, r=args.lora_rank, model_config=config, args=args)
    peft_model = MoM_T5(model, r=args.lora_rank, model_config=config, args=args)
    model_without_ddp = peft_model
    num_params = 0
    for (n, p) in model.named_parameters():
        if p.requires_grad:
            # print(n)
            num_params += p.numel()
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_num_params = sum(p.numel() for p in model.parameters())
    print(f"all params: {all_num_params}, trainable params: {num_params}")

    # Data collator
    data_collator = TaskDataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # function for preprocessing the dataset
    def preprocess_function(examples, max_target_length):
        model_inputs = tokenizer(examples['source'], max_length=args.max_source_length,
                                 padding=False, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'], max_length=max_target_length, padding=False, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["extra_fields"] = examples['extra_fields']
        return model_inputs

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}

    # training dataset
    train_datasets = AutoTask.get(args.task_name, ["en"], seed=args.seed).get(
        split="train",
        split_validation_test=True,
        add_prefix=True,
        n_obs=None)
    max_target_length = AutoTask.get(args.task_name, ["en"]).get_max_target_length(
        tokenizer=tokenizer, default_max_length=128)

    train_dataset = train_datasets.map(
        functools.partial(preprocess_function, max_target_length=max_target_length),
        batched=True,
        remove_columns=column_names,
    )
    print(f"train dataset: {len(train_dataset)}")

    # eval dataset
    eval_dataset = AutoTask.get(args.task_name, ["en"], seed=args.seed).get(
        split="validation",
        split_validation_test=True,
        add_prefix=True)

    max_target_length = AutoTask.get(args.task_name, ["en"]).get_max_target_length(
        tokenizer=tokenizer, default_max_length=128)

    eval_dataset = eval_dataset.map(
        functools.partial(preprocess_function, max_target_length=max_target_length),
        batched=True,
        remove_columns=column_names
    )
    print(f"eval dataset: {len(eval_dataset)}\n")

    # test dataset
    test_dataset = AutoTask.get(args.task_name, ["en"], seed=args.seed).get(
        split="test",
        split_validation_test=True,
        add_prefix=True)

    max_target_length = AutoTask.get(args.task_name, ["en"]).get_max_target_length(
        tokenizer=tokenizer, default_max_length=128)

    test_dataset = test_dataset.map(
        functools.partial(preprocess_function, max_target_length=max_target_length),
        batched=True,
        remove_columns=column_names
    )
    print(f"test dataset: {len(test_dataset)}\n")

    data_info = {"eval": eval_dataset['extra_fields'],
                 "test": test_dataset['extra_fields']}
    # split the dataset
    # if not args.for_baseline and args.using_darts:
    if not args.retrain and args.use_search:
        train_dataset_ = train_dataset.remove_columns(['task', 'extra_fields'])
        train_dataset_length = len(train_dataset_)

        if train_dataset_length % 2 == 0:
            train_dataset_train, train_dataset_eval = random_split(train_dataset_, [train_dataset_length // 2,
                                                                                    train_dataset_length // 2])
        else:
            train_dataset_train, train_dataset_eval, _ = random_split(train_dataset_, [train_dataset_length // 2,
                                                                                       train_dataset_length // 2, 1])
        assert len(train_dataset_train) == len(train_dataset_eval)
        assert len(train_dataset_train) + len(train_dataset_eval) >= len(train_dataset_) - 1

        eval_dataset_ = eval_dataset.remove_columns(['task', 'extra_fields'])
        test_dataset_ = test_dataset.remove_columns(['task', 'extra_fields'])
    else:
        train_dataset_train = train_dataset.remove_columns(['task', 'extra_fields'])
        train_dataset_eval = eval_dataset.remove_columns(['task', 'extra_fields'])
        eval_dataset_ = eval_dataset.remove_columns(['task', 'extra_fields'])
        test_dataset_ = test_dataset.remove_columns(['task', 'extra_fields'])

    # dataloader
    train_dataloader = DataLoader(train_dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=data_collator)
    eval_dataloader = DataLoader(train_dataset_eval, batch_size=args.train_batch_size, shuffle=True,
                                 collate_fn=data_collator)
    eval_dataloader_not_shuffle = DataLoader(eval_dataset_, batch_size=args.valid_batch_size, shuffle=False,
                                             collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset_, batch_size=args.valid_batch_size, shuffle=False,
                                 collate_fn=data_collator)
    print(
        f"train: {len(train_dataloader)} eval: {len(eval_dataloader)} eval_no_shufle: {len(eval_dataloader_not_shuffle)} test: {len(test_dataloader)}")

    # Metric
    eval_metrics = AutoTask.get(args.task_name, ["en"]).metric

    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        if args.test_module:
            print("fwef", preds, labels, preds.shape, labels.shape)
        post_processor = AutoPostProcessor.get(args.task_name, tokenizer, True)
        decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    optimizer = optim.AdamW(weights(model_without_ddp), lr=args.lr, weight_decay=args.weight_decay)

    max_step = args.epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max_step)

    scaler = NativeScaler()
    model, arch_optimizer, optimizer_loaded, loss_scaler_loaded = misc.load_model(args=args, model_without_ddp=model_without_ddp, arch_optimizer=None, optimizer=optimizer, loss_scaler=scaler)
    if not args.retrain_all:
        optimizer = optimizer_loaded
        loss_scaler = loss_scaler_loaded
    else:
        loss_scaler = scaler
    print("model weight optimizer: ", optimizer)
    model.to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_epoch = 0
    architect = None
    if args.use_search and not args.retrain:
        architect = Architect(model, args=args)
    print("architect: ", architect)

    #before retraining or search
    if args.retrain:
        model.finalize_arch()
        num_params = 0
        for (n, p) in model.named_parameters():
            if 'arch' in n:
                p.requires_grad = False
            if p.requires_grad:
                # print(p.numel(), n)
                num_params += p.numel()
        optimizer = optim.AdamW(weights(model), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max_step)
        all_num_params = sum(p.numel() for p in model.parameters())
        print(f"all params: {all_num_params}, trainable params: {num_params}")

    if args.test_module:
        args.start_epoch = args.epochs

    for epoch in range(args.start_epoch, args.epochs):
        train_stats, ty, early_stop_flag = train_one_epoch(model, epoch, train_loader=train_dataloader, eval_loader=eval_dataloader,
                            scaler=scaler, test_loader=test_dataloader, args=args, architect=architect,
                            optimizer=optimizer, log_writer=log_writer, scheduler=scheduler)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if (epoch % args.val_interval == 0) or epoch == args.epochs - 1:
            #change the dataloader for different testing
            if args.use_search and not args.retrain:
                model.finalize_arch()
                print("finalize arch for search eval.")
            test_stats = evaluate(model, tokenizer=tokenizer, dataloader=eval_dataloader_not_shuffle, compute_metrics=compute_metrics, data_info=data_info['eval'], args=args)
            length_test = len(eval_dataset)
            print(f"Evaluation of the network on {length_test} validation data: {test_stats.items()}")
            save_flag = True
            save_best_flag = False
            if not args.retrain and args.early_stop:
                save_best_flag = True
            if max_accuracy <= list(test_stats.items())[0][1]:
                save_flag = True
                save_best_flag = True
                if max_accuracy <= list(test_stats.items())[0][1]:
                    best_epoch = epoch
            max_accuracy = max(max_accuracy, list(test_stats.items())[0][1])
            print(f'Max accuracy: {best_epoch} {max_accuracy:.2f}%')


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': f"{num_params / 2 ** 20:.4f}M"}

            if args.output_dir:
                with open(args.output_dir + "log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            if args.output_dir and save_flag:
                print(f"Save epoch {epoch} as checkpoint!")
                arch_optimizer = None

                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, arch_optimizer=arch_optimizer,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, save_best_flag=save_best_flag)

        if early_stop_flag:
            print(f"Early stop at epoch {epoch}")
            break

    if args.retrain or not args.use_search:
        test_stats = evaluate(model, tokenizer=tokenizer, dataloader=test_dataloader, compute_metrics=compute_metrics, data_info=data_info['test'], args=args)
        length_test = len(test_dataset)
    else:
        test_stats = evaluate(model, tokenizer=tokenizer, dataloader=test_dataloader, compute_metrics=compute_metrics, data_info=data_info['test'], args=args)
        length_test = len(test_dataset)

    print(f"Evaluation of the network on {length_test} test data: {test_stats.items()}")


    for best_epoch in [best_epoch]:
        # if best_epoch == args.epochs - 1:
        #     break
        print(f"loading best epoch {best_epoch} for final testing !")
        args.resume = args.output_dir + f"checkpoint-{best_epoch}.pth"
        args.retrain_all = False
        model, arch_optimizer, optimizer, loss_scaler = misc.load_model(args=args, model_without_ddp=model,
                                                                        arch_optimizer=None, optimizer=optimizer,
                                                                        loss_scaler=scaler)
        model.to(device)
        # if args.retrain or not args.use_search:
        #     test_stats = evaluate(model, dataloader=test_dataloader, compute_metrics=compute_metrics)
        #     length_test = len(test_dataset)
        # else:
        # test_stats = evaluate(model, dataloader=eval_dataloader, compute_metrics=compute_metrics)
        test_stats = evaluate(model, tokenizer=tokenizer, dataloader=eval_dataloader_not_shuffle, compute_metrics=compute_metrics, data_info=data_info['eval'],
                              args=args)
        length_test = len(eval_dataset)
        print(f"Evaluation of the network on the {length_test} val images: {test_stats}")
        test_stats = evaluate(model, tokenizer=tokenizer, dataloader=test_dataloader, compute_metrics=compute_metrics, data_info=data_info['test'],
                              args=args)
        length_test = len(test_dataset)

        print(f"Evaluation of the network on the {length_test} test images: {test_stats}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


