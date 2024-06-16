import torch
from torch.cuda.amp.autocast_mode import autocast
from tqdm import tqdm
import time

import utils.misc as misc
import utils.lr_sched as lr_sched

from lora.lora import weights


def train_one_epoch(model, epoch, train_loader, eval_loader, optimizer, scaler,
                    architect, test_loader=None, args=None, log_writer=None, scheduler=None, early_stop_flag=False):
    retrain_mode = args.retrain
    use_search = args.use_search

    loss_scaler = scaler

    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # architect = None
    search_optimizer = None
    if use_search and not retrain_mode:
        if architect is not None:
            search_optimizer = architect.optimizer
            metric_logger.add_meter('search_lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    ite = 0
    search_step = 1

    # val_iter = iter(eval_loader)
    print(len(eval_loader), "evals")
    val_data_list = []
    if use_search and not retrain_mode:
        val_data_list = [i for i in eval_loader]
    r = 0
    for data_iter_step, inputs in enumerate(
            metric_logger.log_every(train_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if scheduler is None:
            if data_iter_step % accum_iter == 0:
                if use_search and not retrain_mode:
                    lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)
                else:
                    lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)

        loss_search = None

        if use_search and not retrain_mode:
            trn_input, val_input = inputs, val_data_list[r%len(val_data_list)]
            r += 1

            val_input['decoder_input_ids'] = model.t5_model._shift_right(
                val_input['labels'])
            for k, v in val_input.items():
                val_input[k] = v.to(model.t5_model.device)
            loss_search = architect.step(val_input,
                                         unrolled=False, epochs=epoch, data_iter_step=data_iter_step,
                                         accum_iter=accum_iter, epoch_step=data_iter_step, search_step=search_step)
        else:
            trn_input, val_input = inputs, None
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        trn_input['decoder_input_ids'] = model.t5_model._shift_right(trn_input['labels'])
        for k, v in trn_input.items():
            trn_input[k] = v.to(model.t5_model.device)
        outputs = model(x=trn_input, cur_epoch=epoch, main_forward=True)

        if args.use_search:
            c_loss = outputs[0]
        else:
            c_loss = outputs.loss
        c_loss.requires_grad_(True)

        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        ite += 1

        if use_search and not retrain_mode:
            if loss_search is not None:
                search_loss_value = loss_search.item()
            else:
                search_loss_value = 0.00001

        if torch.isnan(loss):
            print("NaN loss encountered. Skipping this batch.")
            continue

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=weights(model),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=args.clip_grad_norm)
        optimizer.step()

        if model.early_stop:
            model.prune_step(epoch) # here we accumulate the sensitivity and calculate the trigger at every step

        if scheduler is not None:
            scheduler.step()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        if use_search and not retrain_mode:
            if data_iter_step % search_step == 0:
                metric_logger.update(search_loss=search_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if use_search and not retrain_mode:
            search_lr = search_optimizer.param_groups[0]["lr"]
            metric_logger.update(search_lr=search_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if use_search and not retrain_mode:
            search_loss_value_reduce = misc.all_reduce_mean(c_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(train_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            if use_search and not retrain_mode:
                log_writer.add_scalar('search_train_loss', search_loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('search_lr', search_lr, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        #early-stop:
        if model.early_stop and model.max_prune_step==0:
            early_stop_flag = True
        if early_stop_flag:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, scheduler, early_stop_flag


@torch.no_grad()
def evaluate(model, tokenizer, dataloader, compute_metrics, data_info, args=None):

    info = data_info
    model.eval()
    loss_list = []
    outputs = []
    labels = []

    for inputs in tqdm(dataloader):
        loss, generated_tokens, label = prediction_step(model.t5_model, tokenizer, inputs, args=args)
        loss_list.append(loss.item())
        outputs.append(generated_tokens.cpu())
        labels.append(label.cpu())

    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    # print(outputs,labels,info)
    result = compute_metrics((outputs, labels, info))
    print(f'metrics: {result}')
    return result



def prediction_step(
    model,
    tokenizer,
    inputs,
    prediction_loss_only: bool = False,
    args=None
):

    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
    has_labels = "labels" in inputs
    # inputs = self._prepare_inputs(inputs)

    gen_kwargs = {
        "max_length": inputs["labels"].shape[-1]+10 if args.task_name=='web_nlg' else model.config.max_length,
        "num_beams": 5 if args.task_name=='web_nlg' else model.config.num_beams,
    }
    all_max_length = 192 if args.task_name=='web_nlg' else model.config.max_length

    generated_tokens = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **gen_kwargs,
    ).cpu()
    # in case the batch is shorter than max length, the output should be padded
    if generated_tokens.shape[-1] < all_max_length and not args.test_module:
        generated_tokens = _pad_tensors_to_max_len(model, tokenizer, generated_tokens, all_max_length)

    loss = torch.Tensor([0])

    if prediction_loss_only:
        return (loss, None, None)

    labels = inputs["labels"].cpu()
    if labels.shape[-1] < all_max_length and not args.test_module:
        labels = _pad_tensors_to_max_len(model, tokenizer, labels, all_max_length)

    if args.task_name in ["superglue-record"] and labels.shape[-1] > all_max_length:
        labels = labels[...,:all_max_length]

    return (loss, generated_tokens, labels)

def _pad_tensors_to_max_len(model, tokenizer, tensor, max_length):
    if tokenizer is not None and hasattr(tokenizer, "pad_token_id"):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    else:
        if model.config.pad_token_id is not None:
            pad_token_id = model.config.pad_token_id
        else:
            raise ValueError(
                "Pad_token_id must be set in the configuration of the model, in order to pad tensors")

    padded_tensor = pad_token_id * torch.ones(
        (tensor.shape[0],
         max_length), dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor