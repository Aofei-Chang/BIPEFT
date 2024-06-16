#$DATASET: ""
seeds=(4 42)
prune_criterion='tra_val_cos1'
cuda_devices=(2 3)  # Define CUDA devices to use for each seed
master_ports=(11949 11101 11113 11114)  # Define master ports for each seed

budget=102600

epochs=(15 40 40 1 20 15)
prune_epochs=(2 4 4 1 3 2)
val_epochs=(1 1 1 1 1 1)
batch_sizes=(32 32 32 16 32 32)

datasets=(superglue-boolq superglue-cb superglue-multirc superglue-record superglue-rte superglue-wic)

ROOT_PATH=/data/user_name

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

for DA in "${!datasets[@]}"
do
  DATASET=${datasets[$DA]}
  epoch=${epochs[$DA]}
  val_epoch=${val_epochs[$DA]}
  prune_epoch=${prune_epochs[$DA]}
  batch_size=${batch_sizes[$DA]}
  for index in "${!seeds[@]}"
  do
      seed=${seeds[$index]}
      prune_cri=$prune_criterion
      cuda_device=${cuda_devices[$index]}
      master_port=${master_ports[$index]}
      output_dir="/data/user_name/output/NLP/moma/search_es_SuperGLUE/${DATASET}/${budget}_${seed}/"
      log_file="/data/user_name/output/NLP/moma/search_es_SuperGLUE/${budget}_${DATASET}_seed${seed}.log"
      CUDA_VISIBLE_DEVICES=$cuda_device torchrun --nproc_per_node 1  --master_port $master_port train.py \
          --model_name_or_path google-t5/t5-large \
          --task_name="${DATASET}" \
          --log_dir=/data/user_name/output/NLP/moma/search_es_SuperGLUE/logs/ \
          --train_batch_size $batch_size \
          --valid_batch_size $batch_size \
          --accum_iter 1 \
          --epochs $epoch \
          --lr 3e-4 \
          --warmup_epochs 1 \
          --val_interval $val_epoch \
          --lora_rank 8 \
          --weight_decay 0.0001 \
          --arch_learning_rate 0.01 \
          --arch_weight_decay 0.0001 \
          --output_dir "$output_dir" \
          --seed $seed \
          --use_search \
          --iter_search \
          --early_stop \
          --use_prefix \
          --use_PA \
          --use_SA \
          --budget_abs $budget \
          --prune_threshold 0.85 \
          --max_prune_step 100 \
          --prune_criterion $prune_cri \
          --prune_begin_epoch $prune_epoch \
          --split_train_data \
          &>> "$log_file" &
  done
  wait
done

epochs=(20 40 40 1 20 15)
val_epochs=(2 4 4 1 3 2)

for DA in "${!datasets[@]}"
do
  DATASET=${datasets[$DA]}
  epoch=${epochs[$DA]}
  val_epoch=${val_epochs[$DA]}
  batch_size=${batch_sizes[$DA]}
  for index in "${!seeds[@]}"
  do
      seed=${seeds[$index]}
      cuda_device=${cuda_devices[$index]}
      master_port=${master_ports[$index]}
      resume="/data/user_name/output/NLP/moma/search_es_SuperGLUE/${DATASET}/${budget}_${seed}/checkpoint.pth"
      resume_retrain="/data/user_name/output/NLP/moma/retrain_es_SuperGLUE/${DATASET}/${budget}_${seed}/checkpoint.pth"
      output_dir="/data/user_name/output/NLP/moma/retrain_es_SuperGLUE/${DATASET}/${budget}_${seed}/"
      log_file="/data/user_name/output/NLP/moma/retrain_es_SuperGLUE/${budget}_${DATASET}_seed${seed}.log"
      CUDA_VISIBLE_DEVICES=$cuda_device torchrun --nproc_per_node 1  --master_port $master_port train.py \
          --model_name_or_path google-t5/t5-large \
          --task_name="${DATASET}" \
          --log_dir=/data/user_name/output/NLP/moma/retrain_es_SuperGLUE/logs/ \
          --train_batch_size $batch_size \
          --valid_batch_size $batch_size \
          --accum_iter 1 \
          --epochs $epoch \
          --lr 3e-4 \
          --min_lr 1e-5 \
          --warmup_epochs 1 \
          --val_interval $val_epoch \
          --lora_rank 8 \
          --weight_decay 0.0001 \
          --output_dir "$output_dir" \
          --seed 4 \
          --use_search \
          --retrain \
          --iter_search \
          --use_PA \
          --use_SA \
          --use_prefix \
          --resume=$resume \
          --resume_retrain=$resume_retrain\
          --early_stop \
          &>> "$log_file" &
  done
  wait
done