#$DATASET: ""
seeds=(42)
cuda_devices=(0)  # Define CUDA devices to use for each seed
master_ports=(18449 11301 11113 11114)  # Define master ports for each seed

epochs=(15 20 3 5)
val_epochs=(2 2 1 1)

budget=102600
source_dataset='qqp'
datasets=(cola stsb qnli sst2)
ROOT_PATH=/data/user_name

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

for DA in "${!datasets[@]}"
do
  DATASET=${datasets[$DA]}
  epoch=${epochs[$DA]}
  val_epoch=${val_epochs[$DA]}
  for index in "${!seeds[@]}"
  do
      seed=${seeds[$index]}
      cuda_device=${cuda_devices[$index]}
      master_port=${master_ports[$index]}
      resume="/data/user_name/output/NLP/mom/search_es/${source_dataset}/${budget}_${seed}/checkpoint.pth"
      output_dir="/data/user_name/output/NLP/adaptation/${source_dataset}/${DATASET}/${budget}_${seed}/"
      log_file="/data/user_name/output/NLP/adaptation/${source_dataset}/${budget}_${DATASET}_seed${seed}.log"
      CUDA_VISIBLE_DEVICES=$cuda_device torchrun --nproc_per_node 1  --master_port $master_port train.py \
          --model_name_or_path google-t5/t5-large \
          --task_name="${DATASET}" \
          --log_dir=/data/user_name/output/NLP/adaptation/logs/ \
          --train_batch_size 32 \
          --valid_batch_size 32 \
          --accum_iter 1 \
          --epochs $epoch \
          --lr 3e-4 \
          --min_lr 1e-5 \
          --warmup_epochs 1 \
          --val_interval $val_epoch \
          --lora_rank 8 \
          --weight_decay 0.0001 \
          --output_dir "$output_dir" \
          --seed 8 \
          --use_search \
          --retrain \
          --retrain_all \
          --iter_search \
          --use_lora \
          --use_lnfit \
          --use_adapter \
          --use_bitfit \
          --resume=$resume \
          --early_stop \
          --zero_lr_adapter \
          &>> "$log_file" &
  done
  wait
done