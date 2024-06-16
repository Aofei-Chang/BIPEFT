#$DATASET: ""
seeds=(42)
cuda_devices=(0)   # Define CUDA devices to use for each seed
master_ports=(17126 17121 17113 17114)  # Define master ports for each seed

epochs=(15 20 20)
val_epochs=(2 2 2)
batch_sizes=(64 64 32)

datasets=(cola mrpc superglue-rte)

ROOT_PATH=/data/user_name

export TRANSFORMERS_CACHE=${ROOT_PATH}/huggingface_cache/transformers
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

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
      output_dir="/data/user_name/output/NLP/ablation/no_gumbel/search/${DATASET}/${seed}/"
      log_file="/data/user_name/output/NLP/ablation/no_gumbel/search/${DATASET}_seed${seed}.log"
      CUDA_VISIBLE_DEVICES=$cuda_device torchrun --nproc_per_node 1  --master_port $master_port train.py \
          --model_name_or_path google-t5/t5-large \
          --task_name="${DATASET}" \
          --log_dir=/data/user_name/output/NLP/ablation/no_gumbel/search/logs/ \
          --train_batch_size $batch_size \
          --valid_batch_size $batch_size \
          --accum_iter 1 \
          --epochs $epoch \
          --lr 3e-4 \
          --min_lr 1e-4 \
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
          --use_lora \
          --use_lnfit \
          --use_adapter \
          --use_bitfit \
          --split_train_data \
          --no_gumbel \
          &>> "$log_file" &
  done
  wait
done


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
      resume="/data/user_name/output/NLP/ablation/no_gumbel/search/${DATASET}/${seed}/checkpoint.pth"
      output_dir="/data/user_name/output/NLP/ablation/no_gumbel/retrain/${DATASET}/${seed}/"
      log_file="/data/user_name/output/NLP/ablation/no_gumbel/retrain/${DATASET}_seed${seed}.log"
      CUDA_VISIBLE_DEVICES=$cuda_device torchrun --nproc_per_node 1  --master_port $master_port train.py \
          --model_name_or_path google-t5/t5-large \
          --task_name="${DATASET}" \
          --log_dir=/data/user_name/output/NLP/ablation/no_gumbel/retrain/logs/ \
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
          --seed $seed \
          --use_search \
          --retrain \
          --retrain_all \
          --iter_search \
          --use_lora \
          --use_lnfit \
          --use_adapter \
          --use_bitfit \
          --resume=$resume \
          &>> "$log_file" &
  done
  wait
done