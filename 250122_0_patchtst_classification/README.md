# README

```bash
CUDA_VISIBLE_DEVICES=3,5,6,7 \
    torchrun --nproc_per_node 4 run_image_classification.py \
    --tokenizer_name ./pretrained_models/MedBERT \
    --text_model_name ./pretrained_models/MedBERT \
    --trust_remote_code \
    --freeze_text_model \
    \
    --dataset_path ./dataset_loading_scripts/abide.py \
    --data_dir /bigdata/yanting/datasets/nilearn_data \
    \
    --output_dir ./outputs --overwrite_output_dir \
    --do_train  --do_eval \
    --eval_strategy epoch \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 --weight_decay 1e-4 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine_with_restarts \
    --logging_steps 1 \
    --save_strategy best \
    --save_total_limit 1 \
    --save_safetensors False \
    --dataloader_drop_last True \
    --dataloader_num_workers 8 \
    --metric_for_best_model loss \
    --run_name patchtst_test \
    --label_names target_values \
    --remove_unused_columns False \
    --report_to wandb
```
