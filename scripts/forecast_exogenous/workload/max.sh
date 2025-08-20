#export CUDA_VISIBLE_DEVICES=4

des='Timexer-MS'

nohup python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/workload/ \
  --data_path X_max_1800_11_KIT-FH2-2016-1_OT.csv \
  --model_id MAX_KIT_1800 \
  --model TimeXer \
  --data custom \
  --features MS \
  --seq_len 64 \
  --pred_len 64 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --learning_rate 0.00001 \
  --des 'Timexer-MS' \
  --itr 1 \
  --train_epochs 20 \
  --patch_len 48 \
    > logs/1800_max.out &

nohup python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/workload/ \
  --data_path X_max_2700_11_KIT-FH2-2016-1_OT.csv \
  --model_id MAX_KIT_2700 \
  --model TimeXer \
  --data custom \
  --features MS \
  --seq_len 64 \
  --pred_len 64 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --learning_rate 0.00001 \
  --des 'Timexer-MS' \
  --itr 1 \
  --train_epochs 20 \
  --patch_len 36 \
  > logs/2700_max.out &

nohup python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/workload/ \
  --data_path X_max_3600_11_KIT-FH2-2016-1_OT.csv \
  --model_id MAX_KIT_3600 \
  --model TimeXer \
  --data custom \
  --features MS \
  --seq_len 64 \
  --pred_len 64 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --learning_rate 0.00001 \
  --des 'Timexer-MS' \
  --itr 1 \
  --train_epochs 20 \
  --patch_len 24 \
  > logs/3600_max.out &