#export CUDA_VISIBLE_DEVICES=1

des='Timexer-MS'

# nohup python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/workload/ \
#   --data_path X_mean_1800_11_KIT-FH2-2016-1_OT.csv \
#   --model_id MEAN_KIT_1800 \
#   --model TimeXer \
#   --data custom \
#   --features MS \
#   --seq_len 64 \
#   --pred_len 64 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 11 \
#   --dec_in 11 \
#   --c_out 11 \
#   --learning_rate 0.00001 \
#   --des 'Timexer-MS' \
#   --itr 1 \
#   --train_epochs 20 \
#   --patch_len 48 \
#   --gpu 4 \
#   > logs/1800_mean.out &


# nohup python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/workload/ \
#   --data_path X_mean_1800_11_KIT-FH2-2016-1_OT.csv \
#   --model_id OLD1800 \
#   --model TimeXer \
#   --data custom \
#   --features MS \
#   --seq_len 72 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 11 \
#   --dec_in 11 \
#   --c_out 11 \
#   --learning_rate 0.00001 \
#   --des 'Timexer-MS' \
#   --itr 1 \
#   --train_epochs 10 \
#   --patch_len 12 \
#   --use_multi_gpu \
#   --devices 0,1 \
#   > logs/OLD1800.out &


# nohup python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/workload/ \
#   --data_path 0625_1548_8_CEA-Curie-2011-2.1-cln.csv \
#   --model_id 1-0.25_CEA \
#   --model TimeXer \
#   --data custom \
#   --features MS \
#   --seq_len 96 \
#   --patch_len 24 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --learning_rate 0.00001 \
#   --des 'Timexer-MS' \
#   --itr 1 \
#   --freq 't' \
#   --train_epochs 10 \
#   --target 'OT'\
#   > logs/1-0.25_CEA.out &


nohup python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/workload/ \
  --data_path 0625_1548_8_CEA-Curie-2011-2.1-cln.csv \
  --model_id 0820_1-0.25-52_CEA \
  --model TimeXer \
  --data custom \
  --features MS \
  --seq_len 96 \
  --patch_len 24 \
  --pred_len 24 \
  --e_layers 5 \
  --d_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --learning_rate 0.0001 \
  --des 'Timexer-MS' \
  --itr 1 \
  --freq 't' \
  --loss  'MAE' \
  --train_epochs 10 \
  --target 'OT'\
  > logs/0820_1-0.25-52_CEA.out &