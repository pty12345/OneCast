if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Tokenizer" ]; then
    mkdir ./logs/Tokenizer
fi

if [ ! -d "./logs/Tokenizer/Cross_CzeLan" ]; then
    mkdir ./logs/Tokenizer/Cross_CzeLan
fi

export CUDA_VISIBLE_DEVICES="5"

source /data/tingyue/anaconda3/bin/activate DD-Time

root_path_name=../dataset
model_id_name=Cross_CzeLan
data_name=TFB

n_embed=128
wave_length=8
wave_stride=8
token_len=16
vq_model='W_SimVQ_decompose' # VanillaVQ, SimVQ, SimVQ_CNN, SimVQ_CNN_double_token
d_model=64
block_num=3

seq_len=96
pred_len=96

num_epoch=25
lr_decay_rate=0.99

for pred_len in 24 48 96 192; do
    python -u main.py \
    --is_training 1 \
    --cross_dataset "CzeLan" \
    --seq_len $seq_len \
    --label_len 0 \
    --model_id $model_id_name \
    --pred_len $pred_len \
    --lr_decay_rate $lr_decay_rate \
    --vq_model $vq_model \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --root_path $root_path_name \
    --data $data_name \
    --wave_length $wave_length \
    --wave_stride $wave_stride \
    --features M \
    --token_len $token_len \
    --n_embed $n_embed \
    --eval_per_epoch \
    --d_model $d_model \
    --block_num $block_num \
    --dropout 0.2 \
    --num_epoch $num_epoch \
    --lr 0.0005 >logs/Tokenizer/Cross_CzeLan/$model_id_name'_'$token_len'_sl'$seq_len'_pl'$pred_len'_emb'$n_embed'_wl'$wave_length'_bl'$block_num'_dm'$d_model'_'$vq_model'_epoch'$num_epoch.log
done
