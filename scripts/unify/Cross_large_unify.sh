if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/unify" ]; then
    mkdir ./logs/unify
fi

if [ ! -d "./logs/unify/Cross_large" ]; then
    mkdir ./logs/unify/Cross_large
fi

model_name=TokenTime

# activate your virtual env
source /data/tingyue/anaconda3/bin/activate DD-Time

# export CUDA_VISIBLE_DEVICES="2"

export CUDA_VISIBLE_DEVICES="1,3"

root_path_name=./dataset
model_id_name=Cross_large
data_name=Cross_large

# VQVAE
token_len=16
wave_length=8
wave_stride=8
block_num=3
d_model=64
n_embed=128
VQ_type='W_SimVQ_decompose'
VQ_epoch=25
# 

# Backbone
backbone='trm' # 'trm' 'gpt2' 'qwen'
backbone_dim=128 # not for gpt2

random_seed=2024

n_classifier_layer=2

infer_step=4

mask_schedule='cosine' # 'cosine' 'linear' 'pow' 'sigmoid'

seq_len=96

elect_rate=0

for pred_len in 24 48 96 192 #  720 # 192      # 336 720
do
    python -u run_longExp.py \
    --random_seed $random_seed \
    --use_multi_gpu \
    --cross_dataset "ETTh2,ETTm2,weather,traffic,CzeLan" \
    --mask_schedule $mask_schedule \
    --is_training 1 \
    --is_unify \
    --backbone $backbone \
    --backbone_dim $backbone_dim \
    --root_path $root_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --VQ_type $VQ_type \
    --VQ_epoch $VQ_epoch \
    --elect_rate $elect_rate \
    --token_len $token_len \
    --wave_length $wave_length \
    --wave_stride $wave_stride \
    --block_num $block_num \
    --n_embed $n_embed \
    --d_model $d_model \
    --d_ff 256 \
    --dropout 0.2 \
    --des 'Exp' \
    --train_epochs 20 \
    --itr 1 \
    --batch_size 32 \
    --n_classifier_layer $n_classifier_layer \
    --infer_batch_size 64 \
    --infer_step $infer_step \
    --unify_lr 0.0001 \
    --adaptive_lr 0.0001 >logs/unify/Cross_large/$model_name'_'$backbone'_'$model_id_name'_'$seq_len'_'$pred_len'_'$VQ_type'_bl'$block_num'_wl'$wave_length'_tl'$token_len'_nl'$n_classifier_layer'_nb'$n_embed'_VQepoch'$VQ_epoch'_'$mask_schedule.log
done