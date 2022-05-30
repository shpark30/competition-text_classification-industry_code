#! /bin/bash

MODEL='ensemble'
N_KOBERT=1
N_KOGPT2=1
N_MLBERT=0
LR=2e-4
SCHEDULER='cosine'
DECAY=0.01
NUM_SAMPLES=550000
SAMPLE_DIST='same'
UPSAMPLE=''
EPOCHS=100
DEVICE='cuda'
PATIENCE=5
KOBERT_LAYERS=2
KOGPT2_LAYERS=2
KOBERT_BN=' --kobert-bn'
KOGPT2_BN=' --kogpt2-bn'
KOBERT_LN='' # ' --kobert-ln'
KOGPT2_LN='' # ' --kogpt2-ln'

eval 'python bagging.py --model=${MODEL} --weight-decay=${DECAY} --lr-scheduler=${SCHEDULER} --learning-rate=${LR} --upsample=${UPSAMPLE} --n-kobert=${N_KOBERT} --n-kogpt2=${N_KOGPT2} --n-mlbert=${N_MLBERT} --kobert-layers=${KOBERT_LAYERS} --kogpt2-layers=${KOGPT2_LAYERS} --num-samples=${NUM_SAMPLES} --sample-dist=${SAMPLE_DIST} --epochs=${EPOCHS} --patience=${PATIENCE} --device=${DEVICE}${KOBERT_BN}${KOGPT2_BN}${KOBERT_LN}${KOGPT2_LN}'

# N_KOBERT=5
# N_KOGPT2=5
# N_MLBERT=0
# NUM_SAMPLES=250000
# eval 'python bagging.py --model=${MODEL} --upsample=${UPSAMPLE} --n-kobert=${N_KOBERT} --n-kogpt2=${N_KOGPT2} --n-mlbert=${N_MLBERT} --num-samples=${NUM_SAMPLES} --sample-dist=${SAMPLE_DIST} --epochs=${EPOCHS} --patience=${PATIENCE} --device=${DEVICE}'