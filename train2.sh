#! /bin/bash

# 3월 29일 월요일
MODEL='kobert'
LOSS='FCE'
EPOCH=10
BATCH=512
OPTIMIZER='AdamW'
DEVICE='cuda:1'
<<<<<<< HEAD
LEARNINGRATE=0.00005

# eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epochs=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER} --learning-rate=${LEARNINGRATE}'

BETA=0.9
eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epochs=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER} --learning-rate=${LEARNINGRATE} --beta1=${BETA}'


BETA=0.1
eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epochs=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER} --learning-rate=${LEARNINGRATE} --beta1=${BETA}'


LOSS='CE'
LEARNINGRATE=0.00001
BETA=0.1
eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epochs=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER} --learning-rate=${LEARNINGRATE} --beta1=${BETA}'

LOSS='CE'
LEARNINGRATE=0.00001
BETA=0.9
eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epochs=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER} --learning-rate=${LEARNINGRATE} --beta1=${BETA}'


LOSS='FCE'
LEARNINGRATE=0.00001
BETA=0.9
eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epochs=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER} --learning-rate=${LEARNINGRATE} --beta1=${BETA}'

LOSS='CE'
LEARNINGRATE=0.00001
BETA=0.1
eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epochs=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER} --learning-rate=${LEARNINGRATE} --beta1=${BETA}'


# OPTIMIZER='Adam'
# eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER}'
=======
eval 'python train.py --model=${MODEL} --batch-size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER}'

OPTIMIZER='Adam'
eval 'python train.py --model=${MODEL} --batch-size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER}'
>>>>>>> bd21f4cccc8883b35d70e569d1eb1b67130fbf11

MODEL='kobert'
LOSS='FCE'
EPOCH=10
BATCH=512
BETA1=0.9
OPTIMIZER='AdamW'
UPSAMPLE='--upsample'
NUM_TEST=30000
LR=5e-4
DEVICE='cuda:1'

eval 'python train.py --model=${MODEL} --num-test=${NUM_TEST} -lr=${LR} ${UPSAMPLE} --lr-scheduler=${SCHEDULER} --batch-size=${BATCH} --learning-rate=${LR} --loss=${LOSS} --epoch=${EPOCH} --optimizer=${OPTIMIZER} --beta1=${BETA1} --device=${DEVICE}'