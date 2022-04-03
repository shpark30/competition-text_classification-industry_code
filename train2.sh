#! /bin/bash

# 3월 29일 월요일
MODEL='kobert'
LOSS='FCE'
EPOCH=10
BATCH=512
OPTIMIZER='AdamW'
DEVICE='cuda:1'
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