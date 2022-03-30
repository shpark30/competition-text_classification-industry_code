#! /bin/bash

# 3월 29일 월요일
MODEL='kobert'
LOSS='FCE'
EPOCH=30
BATCH=512
OPTIMIZER='AdamW'
DEVICE='cuda:1'
eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER}'

OPTIMIZER='Adam'
eval 'python train.py --model=${MODEL} --batch_size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --device=${DEVICE} --optimizer=${OPTIMIZER}'