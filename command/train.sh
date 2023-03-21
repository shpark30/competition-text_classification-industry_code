#! /bin/bash

MODEL='kogpt2'
LOSS='FCE'
EPOCH=30
BATCH=512
OPTIMIZER='AdamW'
eval 'python train.py --model=${MODEL} --batch-size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --optimizer=${OPTIMIZER}'

OPTIMIZER='Adam'
eval 'python train.py --model=${MODEL} --batch-size=${BATCH} --loss=${LOSS} --epoch=${EPOCH} --optimizer=${OPTIMIZER}'