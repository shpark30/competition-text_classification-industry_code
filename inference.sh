#! /bin/bash

BATCH=512
EXP_PATH='./runs/train/exp4'
eval 'python inference.py --exp-path=${EXP_PATH} --batch-size=${BATCH}'

EXP_PATH='./runs/train/exp5'
eval 'python inference.py --exp-path=${EXP_PATH} --batch-size=${BATCH}'