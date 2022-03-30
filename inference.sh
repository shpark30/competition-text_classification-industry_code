#! /bin/bash

BATCH=512
EXP_PATH='./runs/models/exp4'
eval 'python inference.py --exp-path=${EXP_PATH} --batch-size={BATCH}'

EXP_PATH='./runs/models/exp5'
eval 'python inference.py --exp-path=${EXP_PATH} --batch-size={BATCH}'

EXP_PATH='./runs/models/exp6'
eval 'python inference.py --exp-path=${EXP_PATH} --batch-size={BATCH}'