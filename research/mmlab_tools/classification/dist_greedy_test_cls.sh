#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
CHECKPOINT=$3
LAYER_NAME_TO_CHOOSING_MATRIX_PATH=$4
RESULTS_PATH=$5
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
echo "PARAMS:"
echo $CONFIG
echo $GPUS
echo $LAYER_NAME_TO_CHOOSING_MATRIX_PATH
echo $RESULTS_PATH
echo ${@:3}
echo "DONE PARAMS"


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/greedy_test_cls.py \
    $CONFIG $CHECKPOINT $LAYER_NAME_TO_CHOOSING_MATRIX_PATH $RESULTS_PATH \
    --launcher pytorch ${@:6}