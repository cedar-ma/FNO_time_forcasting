#!/bin/bash
HOST=$1
NODES=$2
LOCAL_RANK=${PMI_RANK}
python --nproc_per_node=3 --nnodes=$NODES --node_rank=${LOCAL_RANK} --master_addr=$HOST train_lightning.py
