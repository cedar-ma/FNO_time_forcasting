#!/bin/bash
HOST=$(scontrol show hostname $SLURM_NODELIST | head -n 1) 
NODES=${NODES:-1}        
LOCAL_RANK=${PMI_RANK}

torchrun    --nproc_per_node=3 \
    --nnodes=$NODES \
    --node_rank=${LOCAL_RANK} \
    --master_addr=$HOST \
    train_lightning.py
