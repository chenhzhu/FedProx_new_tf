# !/usr/bin/env bash
python3 -u main.py --dataset=$1 --optimizer='fedavg'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model='mclr' \
            --drop_percent=$2 \

# MNIST
# python3 -u main.py --dataset='mnist' --optimizer='fedavg'  \
#             --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
#             --eval_every=1 --batch_size=10 \
#             --num_epochs=20 \
#             --model='mclr' \
#             --drop_percent=$2 \