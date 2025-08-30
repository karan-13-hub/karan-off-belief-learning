# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay.py \
       --save_dir /data/kmirakho/online-vdn-seed-7777777 \
       --data_dir /data/kmirakho/vdn_offline_data_seed_7777777/ \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --method vdn \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 7777777 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 131072 \
       --num_epoch 2600 \
       --epoch_len 1000 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 3 \
       --prefetch 3\
       --train_device cuda:3 \
       --act_device cuda:4,cuda:5