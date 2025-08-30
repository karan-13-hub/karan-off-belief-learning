import time
import os

import numpy as np
import torch
from torch import nn

from act_group import ActGroup
# from create import create_envs, create_threads
# from eval import evaluate
# import common_utils
import rela
import hanalearn
import pickle
import numpy as np
from collections import namedtuple

def save_replay_buffer(replay_buffer, epoch, score):

    size = replay_buffer.size()
    data = {'action' : [], 'bootstrap' : [], 'h0' : [], 'obs' : [], 'reward' : [], 'seq_len' : [], 'terminal' : []}

    for i in range(size):
        sample = replay_buffer.get(i)
        # import pdb; pdb.set_trace()
        data['action'].append(sample.action)
        data['bootstrap'].append(sample.bootstrap)
        data['h0'].append(sample.h0)
        data['reward'].append(sample.reward)
        data['seq_len'].append(sample.seq_len)
        data['obs'].append(sample.obs)
        data['terminal'].append(sample.terminal)

    data['epoch'] = epoch
    data['score'] = score
    # print(data)
    # Save the data to a pickle file
    folder_name = "/data/kmirakho/obl_offline_data/"
    filename = "data_"+str(epoch)+".pickle"
    if os.path.exists(folder_name) == False:
        os.mkdir(folder_name)
    filename = folder_name + filename
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# if __name__ == "__main__":
#     replay_buffer_size = 10000000
#     seed = 1000000009
#     priority_exponent = 0.9
#     priority_weight = 0.6
#     prefetch = 3
#     replay_buffer = rela.RNNPrioritizedReplay(replay_buffer_size, seed, priority_exponent, priority_weight, prefetch)
#     sample = namedtuple("Sample", ["action", "bootstrap", "h0", "obs", "reward", "seq_len", "terminal"])