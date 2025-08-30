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
import multiprocessing as mp

# def save_replay_buffer(replay_buffer, epoch, score):

#     size = replay_buffer.size()
#     data = {'action' : [], 'bootstrap' : [], 'h0' : [], 'obs' : [], 'reward' : [], 'seq_len' : [], 'terminal' : []}

#     for i in range(size):
#         sample = replay_buffer.get(i)
#         # import pdb; pdb.set_trace()
#         data['action'].append(sample.action)
#         data['bootstrap'].append(sample.bootstrap)
#         data['h0'].append(sample.h0)
#         data['reward'].append(sample.reward)
#         data['seq_len'].append(sample.seq_len)
#         data['obs'].append(sample.obs)
#         data['terminal'].append(sample.terminal)


#     data['epoch'] = epoch
#     data['score'] = score
#     # print(data)
#     # Save the data to a pickle file
#     folder_name = "/data/kmirakho/obl_offline_data/"
#     filename = "data_obl_"+str(epoch)+".pickle"
#     if os.path.exists(folder_name) == False:
#         os.mkdir(folder_name)
#     filename = folder_name + filename
#     print("Saving replay buffer to file: ", filename)
#     with open(filename, 'wb') as file:
#         pickle.dump(data, file)
#     file.close()
#     print("Replay buffer saved successfully.")

class SaveReplayBuffer:
    def __init__(self):
        self.manager = mp.Manager()
        self.is_saving = self.manager.Value('i', 0)
        
    def save_buffer_async(self, replay_buffer, epoch, score, args):
        """Start a non-blocking process to save the buffer"""
        if self.is_saving.value == 1:
            print("Already saving, skipping this save request")
            return
            
        # # Create a copy of the buffer to send to the process
        # buffer_copy = replay_buffer.copy()
        
        def save_job(buffer_data, epoch, score, is_saving):
            is_saving.value = 1
            size = buffer_data.size()
            folder_name = args.data_dir
            filename = "data_obl_"+str(epoch)+".pickle"
            if os.path.exists(folder_name) == False:
                os.mkdir(folder_name)
            filename = folder_name + filename
            try:
                print(f"\nSaving replay buffer with {size} experiences to {filename}\n")
                data = {'action' : [], 'bootstrap' : [], 'h0' : [], 'obs' : [], 'reward' : [], 'seq_len' : [], 'terminal' : []}

                for i in range(size):
                    sample = replay_buffer.get(i)
                    data['action'].append(sample.action)
                    data['bootstrap'].append(sample.bootstrap)
                    data['h0'].append(sample.h0)
                    data['reward'].append(sample.reward)
                    data['seq_len'].append(sample.seq_len)
                    data['obs'].append(sample.obs)
                    data['terminal'].append(sample.terminal)
                data['epoch'] = epoch
                data['score'] = score
                
                # Perform actual saving
                # Save the data to a pickle file
                with open(filename, 'wb') as file:
                    pickle.dump(data, file)
                file.close()
                print(f"\nReplay buffer at epoch: {epoch} saved successfully.\n")
            except Exception as e:
                print(f"Error saving buffer: {e}")
            finally:
                is_saving.value = 0
                
        # Start the save operation in a separate process
        save_process = mp.Process(
            target=save_job, 
            args=(replay_buffer, epoch, score, self.is_saving)
        )
        save_process.daemon = True
        save_process.start()

# if __name__ == "__main__":
#     replay_buffer_size = 10000000
#     seed = 1000000009
#     priority_exponent = 0.9
#     priority_weight = 0.6
#     prefetch = 3
#     replay_buffer = rela.RNNPrioritizedReplay(replay_buffer_size, seed, priority_exponent, priority_weight, prefetch)
#     sample = namedtuple("Sample", ["action", "bootstrap", "h0", "obs", "reward", "seq_len", "terminal"])