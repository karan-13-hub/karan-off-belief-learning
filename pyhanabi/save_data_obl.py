import time
import os
import signal
import sys

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
from functools import partial
from tqdm import tqdm
import gc
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

def process_sample_worker(args_tuple):
    """Worker function to process a single sample"""
    sample_data, sample_idx, ep_folder_name = args_tuple
    
    data = {}
    filename = os.path.join(ep_folder_name, f"{sample_idx}.npz")
    
    # Extract data from the sample
    obs = sample_data['obs']
    action = sample_data['action']
    reward = sample_data['reward']
    terminal = sample_data['terminal']
    
    data['publ_s'] = np.array(obs['publ_s'], dtype=np.float32)
    data['priv_s'] = np.array(obs['priv_s'], dtype=np.float32)
    data['legal_move'] = np.array(obs['legal_move'], dtype=np.bool_)
    data['action'] = np.array(action['a'], dtype=np.int8)
    data['reward'] = np.array(reward, dtype=np.float16)
    data['terminal'] = np.array(terminal, dtype=np.bool_)
    np.savez_compressed(filename, **data)
    # print(f"Saved sample {sample_idx} to {filename}")
    
    return sample_idx

def process_chunk_worker(args_tuple):
    """Worker function to process a chunk of samples"""
    samples_data, sample_indices, ep_folder_name = args_tuple
    
    processed_count = 0
    for i, sample_idx in enumerate(sample_indices):
        data = {}
        filename = os.path.join(ep_folder_name, f"{sample_idx}.npz")
        
        # Extract data from the sample
        sample_data = samples_data[i]
        obs = sample_data['obs']
        action = sample_data['action']
        reward = sample_data['reward']
        terminal = sample_data['terminal']
        
        data['publ_s'] = np.array(obs['publ_s'], dtype=np.float32)
        data['priv_s'] = np.array(obs['priv_s'], dtype=np.float32)
        data['legal_move'] = np.array(obs['legal_move'], dtype=np.bool_)
        data['action'] = np.array(action['a'], dtype=np.int8)
        data['reward'] = np.array(reward, dtype=np.float16)
        data['terminal'] = np.array(terminal, dtype=np.bool_)
        np.savez_compressed(filename, **data)
        processed_count += 1
    
    return processed_count

class SignalHandler:
    """Handle keyboard interrupt gracefully"""
    def __init__(self):
        self.interrupted = False
        self.original_handler = None
        
    def signal_handler(self, signum, frame):
        print("\n\nKeyboard interrupt received. Shutting down gracefully...")
        self.interrupted = True
        # Don't call sys.exit() here as it can cause issues with multiprocessing
        
    def __enter__(self):
        self.original_handler = signal.signal(signal.SIGINT, self.signal_handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_handler is not None:
            signal.signal(signal.SIGINT, self.original_handler)

class SaveReplayBuffer:
    def __init__(self, num_processes=16, use_chunking=False, chunk_size=None):
        self.manager = mp.Manager()
        self.is_saving = self.manager.Value('i', 0)
        self.num_processes = num_processes
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size or max(1, 1000 // self.num_processes)  # Default chunk size
        self.save_process = None
        
    def save_buffer_async(self, replay_buffer, epoch, score, args):
        """Start a non-blocking process to save the buffer"""
        if self.is_saving.value == 1:
            print("Already saving, skipping this save request")
            return
            
        # # Create a copy of the buffer to send to the process
        # buffer_copy = replay_buffer.copy()
        
        def save_job(buffer_data, epoch, score, is_saving, num_processes, use_chunking, chunk_size):
            is_saving.value = 1
            size = buffer_data.size()
            data_folder_name = args.data_dir
            ep_folder_name = os.path.join(data_folder_name, "data_" + str(epoch))
            if not os.path.exists(ep_folder_name):
                os.makedirs(ep_folder_name)
            
            # Set up signal handler for graceful shutdown
            signal_handler = SignalHandler()
            
            try:
                with signal_handler:
                    print(f"\nSaving replay buffer with {size} experiences to {ep_folder_name}")
                    print(f"Using {num_processes} processes for parallel processing")
                    print(f"Chunking: {'enabled' if use_chunking else 'disabled'}")
                    if use_chunking:
                        print(f"Chunk size: {chunk_size}\n")
                    else:
                        print()

                    # Extract all data from replay buffer first (to avoid pickling issues)
                    print("Extracting data from replay buffer...")
                    all_samples = []
                    for i in tqdm(range(size), desc="Extracting samples", unit="sample"):
                        if signal_handler.interrupted:
                            print("\nInterrupted during data extraction. Exiting...")
                            return
                        sample = buffer_data.get(i)
                        sample_data = {
                            'obs': sample.obs,
                            'action': sample.action,
                            'reward': sample.reward,
                            'terminal': sample.terminal
                        }
                        all_samples.append(sample_data)
                    
                    if signal_handler.interrupted:
                        print("\nInterrupted before parallel processing. Exiting...")
                        return
                    
                    print("Starting parallel processing...")
                    if use_chunking and size > num_processes * 10:
                        # Use chunked processing for large datasets
                        # Create chunks of sample indices
                        chunks = [list(range(i, min(i + chunk_size, size))) 
                                 for i in range(0, size, chunk_size)]
                        chunk_args = [(all_samples, chunk, ep_folder_name) for chunk in chunks]
                        
                        try:
                            with mp.Pool(processes=num_processes) as pool:
                                # Use tqdm to show progress for chunks
                                results = []
                                for result in tqdm(
                                    pool.imap(process_chunk_worker, chunk_args),
                                    total=len(chunk_args),
                                    desc="Processing chunks",
                                    unit="chunk"
                                ):
                                    if signal_handler.interrupted:
                                        print("\nInterrupted during chunk processing. Terminating pool...")
                                        pool.terminate()
                                        pool.join()
                                        return
                                    results.append(result)
                                
                                total_processed = sum(results)
                        except KeyboardInterrupt:
                            print("\nKeyboard interrupt received during chunk processing.")
                            return
                        
                    else:
                        # Use individual sample processing for smaller datasets
                        sample_args = [(all_samples[i], i, ep_folder_name) for i in range(size)]
                        try:                    
                            with mp.Pool(processes=num_processes) as pool:
                                # Use tqdm to show progress for individual samples
                                results = []
                                for result in tqdm(
                                    pool.imap(process_sample_worker, sample_args),
                                    total=len(sample_args),
                                    desc="Processing samples for epoch %d" % epoch,
                                    unit="sample"
                                ):
                                    if signal_handler.interrupted:
                                        print("\nInterrupted during sample processing. Terminating pool...")
                                        pool.terminate()
                                        pool.join()
                                        return
                                    results.append(result)
                                    
                        except KeyboardInterrupt:
                            print("\nKeyboard interrupt received during sample processing.")
                            return
                        except Exception as e:
                            print(f"Error processing samples: {e}")
                            return []
                        finally:
                            gc.collect()
                            
                        total_processed = len(results)
                    
                    if not signal_handler.interrupted:
                        print(f"\nReplay buffer at epoch: {epoch} saved successfully.")
                        print(f"Processed {total_processed} samples in parallel.\n")
                    else:
                        print(f"\nSave operation interrupted. Partial data may be saved.")
                        
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Exiting save operation...")
            except Exception as e:
                print(f"Error saving buffer: {e}")
            finally:
                is_saving.value = 0
                
        # Start the save operation in a separate process
        self.save_process = mp.Process(
            target=save_job, 
            args=(replay_buffer, epoch, score, self.is_saving, 
                  self.num_processes, self.use_chunking, self.chunk_size)
        )
        self.save_process.start()
    
    def stop_save_process(self):
        """Stop the save process if it's running"""
        if self.save_process is not None and self.save_process.is_alive():
            print("Terminating save process...")
            self.save_process.terminate()
            self.save_process.join(timeout=5)  # Wait up to 5 seconds
            if self.save_process.is_alive():
                print("Force killing save process...")
                self.save_process.kill()
                self.save_process.join()
            self.is_saving.value = 0
            print("Save process terminated.")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_save_process()

# if __name__ == "__main__":
#     replay_buffer_size = 10000000
#     seed = 1000000009
#     priority_exponent = 0.9
#     priority_weight = 0.6
#     prefetch = 3
#     replay_buffer = rela.RNNPrioritizedReplay(replay_buffer_size, seed, priority_exponent, priority_weight, prefetch)
#     sample = namedtuple("Sample", ["action", "bootstrap", "h0", "obs", "reward", "seq_len", "terminal"])