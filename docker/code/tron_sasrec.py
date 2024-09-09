# Copyright (c) 2024, ZDF.
import numpy as np
from multiprocessing import Process, Queue
import random
from typing import Any, Dict, Optional, Tuple, Union, List
import logging


def tron_negative_items_sampling(one_batch_user_seq_pos: Tuple[int, np.array, np.array], item_num : int, num_uniform_negatives: int, num_batch_negatives: int) -> Tuple[int, np.array, np.array, np.array]:

    """Sampling the negative items  with TRON(Transformer Recommender Optimized Negative Sampling)

    Args:
        one_batch_user_seq_pos (tuple): Batch of users, sequences, pos_items.
        item_num (int): number of items.
        num_uniform_negatives (int): number of negatives items that are to be sampled from total set of available items.
        num_batch_negatives (int): number of negative items that are to be sampled from the batch.

    Returns:
        One_batch_user_seq_pos (tuple): Batch of users, sequences, pos_items, negative_items.
    """

    #Step-1 Collect all the positive items from the batch of users.
    positive_examples_total_batch = [each_sample[2] for each_sample in one_batch_user_seq_pos]

    #Collect non-zero elements (i.e) item ids the user ids interacted from the user sequences. ( Total positive items availbe for batch
    non_zero_elemens_each_user_seq = [item for sample in positive_examples_total_batch for item in sample if item !=0]

    #Tracker to have samples with user, seq, pos, neg items
    batch_user_seq_pos_neg = []

    #Step-2 Iterate over each_sample of the user [(u1,seq1,pos1), (u2,seq2,pos2)..(u.n,, sq.n, pos.n)]
    for user, seq, pos in one_batch_user_seq_pos:
        
        # Step-3 (a) New array to store the negative items for each negative seq alike the postive seq.
        negative_items_array= np.zeros_like(pos, dtype=np.int32)

        # Step 3 (b) Find non-zero indices in pos seq to sample negative items.
        non_zero_positive_element_indices = np.nonzero(pos)[0]

        # Step 3 (c) Keep a track of positive elements the user has interacted with.
        positive_elements_user_interacted = pos[pos !=0]

        # Step 4 Uniform distribution or from in batch or popularity sampling.
        if num_uniform_negatives > 0:
            k = np.random.choice(np.arange(1, item_num), num_uniform_negatives)
        
        elif num_uniform_negatives == 0:
            #if num_uniform_negatives is set to zero, then generate an empty array
            k = np.array([], dtype=int)

        else:
            logging.error(f"Please set the right value of num_uniform_negatives")

        # Step 5 In batch populairty sampling with m negatives.
        if num_batch_negatives > 0:
            m = np.random.choice(non_zero_elemens_each_user_seq, size=num_batch_negatives)
        
        elif num_batch_negatives == 0:
            m = np.array([], dtype=int)
        
        else:
            logging.error(f"Please set the right value of num_batch_negatives")


        # Step 6 random Vector with k+m samples.
        random_vector_with_k_and_m_samples = np.concatenate((k,m))

        # Step 7 Get all items from both (k+m) and exlcude pos interaction items for the user.
        exclude_pos_user_elements = np.setdiff1d(random_vector_with_k_and_m_samples, positive_elements_user_interacted)

        #Step 8 Sample negative items for particular indices in reference to pos seq.
        negative_items_array[non_zero_positive_element_indices] = np.random.choice(exclude_pos_user_elements, size=len(non_zero_positive_element_indices))
        
        batch_user_seq_pos_neg.append((user, seq, pos, negative_items_array))
    
    return batch_user_seq_pos_neg


#Define the class with TRON
def sample_function_seq_positive_items_tron(user_train, usernum, itemnum, batch_size, maxlen, num_uniform_negatives, num_batch_negatives, result_queue, seed):
    # Function licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this function except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #
    # Attribution: Original sampling code
    # by Kang, Wang-Cheng, and Julian McAuley. "Self-attentive sequential recommendation." 2018
    # https://github.com/kang205/SASRec/blob/master/sampler.py
    # Modifications by  ZDF (2024): Adapted to use with TRON  negative sampling.

    """Batch sampler that creates a sequence of negative items based on the
    original sequence of items (positive) that the user has interacted with.

    Args:
        user_train (dict): dictionary of training exampled for each user
        usernum (int): number of users
        itemnum (int): number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        num_uniform_negatives (int) : number of negatives items that are to be sampled from total set of available items.
        num_batch_negatives (int): number of negative items that are to be sampled from the batch.
        result_queue (multiprocessing.Queue): queue for storing sample results
        seed (int): seed for random generator
    """

    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos)

    np.random.seed(seed)
    while True:
        one_batch_user_seq_pos = []
        for i in range(batch_size):
            one_batch_user_seq_pos.append(sample())
        one_batch = tron_negative_items_sampling(one_batch_user_seq_pos, itemnum, num_uniform_negatives, num_batch_negatives)
        result_queue.put(zip(*one_batch))

class WrapSampler_tron(object):
    # Function licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this function except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #
    # Attribution: Original sampling code
    # by Kang, Wang-Cheng, and Julian McAuley. "Self-attentive sequential recommendation." 2018
    # https://github.com/kang205/SASRec/blob/master/sampler.py
    # Modifications by  ZDF (2024): Adapted to use with TRON  negative sampling.
    """ Sampler object that creates an iterator with aiming with TRON for feeding
        batch data while training.
    Attributes:
       User: dict, all the users (keys) with items as values
       usernum: integer, totaal number of users
       itemnum: integer, total number of items
       batch size (int): batch size
       maxlen (int) : maximum input sequence length
       n_workers (int): number of workers for parallel execution
       num_uniform_negatives (int) : number of negatives items that are to be sampled from total set of available items.
       num_batch_negatives (int): number of negative items that are to be sampled from the batch.
    """

    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, num_uniform_negatives = 512, num_batch_negatives = 16):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function_seq_positive_items_tron,
                    args=(
                        User,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        num_uniform_negatives,
                        num_batch_negatives,
                        self.result_queue,
                        42
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()
        
    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
