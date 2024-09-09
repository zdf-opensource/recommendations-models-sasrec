# Copyright (c) 2024, ZDF.
import numpy as np
from multiprocessing import Process, Queue
import random
from typing import Any, Dict, Optional, Tuple, Union, List

from recommenders.models.sasrec.sampler import WarpSampler, sample_function

def popularity_sampling (one_batch_user_seq_pos : Tuple[int, np.array, np.array])  -> Tuple[int, np.array, np.array, np.array]:

    #Step-1 Collect all the positive items from the batch of users.
    positive_examples_total_batch = [each_sample[2] for each_sample in one_batch_user_seq_pos]

    #Collect non-zero elements (i.e) item ids the user ids interacted from the user sequences.
    non_zero_elemens_each_user_seq = [item for sample in positive_examples_total_batch for item in sample if item !=0]

    #Tracker to have samples with user, seq, pos, neg items
    batch_user_seq_pos_neg = []

    #Step-2 Iterate over all the samples
    for each_sample in one_batch_user_seq_pos:
        #Retrieve the user,seq, positive items for each sample
        user, seq, pos = each_sample

        # Step-2 (a) Have a new array to the negative items for each sample
        negative_items_sample = np.zeros_like(pos, dtype=np.int32)

        # Step 2 (b) Find non-zero indices in pos seq
        non_zero_positive_elements = np.nonzero(pos)[0]

        # Step 3  For the each set of positive item interactions sample the negative item
        for index in non_zero_positive_elements:
            negative_items_sample[index]  = random.choice(non_zero_elemens_each_user_seq)
        
        batch_user_seq_pos_neg.append((user, seq, pos, negative_items_sample))
    
    return batch_user_seq_pos_neg


def sample_function_seq_positive_items(user_train, usernum, itemnum, batch_size, maxlen, result_queue, seed):
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
    # Modifications by  ZDF (2024): Adapted to use with popularity sampling.
    """Batch sampler that creates a sequence of negative items based on the
    original sequence of items (positive) that the user has interacted with.

    Args:
        user_train (dict): dictionary of training exampled for each user
        usernum (int): number of users
        itemnum (int): number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
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
        one_batch = popularity_sampling(one_batch_user_seq_pos)
        result_queue.put(zip(*one_batch))


class WarpSampler_popularity_sampling(object):
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
    # Modifications by  ZDF (2024): Adapted to use with popularity sampling.
    """Sampler object that creates an iterator for feeding batch data while training.

    Attributes:
        User: dict, all the users (keys) with items as values
        usernum: integer, total number of users
        itemnum: integer, total number of items
        popularity_last_item_ids_user_clicked: dict, popularity of all last item_ids each user has clicked
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        n_workers (int): number of workers for parallel execution
    """

    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function_seq_positive_items,
                    args=(
                        User,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        np.random.randint(2e9)
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

