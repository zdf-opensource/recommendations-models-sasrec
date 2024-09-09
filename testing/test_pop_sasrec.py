# Copyright (c) 2024, ZDF.
import unittest
import numpy as np
import sys
sys.path.insert(0, '../docker/code')

from popularity_sampling_sasrec import WarpSampler_popularity_sampling
# Define a duplicate User dictionary with users and the interactions (item_ids)
user_train= {
    1: [1,2,3,4,5],
    2: [6,7,8,9,10],
    3 : [11,12,13,14,15],
    4: [21,22,23,24,25],
    5 : [16,17,18,19,20],
    6 : [36,37,38,39,40],
    7 : [41,42,43,44,45],
    8 : [46,47,48,49,50],
    9 : [21,22,23,24,25]
}
class TestWarpSampler(unittest.TestCase):
    
    def setUp(self):
        self.User = user_train
        self.usernum = len(self.User)
        self.itemnum = 50
        self.batch_size = 4
        self.maxlen = 10
        self.n_workers = 1
        
    def test_next_batch(self):

        sampler = WarpSampler_popularity_sampling(self.User, self.usernum, self.itemnum, self.batch_size, self.maxlen, self.n_workers)
       
        batch = sampler.next_batch()
        user_sequences = zip(*batch)

        user_sequences_dict = {}

        for i, sequences in enumerate(user_sequences, start=1):
            user_sequences_dict[f'sample{i}'] = sequences

        #postive_sequences
        positive_items_user_sequences = [user_sequences[2] for sample, user_sequences in user_sequences_dict.items()]

        #Non-zero-elements
        non_zero_elements_postive_items = [sample for each_positive_item in positive_items_user_sequences for sample in each_positive_item if sample!= 0 ]
        
        # Check for the tuple in the form (u, seq, pos, neg), and check for length, and types
        for user, sequences in user_sequences_dict.items():
            self.assertEqual(len(sequences), 4)
            self.assertIsInstance(sequences[0], int)
            self.assertIsInstance(sequences[1], np.ndarray)
            self.assertIsInstance(sequences[2], np.ndarray)
            self.assertIsInstance(sequences[3], np.ndarray)

            positive_items_indices_nonzero = np.nonzero(sequences[2])
            negative_items_indices_nonzero = np.nonzero(sequences[3])

            #Check if indices of non zero elements in pos and negative sequences are same
            self.assertTrue(np.array_equal(positive_items_indices_nonzero, negative_items_indices_nonzero))

            positive_zero_elements = np.where(sequences[2] == 0)[0]
            negative_zero_elements = np.where(sequences[3] == 0)[0]

            #Check if indices of zero elements in pos and negative sequences are same
            self.assertTrue(np.array_equal(positive_zero_elements, negative_zero_elements))

            non_zero_negative_popsmapled_items = [each_negative_item for each_negative_item in sequences[3] if each_negative_item!= 0]

            all_exist = all(value in non_zero_elements_postive_items for value in non_zero_negative_popsmapled_items)

            #Check if the negative_items sampled are within the postive items within the batch
            self.assertTrue(all_exist)
        
        sampler.close()

if __name__ =="__main__":
    unittest.main()
