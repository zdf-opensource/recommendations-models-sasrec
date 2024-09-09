# Copyright (c) 2024, ZDF.
import pytest
import numpy as np
import sys
sys.path.insert(0, '../docker/code')

from popularity_sampling_sasrec import WarpSampler_popularity_sampling

user_sequnces_1= {
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

user_sequnces_2= {
    100: [1000,2000,3000,4000],
    200: [2000,3000,4000,5000],
    300 : [1,2,3,4],
    4: [90,70,80,20,30,40]}


@pytest.mark.parametrize("user_sequnces, usernum, item_num, batch_size, expected_data_user, expected_data_types_sequnces",
    [
    (user_sequnces_1,len(user_sequnces_1), 50, 5, int, np.ndarray),
    (user_sequnces_2,len(user_sequnces_2), 20, 2, int, np.ndarray),
    ],
)
def test_next_batch(user_sequnces, usernum, item_num, batch_size, expected_data_user, expected_data_types_sequnces):

    sampler = WarpSampler_popularity_sampling(user_sequnces, usernum, item_num, batch_size)

    each_batch = sampler.next_batch()
    users_data_unzipped = zip(*each_batch)

    all_users_data = {}
    
    for i, sequneces in enumerate(users_data_unzipped, start=1):
        all_users_data[f'sample{i}'] = sequneces

    # Check for the tuple in the form (u, seq, pos, neg), and check for the data types
    for user, sequences in all_users_data.items():
            
            assert type(sequences[0]) == expected_data_user
            
            #Check if the sequences, pos, negative items are of the form of numpy array
            assert type(sequences[1]) == expected_data_types_sequnces
            assert type(sequences[2]) == expected_data_types_sequnces
            assert type(sequences[3]) == expected_data_types_sequnces
