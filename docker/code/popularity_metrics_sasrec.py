# Copyright (c) 2024, ZDF.
from collections import defaultdict
from scipy import stats
from typing import List, Tuple, Dict, DefaultDict

import numpy as np
class PopQ1():
    """
    Calculation of popularity quantile @1.

    Args:
        externalid_occurences (dict):each_external_id_counts

    Citation: "Countering Popularity Bias by Regularizing Score Differences"

    """

    def __init__(self, external_ids_count: Dict[str, int]):
        self.external_ids_count = external_ids_count
        self.each_ext_popularity_quantile: Dict[str,int] = {
        each_ext_id: sum(1 for other_external_id_occurence in self.external_ids_count.values() if other_external_id_occurence < each_occurence) / len(self.external_ids_count)
        for each_ext_id, each_occurence in self.external_ids_count.items()}
    
    def compute(self, predictions: List[Tuple[str, float]])  -> float:
        #Get the ext_id for the first recommendation of the user
        first_reco_ext_id = predictions[0][0]

        #Get the popularity quantile for the first_reco for the user
        pop_quantile_1 = self.each_ext_popularity_quantile.get(first_reco_ext_id,0)

        return pop_quantile_1

class PopRankCor():
    """
    Calculation of popularity rank correlation for items using Spearman’s rank correlation coefficient.

    Args:
        externalid_occurences (dict):each_external_id_counts

    Citation: "Popularity-Opportunity Bias in Collaborative Filtering"

    """

    def __init__(self, external_ids_count: Dict[str, int]):
        self.external_ids_count = external_ids_count
        self.ext_ids_ranked : DefaultDict[str, List[int]] = defaultdict(list)
        self.ext_id_avg_rank_values : DefaultDict[str, List[float]] = defaultdict(list)

        self.popularity_ext_ids: np.nd.array = np.array([])  
        self.avg_ranked_external_ids: np.nd.array = np.array([]) 

    def compute_ranks_for_recos_for_each_user(self, recos_generated : List[Tuple[str, int]]) ->  Dict[str, List[int]]:

        #Compute the ranks for the recos generated for each user and convert it into dictionary
        recos_ranks_each_user = {key: rank + 1 for rank, (key,_) in enumerate(recos_generated)}

        #For each external_id store all the ranks
        for ext_id, rank in recos_ranks_each_user.items():
            self.ext_ids_ranked[ext_id].append(rank)

        return self.ext_ids_ranked
    
    def average_the_ranks_for_each_item(self, ext_id_ranks : Dict[str, List[int]]) -> Dict[str, float]:
        
        #Average the rank for each external_id
        self.ext_id_avg_rank_values = {ext_id : sum(ranks_list) / len(ranks_list) for ext_id, ranks_list in ext_id_ranks.items()}

        return self.ext_id_avg_rank_values
    
    def compute(self, popularity_all_external_ids : Dict[str, int], avg_ranked_ext_values: Dict[str, float]) -> float:
        
        #Get all ext_ids_values from intersection and Obtain PRI
       common_external_ids = set(popularity_all_external_ids.keys()) & set(avg_ranked_ext_values.keys())

       self.popularity_external_ids = np.array([popularity_all_external_ids[each_external_id] for each_external_id in common_external_ids])
       self.avg_ranked_external_ids = np.array([avg_ranked_ext_values[each_external_id] for each_external_id in common_external_ids ])

       pri = stats.spearmanr(self.popularity_external_ids , self.avg_ranked_external_ids)

       return -pri.statistic
