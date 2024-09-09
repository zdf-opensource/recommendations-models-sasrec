# Copyright (c) 2024, ZDF.
import json
import logging
import os
import re
import sys
from typing import Dict, Any, List, Tuple

import dataclasses


@dataclasses.dataclass
class Hyperparams:
    """
    Hyperparams dataclass
    """

    # for preprocess and train of SASRec model
    model_variant_name: str
    hyperparams: Dict[str, str] = dataclasses.field(default_factory=dict)

    train_with_brand_mapping: bool = False

    filter_params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # Hyperparameters to be used during Hyperparameter Tuning only for SARSEC
    hpt_n_days: int = None
    hpt_n_num_epochs: int = None
    hpt_hidden_units: int = None
    hpt_batch_size: int = None
    hpt_learning_rate: float = None
    hpt_l2_emb: float = None
    hpt_dropout_rate : float = None
    hpt_maxlen : int = None
    hpt_num_blocks : int = None
    hpt_num_heads : int = None
    hpt_user_count_threshold : int = None
    hpt_loss_function : str = None
    hpt_calibration_gbce : int = None
    hpt_num_uniform_negatives : int = None
    hpt_num_batch_negatives : int = None
    hpt_top_k_negatives : int = None

    @property
    def targets(self) -> List[str]:
        # allow to have more than one target (e.g. for the stage model)
        targets = list(
            set(
                re.findall(
                    r"(?:[\w-]|^)(pctablet|tv|mobile)(?=[\w-]|$)",
                    self.model_variant_name
                )
            )
        )
        if len(targets) < 1:
            raise ValueError(
                "No targets defined in the model_variant_name. "
                "Add at least one option of the following: pctablet|tv|mobile"
            )
        return targets

    @property
    def n_days(self) -> int:
        if self.hpt_n_days is not None:
            logging.info(f"using hpt_n_days = {self.hpt_n_days}")
            return self.hpt_n_days
        return self.hyperparams.get("n_days", 30)
    
    # The user is included only when has interacted with more less than or equal to "user_count_threshold"
    #Default value is 2
    @property
    def user_count_threshold(self) -> int:
        if self.hpt_user_count_threshold is not None:
            logging.info(f"using hpt_user_count_threshold = {self.hpt_user_count_threshold}")
            return self.hpt_user_count_threshold
        return self.hyperparams.get("user_count_threshold", 2)
    
    # The item/ external_id is included only when external_id has  atleast 'min_item_interactions'
    #Default value is 20
    @property
    def min_item_interactions(self) -> int:
        return self.hyperparams.get("min_item_interactions", 20)
    
    @property
    def minimum_coverage(self) -> float:
        return self.hyperparams.get("minimum_coverage", 0.35)

    @property
    def num_epochs(self) -> int:
        if self.hpt_n_num_epochs is not None:
            logging.info(
                f"using hpt_n_num_epochs = {self.hpt_n_num_epochs}"
            )
            return self.hpt_n_num_epochs
        return self.hyperparams.get("num_epochs", 20)

    @property
    # hidden units, attention dim and conv dimension are taken as the same for SARSec
    def hidden_units(self) -> int:
        if self.hpt_hidden_units is not None:
            logging.info(f"using hpt_hidden_units = {self.hpt_hidden_units}")
            return self.hpt_hidden_units
        return self.hyperparams.get("hidden_units", 100)

    @property
    def batch_size(self) -> int:
        if self.hpt_batch_size is not None:
            logging.info(f"using hpt_batch_size = {self.hpt_batch_size}")
            return self.hpt_batch_size
        return self.hyperparams.get("batch_size", 256)

    @property
    def learning_rate(self) -> float:
        if self.hpt_learning_rate is not None:
            logging.info(f"using hpt_learning_rate = {self.hpt_learning_rate}")
            return self.hpt_learning_rate
        return self.hyperparams.get("learning_rate", 0.00011)

    @property
    def l2_emb(self) -> float:
        if self.hpt_l2_emb is not None:
            logging.info(f"using hpt_l2_emb = {self.hpt_l2_emb}")
            return self.hpt_l2_emb
        return self.hyperparams.get("l2_emb", 0.0001)
    
    @property
    def dropout_rate(self) -> float:
        if self.hpt_dropout_rate is not None:
            logging.info(f"using hpt_dropout_rate = {self.hpt_dropout_rate}")
            return self.hpt_dropout_rate
        return self.hyperparams.get("dropout_rate", 0.2)
    
    @property
    def maxlen(self) -> int:
        if self.hpt_maxlen is not None:
            logging.info(f"using hpt_maxlen ={self.hpt_maxlen}")
            return self.hpt_maxlen
        return self.hyperparams.get("maxlen", 10)
        
    @property
    def num_heads(self) -> int:
        if self.hpt_num_heads is not None:
            logging.info(f"using hpt_num_heads ={self.hpt_num_heads}")
            return self.hpt_num_heads
        return self.hyperparams.get("num_heads", 1)
    
    @property
    def num_blocks(self) -> int:
        if self.hpt_num_blocks is not None:
            logging.info(f"using number of hpt_num_blocks = {self.hpt_num_blocks}")
            return self.hpt_num_blocks
        return self.hyperparams.get("num_blocks", 2)
    
    @property
    def evaluate_model(self) -> bool:
        return self.hyperparams.get("evaluate_model", False)
    
    @property
    def data_split_type(self) -> str:
        return self.hyperparams.get("data_split_type", "time_based_split")
    
    @property
    def use_fixed_data(self) -> bool:
        return self.hyperparams.get("use_fixed_data", False)

    #Select sampling_type to "tron" or "batch_pop_sampling" (Two diferent methods sampling strategies)
    #In any other case or not selecting the two methods uniform sampling is done.
    @property
    def sampling_type(self) -> str:
        return self.hyperparams.get("sampling_type", None)
    
    @property
    def num_uniform_negatives(self) -> int:
        if self.hpt_num_uniform_negatives is not None:
            logging.info(f"using hpt_num_uniform_negatives = {self.hpt_num_uniform_negatives}")
            return self.hpt_num_uniform_negatives
        return self.hyperparams.get("num_uniform_negatives", 64)
    
    @property
    def num_batch_negatives(self) -> int:
        if self.hpt_num_batch_negatives is not None:
            logging.info(f"using hpt_num_batch_negatives = {self.hpt_num_batch_negatives}")
            return self.hpt_num_batch_negatives
        return self.hyperparams.get("num_batch_negatives", 16)
    
    @property
    def apply_topk_function(self) -> bool:
        return self.hyperparams.get("apply_topk_function", False)
    
    @property
    def top_k_negatives(self) -> int:
        if self.hpt_top_k_negatives is not None:
            logging.info(f"using hpt_top_k_negatives = {self.hpt_top_k_negatives}")
            return self.hpt_top_k_negatives
        return self.hyperparams.get("top_k_negatives", 200)
        
    #Select "gbce" to employ gbce loss_function
    # else the default bce loss is considered for model training
    @property
    def loss_function(self) -> str:
        if self.hpt_loss_function is not None:
            logging.info(f"using hpt_loss_function {self.hpt_loss_function}")
            return self.hpt_loss_function
        return self.hyperparams.get("loss_function", "bce")
    
    #Calibration_gbce is used to control the power of the positive scores in gbce loss function.
    @property
    def calibration_gbce(self) -> float:
        if self.hpt_calibration_gbce is not None:
            logging.info(f"using hpt_power_parameter_gbce {self.hpt_calibration_gbce}")
            return self.hpt_calibration_gbce
        return self.hyperparams.get("calibration_gbce", 1.0)
    
    @property
    def epsilon_loss_value(self) -> float:
        return self.hyperparams.get("epsilon_loss_value",  1e-24)
    
    @property
    def gradient_clipping(self) -> bool:
        return self.hyperparams.get("gradient_clipping", False)
    
    @property
    def n_workers(self) -> int:
        return self.hyperparams.get("n_workers", 3)

    @property
    def metrics(self) -> List[str]:
        return self.hyperparams.get(
            "metrics", ["ndcg_at_n", "hit_at_n", "hit_at_1", "mrr", "pop_q_1"]
        )
    
    @property
    def eval_batchsize(self) -> int:
        return self.hyperparams.get("eval_batchsize", 256)
    
    @property
    def top_nitems(self) -> int:
        return self.hyperparams.get("top_nitems", 5)
    
    @property
    def test_size(self) -> int:
        return self.hyperparams.get("test_size", 0.1)

    @property
    def log_cloud_watch(self) -> bool:
        return self.hyperparams.get("log_cloud_watch", False)
    

    @classmethod
    def from_json(cls: "Hyperparams", **kwargs):
        """
        instantiates :class:`Hyperparams` with all keys from json that exist in :class:`Hyperparams`

        expects either a single kwarg ``json_str`` that contains all relevant items
        or multiple keys corresponding to ``dataclasses.fields(Hyperparams)``
        """
        if "json_str" in kwargs:
            # assume all params are passed as one dict
            json_str = kwargs["json_str"]
            if isinstance(json_str, str):
                json_dict = json.loads(json_str)
            else:
                json_dict = json_str
        else:
            # assume params are passed as multiple params
            json_dict = kwargs
        field_value_dict = {}
        for field in set(json_dict.keys()).intersection(
            {field.name for field in dataclasses.fields(cls)}
        ):
            value = json_dict.get(field)
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # this is expected, for example for string fields
            if value is not None:
                field_value_dict[field] = value
        return cls(**field_value_dict)


def load_hyperparams(SM_PATH_PREFIX :str) -> Tuple[str, Hyperparams]:
    # This is where params passed in to SageMaker are stored
    param_path = os.path.join(SM_PATH_PREFIX, 'input/config/hyperparameters.json')

    with open(param_path) as f:
        hyperparams: dict = json.load(f)

    logging.info("----- Hyperparameters (JSON) -----")
    logging.info(hyperparams)

    if 'SITE' not in hyperparams:
        logging.error('You must specify the "SITE" hyperparameter!')
        logging.error('Exiting')
        sys.exit(255)
    site = hyperparams['SITE']
    os.environ['SITE'] = site

    parsed_hp = Hyperparams.from_json(json_str=hyperparams)

    logging.info("----- Hyperparameters -----")
    logging.info(parsed_hp)

    return site, parsed_hp
