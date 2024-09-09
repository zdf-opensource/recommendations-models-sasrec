# Copyright (c) 2024, ZDF.
import logging
import operator
import pickle
from pathlib import Path
import time
from itertools import groupby
from os import path
from typing import Any, Dict, Optional,Iterable, Tuple, Union, List, TypeVar

import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import sys

# Set the TensorFlow log level to INFO
tf.get_logger().setLevel("INFO")

# Log TensorFlow version
tf_version = tf.__version__
tf.get_logger().info(f"TensorFlow version: {tf_version}")

# Log the number of available GPUs
gpus_available = len(tf.config.list_physical_devices('GPU'))
tf.get_logger().info(f"Number of GPUs Available: {gpus_available}")

from recommenders.models.sasrec.sampler import WarpSampler
from sasrec_model import CustomSASRec
from sasrec_data import CustomSASRecDataSet
from popularity_sampling_sasrec import WarpSampler_popularity_sampling
from tron_sasrec import WrapSampler_tron
from pa_base.zdf.configuration.dynamic_configs import PLAY_COVERAGE_THRESHOLD, PLAY_MAX_AGE_DAYS, MAX_SEQUENCE_LEN_SEQUENTIAL
from pa_base.zdf.data.dataframes import get_content_df, get_denoised_data
from pa_base.models.base_model import normalize_scores, normalize_scores_batches
from pa_base.train.model_evaluation import (
    leave_out_last_split,
    train_test_validation_split_by_time,
    train_test_validation_split_by_users,
    MLmetricsCalculator,
    MLmetricsLogger
    
)
from pa_base.data.s3_util import download_dataframe_from_s3
from pa_base.configuration.config import _S3_BASENAME
from mlflow.exceptions import MissingConfigException

from hyperparams import Hyperparams

#Bucket name to be used for obtaining the golden dataset or fixed dataset
MODEL_VALIDATION_DATA_BUCKET = f"{_S3_BASENAME}.model-validation-data"


def define_the_sampler(data : CustomSASRecDataSet, hyperparams : Hyperparams) -> WarpSampler: 

    if hyperparams.sampling_type == "tron":
        logging.info(f"Using {hyperparams.sampling_type} with {hyperparams.num_uniform_negatives} uniform negatives and {hyperparams.num_batch_negatives} batch negatives.")
        sampler = WrapSampler_tron(data.user_data, data.usernum, data.itemnum, batch_size = hyperparams.batch_size, maxlen = hyperparams.maxlen, n_workers= hyperparams.n_workers, num_uniform_negatives=hyperparams.num_uniform_negatives, num_batch_negatives=hyperparams.num_batch_negatives)
    
    elif hyperparams.sampling_type == "batch_pop_sampling":
        logging.info(f"Using {hyperparams.sampling_type} ")
        sampler = WarpSampler_popularity_sampling(data.user_data, data.usernum, data.itemnum, batch_size = hyperparams.batch_size, maxlen = hyperparams.maxlen, n_workers= hyperparams.n_workers)
    
    else:
        logging.info(f"Uniform sampling is done, as the 'sampling_type' is set to '{hyperparams.sampling_type}' ")
        sampler = WarpSampler(data.user_data, data.usernum, data.itemnum, batch_size = hyperparams.batch_size, maxlen = hyperparams.maxlen, n_workers= hyperparams.n_workers)

    return sampler

def train_test_validation_split(data_split_type, data, test_size, min_views_user):

    """Splits the data either with usage based or time based """

    if data_split_type == "time_based_split":
        logging.info("Splitting the data time wise")
        train, test, _ = train_test_validation_split_by_time(data, test_size = test_size, with_validation_set=False, min_views_user = min_views_user)
        
    elif data_split_type == "usage_based_split":
         logging.info("Splitting the data users wise")
         train, test, _ = train_test_validation_split_by_users(data, test_size = test_size, with_validation_set=False,  min_views_user = min_views_user)
    
    else:
        logging.error(f"Please check the hyperparam {data_split_type}. Please set it to either time_based_split or usage_based_split")
        sys.exit(1)
    
    return train, test, _

def count_extids_popularity(train_data):
    """
    Counts the externalids for the training dataset to be used for popularity metric
    
    """
    external_ids_count = train_data['externalid'].value_counts()
    external_ids_count_values = external_ids_count.to_dict()

    return external_ids_count_values

    
def data_split(data, model_desc, ml_metric_logger, data_split_type, test_size=0.1, min_views_user=-1):

    """Evaluates the model after complete training has been done
       The data is split into train and test sets respectively """

    logging.info("Splitting the data explicitly for SASRec evaluation")

    train, test, _ = train_test_validation_split(data_split_type, data, test_size, min_views_user)
    
    # Logging the reseptive dataset with number of unique users, items, and number of interactions as mlflow metrics.
    ml_metric_logger.log_mlflow_dataset_metrics(test, "test")
    ml_metric_logger.log_mlflow_dataset_metrics(train, "train")

    user_set_test, item_set_test = set(test['uid'].unique()), set(test['externalid'].unique())
    logging.info(f"For Sasrec testing, Total {len(user_set_test)} unique uids, and unique {len(item_set_test)} externalids are considered respectively")
    logging.info("Train, test splitting has been done")

    external_ids_count_values = count_extids_popularity(train)

    #Test data preparation  
    test_data = CustomSASRecDataSet(data=test)

    #Prepare data for SASREC Model training
    train_data, sasrec_model_meta_data, itemid_for_extid = prepare_data_for_training(train, model_desc)
    logging.info("Data preparation done for model training")

    return train_data, sasrec_model_meta_data, test_data, itemid_for_extid, external_ids_count_values


def create_test_batches(sequences:List[List[int]], batch_size:int):
    """
    Creates batches for the evaluation
    """
    for i in range(0, len(sequences), batch_size):
        yield sequences[i:i+batch_size]


# Function to extract item ids from predictions and targets
def extract_item_ids_predictions(batch_predictions_normalizedscores :List[List[Tuple[int, str]]])-> List[List[int]]:
    """
    Extract and store the item ids obtained from the particular batch of model predictions.
    : param batch_predictions_normalizedscores : obtained batch predictions of different sequences of items ids with their scores.
      [[seq_1_predictions], [seq_2_predictions]] = [[(item_id_1, score), (item_id_2, score)], [(item_id_1, score), (item_ids_2, score)]]                      
      An example [[(10, 1), (20, 0.3)], [(100, 1.0), (10, 0.5)]]
    : return: The items for the batch
      [[seq_1_item_ids_interacted], [seq_2_item_ids_interacted]]  = [[item_id_1, item_id_2], [[item_id_1, item_id_2]]    
      An example [[10, 20], [100, 10]]
    """

    return [[prediction[0] for prediction in each_sequence_pred] for each_sequence_pred in batch_predictions_normalizedscores]


def offline_evaluation(test_data : Dict[int, List[int]] , external_ids_count: Dict[str, int], itemid_for_extid : Dict[str, int], hyperparams : Hyperparams,  ml_metric_logger: MLmetricsLogger) -> Tuple[float, float] :

    """Calculates the offline metrics evaluating the performance of the model
    : param test_data : test data user item ids interactions.
    : param external_ids_count: Count of external ids after train test split for popularity calculation.
    : param itemid_for_extid: Mappings with external_id as key and item_id as value.
    : param hyperparams: The set hyperparameters.
    : param ml_metric_logger: Logger for updating the metric and time values.
    : return: Calculated metric values and total test duration.
    """

    logging.info(F"Offline evaluation started")

    from predictor import ModelService
    sasrec_model = ModelService()

    timestamp_start_test = time.time()

    metrics_str = ', '.join(hyperparams.metrics)
    logging.info(f"Calculating the provided metrics '{metrics_str}' for evaluation")

    #Map the ext_ids to item_ids for popularity metric
    item_id_count_popularity = { itemid_for_extid[ext_id]: count for ext_id, count in external_ids_count.items() if ext_id in itemid_for_extid}
    metrics_calculator = MLmetricsCalculator(metrics= hyperparams.metrics, item_ids_count=item_id_count_popularity)

    # Filter the test sequnces with greater than or equal to 2 and consider the  last ten interactions for the user within the interactions
    test_sequences = [ seq[-hyperparams.maxlen:] if len(seq) >= hyperparams.maxlen else seq
    for seq in test_data.values() if len(seq) >=2]

    total_number_test_user_sequences = len(test_sequences)
    logging.info(f"Total number of test sequences are {total_number_test_user_sequences}")

    time_for_sequences_mapping_and_filtering =  time.time() - timestamp_start_test
    ml_metric_logger.log_metric("test", "seq_mapping_and_filtering_time", time_for_sequences_mapping_and_filtering)

    total_batches = (total_number_test_user_sequences + hyperparams.eval_batchsize - 1) // hyperparams.eval_batchsize
    logging.info(f"Total batches considered for evaluation are {total_batches} with 'eval_batchsize' taken as {hyperparams.eval_batchsize}")

    progress_interval = total_batches // 4
    next_progress_mark = progress_interval
    batches_processed = 0

    # All targets
    all_predictions_item_ids = []
    all_targets_item_ids = []

    timestamp_start_test_batches_collection =  time.time()

    for batch_sequences in create_test_batches(test_sequences, hyperparams.eval_batchsize):

        input_model_batch_sequences, batch_targets = zip(*[leave_out_last_split(seq) for seq in batch_sequences])

        #Prepare the input sequences for predicstions with padding.
        padded_model_input_sequences = np.array([np.pad(seq, (hyperparams.maxlen - len(seq), 0), 'constant') for seq in input_model_batch_sequences])

        batch_predictions_all_items = sasrec_model.predict_batch_eval(padded_model_input_sequences)

        #Convert into numpy
        batch_predictions_all_items_numpy = batch_predictions_all_items.numpy()

        # Obtain the batchpreds and  obtain the top 25 ranked items accoring to the scores
        batchpreds_index_score =  [list(enumerate(each_pred, start=1)) for each_pred in batch_predictions_all_items_numpy]
        batch_predictions_ranked = [sorted(eachbacth_pred, key=operator.itemgetter(1), reverse=True)[:25] for eachbacth_pred in batchpreds_index_score]

        batch_predictions_normalizedscores = normalize_scores_batches(batch_predictions_ranked)

        # Filter consider only predictions that are not in the input sequences
        batch_predictions = [[pred for pred in preds if pred[0] not in seq ] for seq, preds in zip(input_model_batch_sequences, batch_predictions_normalizedscores)]

        #For each batch use list comprehension to accumulate items for predictions from the model and targets.
        all_predictions_item_ids.extend(extract_item_ids_predictions(batch_predictions))
        
        all_targets_item_ids.extend(batch_targets)

        # Update the count of processed batches
        batches_processed += 1

         # Log progress
        if batches_processed >= next_progress_mark:
            percentage_batches_processed = (batches_processed / total_batches) * 100
            logging.info(f"{percentage_batches_processed:.2f} % of batches processed with the targets and predictions collected")
            next_progress_mark += progress_interval
    
    timestamp_all_batches_collection = time.time() - timestamp_start_test_batches_collection
    ml_metric_logger.log_metric("test", "time_for_all_batches_collection", timestamp_all_batches_collection)

    timestamp_for_all_batches_metrics_calculation = time.time()

    offline_metric_values = metrics_calculator.compute_offline_metrics( all_targets_item_ids, all_predictions_item_ids, hyperparams.top_nitems)

    time_for_all_batches_metrics_calculation = time.time() - timestamp_for_all_batches_metrics_calculation
    ml_metric_logger.log_metric("test", "time_for_all_batches_offline_metrics_calculation", time_for_all_batches_metrics_calculation)

    for metric_name, metric_value in offline_metric_values.items():
        if metric_name == "ndcg_at_n":
            ml_metric_logger.log_metric("test", f"ndcg_at_{hyperparams.top_nitems}", metric_value, log_cloud_watch=hyperparams.log_cloud_watch)
        
        elif metric_name == "hit_at_n":
            ml_metric_logger.log_metric("test", f"hit_at_{hyperparams.top_nitems}", metric_value, log_cloud_watch=hyperparams.log_cloud_watch)

        else:
            ml_metric_logger.log_metric("test", metric_name, metric_value, log_cloud_watch=hyperparams.log_cloud_watch)

    total_test_duration = time.time() - timestamp_start_test
    ml_metric_logger.log_metric("test", "Total_test_duration", total_test_duration)

    return offline_metric_values, total_test_duration

def apply_brand_mapping_to_interactions(data: pd.DataFrame, content: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Replaces the externalids of episodes in the usage data by the corresponding brand_externalids or topic_ids, if applicable
    :param data: usage data
    :return: data with replaced values in the column externalids
    """
    logging.info("Applying brand mapping to interactions.")

    if content is None:
        # only load content if not provided as param
        content = get_content_df()
    content = apply_brand_mapping_to_content(content)

    # replace externalids by topic_ids or brand_ids
    data["externalid"] = content.loc[
        :, "mapped_externalid"          # Series with index: externalid, column: mapped_externalid
    ].reindex(
        data.externalid                 # reindex Series, mapping externalid->mapped_externalid (or NaN)
    ).set_axis(
        data.index, inplace=False       # Series now has the same index as data DataFrame
    ).fillna(
        data.externalid                 # fill unknown values (mapped to NaN) with original externalids
    )  # .rename("externalid")          # renaming is not required since we explicitly set it to data["externalid"]
    logging.info(
        "Finished applying brand mapping to interactions. Replaced externalids"
    )
    return data

def postprocess_aggreaget_recos_applying_brand_mapping(predictions, content: pd.DataFrame):
    """

    :param predictions:
    :param content:
    :return:
    """
    # create a mapping externalid -> brand/topic-id
    predictions = [
        (get_brand_or_topic_or_externalid_of_item(item[0], content), item[1])
        for item in predictions
    ]

    # aggregate the scores of the brands in the recos, sort by scores, normalize
    predictions = [(k, sum(item[1] for item in values))
            for k, values in groupby(predictions, key=operator.itemgetter(0))]
    predictions = sorted(predictions, key=operator.itemgetter(1), reverse=True)
    predictions = normalize_scores(predictions)
    return predictions


def extract_brands_from_content(content: pd.DataFrame) -> pd.DataFrame:
    brands: pd.DataFrame = content[content.contenttype == "brand"].copy()
    brands.drop_duplicates(subset="brand_externalid", inplace=True)
    # convert categorical column labels to normal index (otherwise at[] won't work)
    brands.set_index(brands["brand_externalid"].astype(str), drop=False, inplace=True)
    return brands


def extract_topics_from_content(content: pd.DataFrame) -> pd.DataFrame:
    topics: pd.DataFrame = content[content.contenttype == "topic"].copy()
    topics.drop_duplicates(subset="externalid", inplace=True)
    # convert categorical column labels to normal index (otherwise at[] won't work)
    topics.set_index(topics["externalid"].astype(str), drop=False, inplace=True)
    return topics


def apply_brand_mapping_to_content(content: pd.DataFrame) -> pd.DataFrame:
    """
    Saves each item's mapped_externalid (brand_externalid or topic_id, if available, externalid otherwise)
    :param content:
    :return: content with an additional column "mapped_externalid"
    """
    logging.info("Applying brandmapping to content.")

    brands: pd.DataFrame = extract_brands_from_content(content)
    logging.info(f"Brands: {len(brands)}.")
    topics: pd.DataFrame = extract_topics_from_content(content)
    logging.info(f"Topics: {len(topics)}.")

    content["mapped_externalid"] = content.apply(
        lambda x: (
            x.series_index_page_externalid
            if (x.series_index_page_externalid != ""
                and (x.contenttype == "episode"
                     or x.contenttype == "clip")
                and x.series_index_page_externalid in (topics["externalid"]))
            else (
                x.brand_externalid
                if (x.brand_externalid != ""
                    and (x.contenttype == "episode"
                         or x.contenttype == "clip")
                    and x.brand_externalid in (brands["externalid"]))
                else x.externalid
                )
        ),
        axis=1,
    )

    return content


def get_brand_or_topic_or_externalid_of_item(externalid: str, content: pd.DataFrame) -> str:
    """
    Returns an item's mapped_externalid (brand_externalid or topic_id, if available, externalid otherwise)
    :param externalid:
    :param content:
    :return: str - an item's mapped_externalid
    """
    return content.at[externalid, "mapped_externalid"] if externalid in content.externalid else externalid

def load_golden_dataset_for_hyperparameter_tuning(targets: List[str]) -> pd.DataFrame:
    """
    Loads the golden or fixed data set specified within the bucket and S3 Key prefix
    : param targets : Target variant obtained from the specified hyperparams.
    : return: Fixed usage data set for the specified targets.
    """ 

    try:

        # Based on the model variant name (example: sasrec-pctablet, sasrec-tv-mobile-pctablet)
        # The data is loaded from S3 bucket
        if len(targets) == 1:

            #For the specified single target (pc-tablet, tv, or mobile), load the data from the S3 bucket.
            s3_key_prefix = f"{targets[0]}-usage-data.parquet"
            
            logging.info(f"Using the data only for the specified one target - {targets[0]}.")
            logging.info(f"Loading the data from the bucket {MODEL_VALIDATION_DATA_BUCKET} and s3_key_prefix {s3_key_prefix}")

            data : pd.DataFrame = download_dataframe_from_s3(s3_bucket= MODEL_VALIDATION_DATA_BUCKET, s3_key_prefix= s3_key_prefix)
        
        elif len(targets) == 3:
            # S3 prefix hard coded with three targets. The data with all the three varaints (pc-tablet, tv and mobile) loaded.
            s3_key_prefix = f"tv-mobile-pctablet-usage-data.parquet"

            logging.info(f"Loading the data for all three targets tv, mobile and pc-tablet")
            logging.info(f"Loading the data from the bucket {MODEL_VALIDATION_DATA_BUCKET} and s3_key_prefix {s3_key_prefix}")

            data : pd.DataFrame = download_dataframe_from_s3(s3_bucket= MODEL_VALIDATION_DATA_BUCKET, s3_key_prefix= s3_key_prefix)

        else:
            #Check the hyperparameter 'model_variant_name', only one or all the three variants data can be loaded.
            logging.error(f"Please check the number of targets {targets} specified or the name of the hyperparameter 'model_variant_name' ")
            logging.error(f"Only one of the variant data (pc-tablet, mobile, tv) or all the three data variants can be loaded")
            sys.exit(1)
    
    except Exception as e:

        logging.error(f"Error loading the data file from specified bucket {MODEL_VALIDATION_DATA_BUCKET}")
        logging.error(f"Please recheck the specified bucket and the targets for the required data. Exception: {e}")
        sys.exit(1)
    
    return data

def prepare_data_for_training(data :pd.DataFrame, model_desc:dict) -> Tuple[CustomSASRecDataSet, dict, dict]:
        """
        Prepares the data for the model training

        :param data: Dataframe with uid, externalid, and timestamp as features
        :param model_desc: Model description
        Returns: The pre-processed data for the model training, model description along with the mappings
        """
        user_set, item_set = set(data['uid'].unique()), set(data['externalid'].unique())

        logging.info(f"For training with sasrec-model, Total {len(user_set)} unique uids, and unique {len(item_set)} externalids are considered respectively")
        
        #Categorical encodings of the user_ids and item_ids have been already done within preprocess function
        #Drop the duplicate values, and store item mappings in metetadata
        item_lookup = data[["item_id","externalid"]].drop_duplicates()

        item_for_cf_id = dict(item_lookup.to_dict(orient="split")["data"])
        itemid_for_extid = {v:k for k, v in item_for_cf_id.items()}

        data = data.sort_values(by=["user_id", "timestamp"])

        # Convert the u_id values starting with 1 for the Sasrec sampler for training
        unique_values = data["user_id"].unique()
        mappings = {value: i+1 for i, value in enumerate(unique_values)}
        data['user_id'] = data['user_id'].map(mappings)

        #Drop the unnecessary features used.
        data.drop(columns = ["timestamp", "uid", "externalid", "datetime_local"], inplace =True)
        logging.info(f"Sorting the data with timestamp, and dropping the unused features")

        logging.info(f"Train data/sequences grouped by users. Statistics: {data.groupby('user_id').size().describe()}")


        #Complete_data with sub class with users, items count
        logging.info(f"Preparing the data for training")
        train_data = CustomSASRecDataSet(data=data)

        # Create only train sequences, no test, and validation sequences
        logging.info(f"For all user sequences, only training part is considered, no validation and testing sequences generated within the recommenders package")

        sasrec_model_meta_data = {"Sasrec_model_descritption":model_desc,
                                  "item_id_for_extid": itemid_for_extid,
                                  "external_id_for_item_id": item_for_cf_id,
                                  "num_items_training": train_data.itemnum}

        return train_data, sasrec_model_meta_data, itemid_for_extid

def dump_meta_data_model(model:CustomSASRec, meta:dict, output_dir:Path) -> None:
    
    """
    Dumps the model, model weights and metadata
    param model: Trained model
    param meta: Model description with mappingss
    param output_dir: Path within the sagmaker for saving the model, and metadata
    """

    output_file_meta = path.join(output_dir, "meta.pkl")

    with open(output_file_meta, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    model_weights_path = path.join(output_dir, 'sasrec_weights/')
    model_path = path.join(output_dir, 'model_sasrec/')

    # Save weights
    model.save_weights(model_weights_path)
    logging.info(f'Model weights saved for Inference part and saved withinin the directory {model_weights_path}')

    # Save the Model
    tf.saved_model.save(model, export_dir=model_path)
    logging.info(f'Tensorflow Model Saved for the inference path in the directory {model_path}')


def preprocess(
    *,
    site: str,
    hyperparams: Hyperparams,
    ids_n_days=PLAY_MAX_AGE_DAYS,
    metric_logger=None,
) -> pd.DataFrame:
    n_days_offset = 0

    timestamp = time.process_time()

    # If use_fixed_data hyperparams is set to true, load the fixed data set from the specifed S3 bucket.
    if hyperparams.use_fixed_data:

        data: pd.DataFrame = load_golden_dataset_for_hyperparameter_tuning(targets=hyperparams.targets)

    else: 
        # get aggregated clickstream data data for 'n_days' and all targets
        data: pd.DataFrame = get_denoised_data(
            use_ids_data=False,
            use_cl_data=True,
            ids_n_days=ids_n_days,
            cl_frontend=hyperparams.targets,
            cl_n_days=hyperparams.n_days,
            cl_n_days_offset=n_days_offset,
            coverage_threshold=hyperparams.minimum_coverage,
            count_threshold=hyperparams.min_item_interactions,
            user_count_threshold=hyperparams.user_count_threshold,
        )

    metric_logger._log_metric(
        metric_name=f"usage_data_fetching_duration-{site}", value=time.process_time() - timestamp, unit="Seconds"
    )
    metric_logger._log_metric(
        metric_name=f"initial_usage_data_size-{site}", value=len(data)
    )
    
    #Drop the columns which are not being used, datetime_local is further used in pa-base, so retain datetime_local for evaluation
    data.drop(["coverage", "date", "genre"], axis =1, inplace =True)

    #Convert uids, externalids to user_id, and item_id starting with 1, include here for pa-base train, test split
    data["user_id"] = data["uid"].astype("category").cat.codes + 1
    data["item_id"] = data["externalid"].astype("category").cat.codes + 1

    n_samples = len(data)
    logging.info(f"Working with {n_samples} data samples.")

    prep_duration = time.process_time() - timestamp
    logging.info(f"Data preparation took {prep_duration:.4f}s.")

    metric_logger._log_metric(
        metric_name=f"train_usage_data_size-{site}", value=data.shape[0])
    
    metric_logger._log_metric(
        metric_name=f"train_usage_data_preparation_duration-{site}", value=prep_duration, unit="Seconds")
   
    return data

def train_model(site: str, data : CustomSASRecDataSet, hyperparams: Hyperparams,
                metric_logger=None, ml_metric_logger=None) -> CustomSASRec:
    
    """
    Training of the SASREC model, along with hyper parameter tuning when specified.
    """
    
    timestamp = time.process_time()

    #The num_neg_test is reduntant during training, currently we set the value from the maxlen hyperparam
    model = CustomSASRec(item_num = data.itemnum,
                         seq_max_len = hyperparams.maxlen,
                         num_blocks =  hyperparams.num_blocks,
                         embedding_dim = hyperparams.hidden_units,
                         attention_dim = hyperparams.hidden_units,
                         attention_num_heads = hyperparams.num_heads,
                         dropout_rate = hyperparams.dropout_rate,
                         conv_dims = [hyperparams.hidden_units, hyperparams.hidden_units],
                         l2_reg = hyperparams.l2_emb,
                         num_neg_test = hyperparams.maxlen)
    
    # Define the Sampler , also popularity 
    sampler = define_the_sampler(data, hyperparams)
            
    #Include loss trackers with mlflow when evaluating the model.
    #Loss removed from metrics, so, always logs the loss value within the metrics.
    if hyperparams.evaluate_model:

        #Currently the validation, testing for the model within recommenders is ignored.
        logging.info(f"Training the model. No validation or testing done within the recommenders, only training the model")
        model.train(dataset=data, sampler=sampler, hyperparams=hyperparams)

        for epoch, loss in enumerate(model.epochs_loss_tracker, start=1):
            mlflow.log_metric(key="loss", value=loss, step=epoch)

    else:
        #Currently the validation, testing for the model within recommenders is ignored.
        logging.info(f"Training the model. No validation or testing done within the recommenders, only training the model")
        model.train(dataset=data, sampler=sampler, hyperparams=hyperparams)

    gen_duration = time.process_time() - timestamp
    logging.info(f"Model genration took {gen_duration:.4f}s.")

    if hyperparams.evaluate_model:
        ml_metric_logger.log_metric("gen", "duration", gen_duration)

    metric_logger._log_metric(
    metric_name=f"training_duration-{site}", value=gen_duration, unit="Seconds")

    return model
