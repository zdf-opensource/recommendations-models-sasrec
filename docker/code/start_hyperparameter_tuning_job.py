# Copyright (c) 2024, ZDF.
"""Hyperparameter tuning.

This module starts a hyperparameter tuning job in Sagemaker for a single provided model and its parameter ranges.

"""

import argparse
import datetime
import json
import pickle
from subprocess import check_output

import boto3
import sagemaker as sm
# The default role name to use for the preprocessing job
from sagemaker import HyperparameterTuningJobAnalytics, LocalSession, Session
from sagemaker.parameter import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
)
from sagemaker.tuner import HyperparameterTuner, WarmStartConfig, WarmStartTypes
from sagemaker.workflow.parameters import ParameterString

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
TUNING_JOB_BASENAME = "tune-"
# Default region
DEFAULT_REGION = "eu-central-1"
# The default role name to use for the preprocessing job
ROLE_NAME = "reco-training-execution-role"

# Default Hyperparameters to pass to the job
DEFAULT_HYPERPARAMS = {
    "log_level": "INFO",
    "SITE": "dev",
    "S3_USE_V2": "True",
    "use_gpu": "True",
    "evaluate_model": "True",
    "top_nitems": 5
}


def start_hyperparameter_tuning_job(
    extra_hyperparams: dict = None,
    log_level: str = "DEBUG",
    image: str = None,
    instance: str = None,
    name: str = "sasrec-model",
    region: str = DEFAULT_REGION,
    role: str = ROLE_NAME,
    hyperparameter_ranges=None,
    objective_metric: str = None,
    strategy: str = None,
    optimize="Maximize",
    max_jobs: int = 1,
    warm_start_name=None,
    site="dev",
    local=False,
):
    """
    Starts a Sagemaker Hyperparameter Tuning job

    Args:
        log_level (str): Set the log level for the job. Can be one of "DEBUG", "INFO", "WARN", "ERROR"
        image (str): The imange name including tag for running the job,
        instance (str): The instance type to use for the job, e.g. "ml.m4.2xlarge"
        model (str): The model to tune parameters of with this tuning job
        name (str): The job basename. Should start with reco-training. If let blank, will be set to the current git branch
        region (str): The AWS region
        role (str): The exectuion role of the job
        volume (str): The volume size (in GB) for the job.
        variant (str): the image variant to use, e.g "cpu"
    """

    account = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account}:role/{role}"

    if instance is None:
        instance = "ml.g4dn.xlarge"

    # if the image name was provided as an argument, use that to set the job name and full image URI
    if image is not None:
        fullimage = f"{account}.dkr.ecr.{region}.amazonaws.com/{image}"
        job_basename = TUNING_JOB_BASENAME

    # Otherwise set the names to the currently active git branch
    else:
        branch = (
            check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode("utf-8")
            .lower()
            .strip()
        )

    # only override the job name if it was explicitly provided as an argument to this function
    if name is not None:
        job_basename = name
    
    # Determine the value for top_nitems, the default to_p_five_is five, else the number stated by the user is taken
    top_nitems_value = int(extra_hyperparams.get("top_nitems", 5))

    #All metric values must be compiled with the metric names defined in the logs mentioned in offline_evalaution function
    #Using the genric value of top_nitems with default value or specified from the user.
    metric_definitions = [
        {
            "Name": "train:loss",  # The loss/epoch of Sasrec is tracked inside the model module itself
            "Regex": r": loss (\S+)",
        },
        {"Name": "train:epoch", "Regex": r"Epoch (\S+):"},
        {"Name": f"test:ndcg_at_{top_nitems_value}", "Regex": fr"test:ndcg_at_{top_nitems_value}=(\S+)"},
        {"Name": f"test:hit_at_{top_nitems_value}", "Regex": fr"test:hit_at_{top_nitems_value}=(\S+)"},
        {"Name": "test:hit_at_1", "Regex": r"test:hit_at_1=(\S+)"},
        {"Name": "test:mrr", "Regex": r"test:mrr=(\S+)"},
        {"Name": "test:pop_q_1", "Regex": r"test:pop_q_1=(\S+)"}
    ]

    objective_metric_name = f"{objective_metric}"

    hyperparams = DEFAULT_HYPERPARAMS

    # update the job hyperparams with values supplied by the user
    for k, v in extra_hyperparams.items():
        hyperparams[k] = v

    print("============================================================")
    print(f"Job Basename: {name}")
    print(f"Image: {fullimage}")
    print(f"Role: {role_arn}")
    print(f"Instance type: {instance}")
    print("Hyperparams:")
    for k, v in hyperparams.items():
        print(f"\t{k}: {v}")
    print(f"Objective Metric: {objective_metric_name}")
    print("Hyperparameter Ranges:")
    for k, v in hyperparameter_ranges.items():
        if isinstance(v, CategoricalParameter):
            print(f"\t{k}: {v.values}")
        else:
            print(f"\t{k}: {v.min_value} - {v.max_value}")
    print("============================================================")

    if local:
        sess = LocalSession()
    else:
        sess = sm.Session()

    # setup the job
    model_path = f"s3://{sess.default_bucket()}/{name}/SasrecTrain-{timestamp}"
    estimator = sm.estimator.Estimator(
        subnets=None,
        security_group_ids=None,
        image_uri=fullimage,
        role=role_arn,
        instance_type=instance,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{name}-SasrecTrain-{timestamp}",
        sagemaker_session=sess,
        environment={"SITE": site},
    )

    hyperparams = ParameterString(
        name="hyperparams",
        default_value=json.dumps(hyperparams),
    )

    filter_params = ParameterString(
        name="filter_params",
        default_value="{}",
    )

    train_with_brand_mapping = ParameterString(
        name="train_with_brand_mapping",
        default_value="false",
    )

    model_variant_name = ParameterString(
        name="model_variant_name", default_value="sasrec-pctablet"
    )
    estimator.set_hyperparameters(
        SAGEMAKER_PROGRAM="train",
        hyperparams=hyperparams,
        filter_params=filter_params,
        train_with_brand_mapping=train_with_brand_mapping,
        model_variant_name=model_variant_name,
        SITE=site,
    )

    warm_start_config = None
    if warm_start_name:
        warm_start_config = WarmStartConfig(
            warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
            parents={warm_start_name},
        )

    #  should we minimize or maximize the objective metric?
    if "loss" in objective_metric_name:
        optimize = "Minimize"

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name=objective_metric_name,
        objective_type=optimize,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=metric_definitions,
        max_jobs=max_jobs,
        max_parallel_jobs=1,
        strategy=strategy,
        warm_start_config=warm_start_config,
        early_stopping_type="Auto",
        base_tuning_job_name=job_basename,
    )

    tuner.fit()

    hp_job = HyperparameterTuningJobAnalytics(tuner.latest_tuning_job.job_name)
    df = hp_job.dataframe()
    output_file_model = f"{job_basename}_analytics.pkl"
    pickle.dump(df, open(output_file_model, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def main():
    """
    provides a simple command line interface and executes the training job
    """
    parser = argparse.ArgumentParser(description="Launch a SageMaker training job")

    parser.add_argument(
        "--hyperparam",
        help="Add a hyperparam to the job in the form 'key=val', can be repeated multiple times",
        action="append",
    )
    parser.add_argument(
        "--range_cat",
        help="Add a list of values for a categorical hyperparameter, can be repeated multiple times. Use form: name=value,value,value",
        action="append",
    )
    parser.add_argument(
        "--range_int",
        help="Add a range of int values for a discrete hyperparameter, can be repeated multiple times. Use form: name=min_value,max_value",
        action="append",
    )
    parser.add_argument(
        "--range_float",
        help="Add a range of float values for a continuous hyperparameter, can be repeated multiple times. Use form: name=min_value,max_value",
        action="append",
    )
    parser.add_argument(
        "--loglevel",
        help="The log level to use",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="DEBUG",
    )
    parser.add_argument(
        "--image",
        help="image name including tag to use for the job,"
        " e.g. reco-training-dev:peraut-XXX-cpu",
    )
    parser.add_argument(
        "--instance",
        help="The instance type used for the job, e.g. ml.m5.xlarge",
    )
    parser.add_argument(
        "--warm_start_name",
        type=str,
        help="Name of a previous Hyperparameterjob to continue using as a warm start.",
    )

    parser.add_argument(
        "--objective_metric",
        help="Metric name for the tuning job to optimize for in the format: validation:metic or test:metric",
    )
    parser.add_argument(
        "--strategy",
        help="Search strategy to be used during hyperparameter tuning.",
        default="Bayesian",
        choices=[
            "Random",
            "Bayesian",
        ],
    )
    parser.add_argument(
        "--name", help="preprocessing job basename", default="sasrec-model"
    )
    parser.add_argument(
        "--SITE",
        help="SITE",
    )
    parser.add_argument(
        "--region",
        help="AWS Region",
        default=DEFAULT_REGION,
    )
    parser.add_argument(
        "--role",
        help=f"The IAM role name to use for the job, current default role is: {ROLE_NAME}",
        default=ROLE_NAME,
    )

    parser.add_argument(
        "--optimize",
        help=f"Should the objective_metric be minimized or maximized.",
        default="Maximize",
        choices=["Maximize", "Minimize"],
    )

    parser.add_argument(
        "--max_jobs",
        type=int,
        help="Max. number of training jobs to perform.",
    )
    args = parser.parse_args()

    hyperparams = {}
    if args.hyperparam is not None:
        for pair in args.hyperparam:
            k, v = pair.split("=")
            hyperparams[k] = v

    hyperparameter_ranges = {}
    if args.range_cat is not None:
        for pair in args.range_cat:
            k, v = pair.split("=")
            hyperparameter_ranges[k] = CategoricalParameter(v.split(","))

    if args.range_int is not None:
        for pair in args.range_int:
            k, v = pair.split("=")
            min, max = v.split(",")
            hyperparameter_ranges[k] = IntegerParameter(min, max)

    if args.range_float is not None:
        for pair in args.range_float:
            k, v = pair.split("=")
            min, max = v.split(",")
            hyperparameter_ranges[k] = ContinuousParameter(min, max)

    start_hyperparameter_tuning_job(
        extra_hyperparams=hyperparams,
        log_level=args.loglevel,
        image=args.image,
        instance=args.instance,
        strategy=args.strategy,
        name=args.name,
        region=args.region,
        role=args.role,
        objective_metric=args.objective_metric,
        hyperparameter_ranges=hyperparameter_ranges,
        optimize=args.optimize,
        max_jobs=args.max_jobs,
        warm_start_name=args.warm_start_name,
    )


if __name__ == "__main__":
    main()
