# Copyright (c) 2024, ZDF.
""" Workflow pipeline script for the sasrec recommender pipeline.

                             . -RegisterModel
                            .
    Preprocess -> Train -> .
                            .
                             . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import datetime
import os

import boto3
import sagemaker
import sagemaker.session
from botocore.exceptions import ClientError
from sagemaker.estimator import Estimator, TrainingInput
from sagemaker.processing import ProcessingOutput, ScriptProcessor
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterString, ParameterBoolean
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)  # noqa
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def create_bucket_delete_policy(region, bucket_name, model_name):
    boto_session = boto3.Session(region_name=region)  # noqa
    s3_client = boto_session.client("s3")

    rule_for_current_model = {
        'Expiration': {
            'Days': 90,
        },
        'Status': 'Enabled',
        'ID': model_name,
        'Filter': {
            'Prefix': f'{model_name}/'
        }
    }

    #Create Lifecycle configuration if it not exists
    try:
        s3_client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
    except ClientError as e:
        print('No Lifecycle Configuration. Creating it...')
        response = s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration={
                'Rules': [rule_for_current_model],
            },
        )
    # Read existing configs and only updates the one belonging to the current model
    policy = s3_client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
    rules = [rule for rule in policy['Rules'] if rule['ID'] != model_name]
    rules.append(rule_for_current_model)

    s3_client.put_bucket_lifecycle_configuration(Bucket=bucket_name, LifecycleConfiguration={
        'Rules': rules,
    })


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)  # noqa

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def _get_preprocessing_step(
    model_variant_name: ParameterString,
    hyperparams: ParameterString,
    filter_params: ParameterString,
    train_with_brand_mapping: ParameterString,
    site: ParameterString,
    pipeline_name: str,
    role: str,
    s3_base_name: str,
    sagemaker_session,
    training_image_uri: str,
    preprocessing_instance_type: ParameterString,
):
    """pipeline step for data preprocessing"""
    processing_base_job_name = f"{pipeline_name}/SasrecPreprocess-{timestamp}"

    script_preprocess = ScriptProcessor(
        image_uri=training_image_uri,
        command=["python3", "preprocess"],
        instance_type=preprocessing_instance_type,
        instance_count=1,
        base_job_name=processing_base_job_name,
        sagemaker_session=sagemaker_session,
        role=role,
        env={
            "SITE": site,
            "S3_BASENAME": s3_base_name,
            "model_variant_name": model_variant_name,
            "hyperparams": hyperparams,
            "filter_params": filter_params,
            "train_with_brand_mapping": train_with_brand_mapping,
        },
    )
    # TODO property file for quality gate
    # data_preprocessing_report = PropertyFile(
    #     name="SasrecPreprocessingReport",
    #     output_name="preprocessing_report",
    #     path="preprocessing_report.json",
    # )
    step_preprocess = ProcessingStep(
        name="Preprocess-Sasrec-Data",
        processor=script_preprocess,
        outputs=[
            # TODO property file for quality gate
            # ProcessingOutput(
            #     output_name="preprocessing_report",
            #     source="/opt/ml/processing/preprocessing_report",
            #     destination=Join(  # noqa
            #         # manually append PIPELINE_EXECUTION_ID
            #         #   because this is not automatically done for file inputs (code) to allow for caching
            #         on="/",
            #         values=[
            #             "s3:/",
            #             sagemaker_session.default_bucket(),
            #             processing_base_job_name,
            #             ExecutionVariables.PIPELINE_EXECUTION_ID,
            #             "preprocessing_report",
            #         ],
            #     ),
            # ),
            ProcessingOutput(
                output_name="preprocessed",
                source="/opt/ml/processing/preprocessed",
                destination=Join(
                    # manually append PIPELINE_EXECUTION_ID
                    #   because this is not automatically done for file inputs (code) to allow for caching
                    on="/",
                    values=[
                        "s3:/",
                        sagemaker_session.default_bucket(),
                        processing_base_job_name,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        "preprocessed",
                    ],
                ),
            ),
        ],
        # sagemaker sdk expects a file here, although we set the real entrypoint in ScriptProcessor command
        code=os.path.join(BASE_DIR, "dummy_code_file.py"),
        # TODO property file for quality gate
        # property_files=[data_preprocessing_report],
    )
    return step_preprocess


def _get_training_step(
    model_variant_name: ParameterString,
    hyperparams: ParameterString,
    filter_params: ParameterString,
    train_with_brand_mapping: ParameterString,
    site: ParameterString,
    pipeline_name: str,
    role: str,
    s3_base_name: str,
    step_preprocess: ProcessingStep,
    sagemaker_session,
    security_groups: str,
    subnets: str,
    training_image_uri: str,
    training_instance_type: ParameterString,
):
    """pipeline step for model training"""
    model_path = f"s3://{sagemaker_session.default_bucket()}/{pipeline_name}/SasrecTrain-{timestamp}"
    sasrec_estimator = Estimator(
        subnets=subnets.split(",") if len(subnets) > 0 else None,
        security_group_ids=security_groups.split(",") if len(security_groups) > 0 else None,
        image_uri=training_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{pipeline_name}/SasrecTrain-{timestamp}",
        sagemaker_session=sagemaker_session,
        role=role,
        environment={
            "SITE": site,
            "S3_BASENAME": s3_base_name,
        },
    )
    sasrec_estimator.set_hyperparameters(
        SAGEMAKER_PROGRAM="train",
        hyperparams=hyperparams,
        filter_params=filter_params,
        train_with_brand_mapping=train_with_brand_mapping,
        model_variant_name=model_variant_name,
        SITE=site
    )
    step_train = TrainingStep(
        name=f"Train-Sasrec-Model",
        estimator=sasrec_estimator,
        inputs={
            "preprocessed": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "preprocessed"
                ].S3Output.S3Uri,
                input_mode="FastFile",  # mount read-only from S3
            ),
        },
    )
    return step_train


def get_pipeline(
        region,
        training_image_uri,
        s3_base_name,
        role=None,
        default_bucket=None,
        pipeline_name="sasrec_model",
        subnets="",
        security_groups="",
):
    """Gets a SageMaker ML Pipeline instance working with on ZDF's clickstream data.

    Args:
        region: AWS region to create and run the pipeline.
        training_image_uri: Docker image used for processing, training and evaluation
        role: IAM role to create and run steps and pipeline.
        s3_base_name: base name of the S3 bucket for interaction data
        default_bucket: the bucket to use for storing the artifacts
        pipeline_name: used as pipeline name and in job prefixes
        subnets: subnets for redis access
        security_groups: security groups for redis access

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    create_bucket_delete_policy(region, sagemaker_session.default_bucket(), pipeline_name)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    # ##### #####  parameters for pipeline execution  ##### #####
    # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    parameters = []

    # parameters for pipeline execution
    model_approval_status = ParameterString(
        # This input parameter is added on purpose in case a Manual Approval needs to be added to the pipeline.
        #     In this case, default value should be "PendingApproval"
        name="ModelApprovalStatus",
        default_value="Approved",
    )
    parameters.append(model_approval_status)

    preprocessing_instance_type = ParameterString(
        # TODO switch to ml.r5.2xlarge when quota is increased (currently 0 on DEV)
        name="preprocessing_instance_type", default_value="ml.m5.2xlarge"
    )
    parameters.append(preprocessing_instance_type)

    training_instance_type = ParameterString(
        name="training_instance_type", default_value="ml.g4dn.xlarge"
    )
    parameters.append(training_instance_type)

    endpoint_instance_type = ParameterString(
        name="endpoint_instance_type", default_value="ml.m5.large"
    )
    parameters.append(endpoint_instance_type)

    site = ParameterString(
        name="SITE",
        default_value="dev"
    )
    parameters.append(site)

    hyperparams = ParameterString(
        name="hyperparams",
        default_value='{"n_days" : 2, "use_gpu" : "true"}',
    )
    parameters.append(hyperparams)

    filter_params = ParameterString(
        name="filter_params",
        default_value='{}',
    )
    parameters.append(filter_params)

    train_with_brand_mapping = ParameterString(
        # this has to be ParameterString instead of ParameterBoolean because the ProcessingJob only handles strings
        name="train_with_brand_mapping",
        default_value="false",  # has to be lowercase for json parsing
        enum_values=["false", "true"]
    )
    parameters.append(train_with_brand_mapping)

    model_variant_name = ParameterString(
        name="model_variant_name",
        default_value="sasrec-pctablet"
    )
    parameters.append(model_variant_name)

    # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    # ##### ##### #####     pipeline steps      ##### ##### #####
    # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    steps = []

    step_preprocess = _get_preprocessing_step(
        model_variant_name=model_variant_name,
        hyperparams=hyperparams,
        filter_params=filter_params,
        train_with_brand_mapping=train_with_brand_mapping,
        site=site,
        pipeline_name=pipeline_name,
        role=role,
        s3_base_name=s3_base_name,
        sagemaker_session=sagemaker_session,
        training_image_uri=training_image_uri,
        preprocessing_instance_type=preprocessing_instance_type,
    )
    steps.append(step_preprocess)

    step_train = _get_training_step(
        model_variant_name=model_variant_name,
        hyperparams=hyperparams,
        filter_params=filter_params,
        train_with_brand_mapping=train_with_brand_mapping,
        site=site,
        pipeline_name=pipeline_name,
        role=role,
        s3_base_name=s3_base_name,
        step_preprocess=step_preprocess,
        sagemaker_session=sagemaker_session,
        security_groups=security_groups,
        subnets=subnets,
        training_image_uri=training_image_uri,
        training_instance_type=training_instance_type,
    )
    steps.append(step_train)

    step_register = RegisterModel(
        name=f"Register-Sasrec-Model",
        estimator=step_train.estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", endpoint_instance_type],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_variant_name,
        approval_status=model_approval_status
    )
    steps.append(step_register)

    # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    # ##### ##### #####     pipeline instance   ##### ##### #####
    # ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=parameters,
        steps=steps,
        sagemaker_session=sagemaker_session,
    )
    return pipeline
