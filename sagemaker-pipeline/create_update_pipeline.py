# Copyright (c) 2024, ZDF.
"""A CLI to create or update and run pipelines."""

import argparse
import json
import sys
from pipeline import get_pipeline


def main():  # pragma: no cover
    """The main harness that creates or updates and runs the pipeline.

    Creates or updates the pipeline and runs it.
    """
    parser = argparse.ArgumentParser("Creates or updates and runs the pipeline for the pipeline script.")

    parser.add_argument(
        "-default-bucket",
        "--default-bucket",
        dest="default_bucket",
        type=str,
        default=None,
        help="The default S3 Bucket"
    )
    parser.add_argument(
        "-region",
        "--region",
        dest="region",
        type=str,
        help="The default AWS region",
    )
    parser.add_argument(
        "-training-image-uri",
        "--training-image-uri",
        dest="training_image_uri",
        default=None,
        help="Training image URI",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-pipeline-name",
        "--pipeline-name",
        dest="pipeline_name",
        type=str,
        help="The name of the pipeline",
    )
    parser.add_argument(
        "-start_pipeline",
        "--start-pipeline",
        dest="start_pipeline",
        type=bool,
        help="boolean",
    )
    parser.add_argument(
        "-security_groups_for_redis_access",
        "--security_groups_for_redis_access",
        dest="security_groups_for_redis_access",
        type=str,
        help="Security groups which are allowed to connect to Redis",
    )
    parser.add_argument(
        "-subnets_for_redis_access",
        "--subnets_for_redis_access",
        dest="subnets_for_redis_access",
        type=str,
        help="Subnets where the redis server is located",
    )
    parser.add_argument(
        "-s3_base_name",
        "--s3_base_name",
        dest="s3_base_name",
        type=str,
        help="The base name of the S3 bucket for interaction data",
    )

    args = parser.parse_args()
    print(args.subnets_for_redis_access)

    if args.role_arn is None:
        parser.print_help()
        sys.exit(2)

    try:
        pipeline = get_pipeline(
            region=args.region,
            s3_base_name=args.s3_base_name,
            training_image_uri=args.training_image_uri,
            role=args.role_arn,
            default_bucket=args.default_bucket,
            pipeline_name=args.pipeline_name,
            subnets=args.subnets_for_redis_access,
            security_groups=args.security_groups_for_redis_access
        )

        print("###### Creating/updating a SageMaker Pipeline with the following definition:")
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        upsert_response = pipeline.upsert(role_arn=args.role_arn, description=args.description)
        print("\n###### Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)

        if args.start_pipeline:
            execution = pipeline.start()
            print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")
    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
