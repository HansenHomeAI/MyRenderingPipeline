# cdk_app/lambda/handler.py

import os
import json
import boto3
import uuid

sagemaker = boto3.client("sagemaker")
dynamodb = boto3.resource("dynamodb")

def handler(event, context):
    # Grab environment variables
    output_bucket = os.environ["OUTPUT_BUCKET"]
    table_name = os.environ["STATUS_TABLE"]
    table = dynamodb.Table(table_name)

    # Generate a random jobId
    job_id = str(uuid.uuid4())

    # TODO: Extract any parameters from `event` for training
    # Example: hyperparams, S3 input path, etc.

    # This is a minimal example for SageMaker's 'create_training_job'
    # You must customize it to your container, instance type, hyperparams, etc.
    response = sagemaker.create_training_job(
        TrainingJobName=f"nerf-training-{job_id}",
        AlgorithmSpecification={
            "TrainingImage": "123456789012.dkr.ecr.us-west-2.amazonaws.com/nerfstudio:latest",
            "TrainingInputMode": "File",
        },
        RoleArn=context.invoked_function_arn.replace(":function:", ":role/"),  # simplistic approach
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://user-submissions/my-training-data/",  # or from event
                    "S3DataDistributionType": "FullyReplicated",
                }
            }
        }],
        OutputDataConfig={
            "S3OutputPath": f"s3://{output_bucket}/models/"
        },
        ResourceConfig={
            "InstanceType": "ml.p3.2xlarge",  # example GPU instance
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600}
    )

    # Store job status in DynamoDB
    table.put_item(
        Item={
            "jobId": job_id,
            "status": "IN_PROGRESS"
        }
    )

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "SageMaker job started", "jobId": job_id})
    }
