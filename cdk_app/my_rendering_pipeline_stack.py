import os
import json
import boto3
import uuid

sagemaker = boto3.client("sagemaker", region_name="us-west-2")
dynamodb = boto3.resource("dynamodb", region_name="us-west-2")

def handler(event, context):
    """
    Starts a SageMaker training job with user-supplied S3 path (archiveName) 
    and containerName (e.g., nerfstudio).
    """

    # Grab environment variables
    output_bucket = os.environ["OUTPUT_BUCKET"]
    table_name = os.environ["STATUS_TABLE"]
    table = dynamodb.Table(table_name)

    # Generate a random jobId
    job_id = str(uuid.uuid4())

    # Parse incoming event data
    # If the event is from API Gateway, "event['body']" typically holds the POST payload as a string.
    # Otherwise, if it's invoked directly, "event" itself might be the JSON.
    body = {}
    if "body" in event:
        try:
            body = json.loads(event["body"])
        except:
            pass
    else:
        body = event

    # Extract parameters
    s3_archive_name = body.get("s3ArchiveName", "my-training-data")  # Default if none provided
    container_name = body.get("containerName", "nerfstudio")         # Default if none provided

    # Build the final ECR URI
    # Replace these with your actual AWS account/region if needed
    account_id = "975050048887"
    region = "us-west-2"
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{container_name}:latest"

    # Input S3 location
    # We'll assume your entire file/folder is in: s3://user-submissions/<archiveName>
    input_s3_uri = f"s3://user-submissions/{s3_archive_name}"

    # Construct a unique training job name
    training_job_name = f"nerf-training-{job_id}"

    # Create the SageMaker training job
    response = sagemaker.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            "TrainingImage": ecr_uri,
            "TrainingInputMode": "File",
        },
        # For production, you should provide a properly configured Role ARN here,
        # or store it in an environment variable. This simplistic approach attempts
        # to transform the Lambda's ARN into a role ARN.
        RoleArn=context.invoked_function_arn.replace(":function:", ":role/"),
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_s3_uri,
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

    # Store job status in DynamoDB, including the actual SageMaker job name
    table.put_item(
        Item={
            "jobId": job_id,
            "status": "IN_PROGRESS",
            "sageMakerJobName": training_job_name
        }
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "SageMaker job started",
            "jobId": job_id,
            "containerUsed": ecr_uri,
            "inputData": input_s3_uri,
            "trainingJobName": training_job_name
        })
    }

