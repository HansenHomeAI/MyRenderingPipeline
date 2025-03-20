import os
import json
import boto3
import uuid

# Initialize the SageMaker and DynamoDB clients once at module level
# so they can be reused by subsequent invocations
sagemaker = boto3.client("sagemaker")
dynamodb = boto3.resource("dynamodb")

def handler(event, context):
    """
    Starts a SageMaker training job with user-supplied S3 path (archiveName) 
    and containerName (e.g., nerfstudio).
    """

    # Grab environment variables
    output_bucket = os.environ["OUTPUT_BUCKET"]
    table_name = os.environ["STATUS_TABLE"]
    
    # NEW: read the SageMaker execution role ARN from environment
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")

    table = dynamodb.Table(table_name)

    # Generate a random jobId
    job_id = str(uuid.uuid4())

    # Parse incoming event data
    body = {}
    if "body" in event:
        try:
            body = json.loads(event["body"])
        except:
            pass
    else:
        body = event

    # Extract parameters
    s3_archive_name = body.get("s3ArchiveName", "my-training-data")
    container_name = body.get("containerName", "nerfstudio")

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
        # Use the role ARN from environment here
        RoleArn=role_arn,
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
            "InstanceType": "ml.p3.2xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600}
    )

    # Store job information in DynamoDB
    table.put_item(
        Item={
            "jobId": job_id,
            "status": "IN_PROGRESS",
            "sageMakerJobName": training_job_name
        }
    )

    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*"
        },
        "body": json.dumps({
            "message": "SageMaker job started",
            "jobId": job_id,
            "containerUsed": ecr_uri,
            "inputData": input_s3_uri,
            "trainingJobName": training_job_name
        })
    }
