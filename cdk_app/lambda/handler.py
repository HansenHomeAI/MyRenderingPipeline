import os
import json
import boto3
import uuid

sagemaker = boto3.client("sagemaker")
dynamodb = boto3.resource("dynamodb")

def handler(event, context):
    # 1) Figure out which path and method were called from API Gateway
    path = event["requestContext"]["resourcePath"] if "requestContext" in event else None
    method = event["httpMethod"] if "httpMethod" in event else None

    # 2) If it's /stop and POST, call stop logic
    if path == "/stop" and method == "POST":
        return stop_job_logic(event)

    # 3) If it's /start and POST, call start logic
    elif path == "/start" and method == "POST":
        return start_job_logic(event)

    # 4) Otherwise, return an error
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid route or method."})
        }

def start_job_logic(event):
    # Grab environment variables
    output_bucket = os.environ["OUTPUT_BUCKET"]
    table_name = os.environ["STATUS_TABLE"]
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")

    table = dynamodb.Table(table_name)

    # Generate a random jobId
    job_id = str(uuid.uuid4())

    # Parse the event for input
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
    train_command = body.get("trainCommand", "")

    account_id = "975050048887"  # adjust if needed
    region = "us-west-2"
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{container_name}:latest"

    input_s3_uri = f"s3://user-submissions/{s3_archive_name}"
    training_job_name = f"nerf-training-{job_id}"

    # Create the SageMaker training job
    sagemaker.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            "TrainingImage": ecr_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": [
                "/bin/bash",
                "-c",
                train_command
            ]
        },
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
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )

    # Store job status in DynamoDB
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

def stop_job_logic(event):
    # Grab environment variables
    table_name = os.environ["STATUS_TABLE"]
    table = dynamodb.Table(table_name)

    # Parse the incoming request body for "jobId"
    body = {}
    if "body" in event:
        try:
            body = json.loads(event["body"])
        except:
            pass
    else:
        body = event

    job_id = body.get("jobId", None)
    if not job_id:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No jobId provided"})
        }

    # Retrieve the training job name from DynamoDB
    response = table.get_item(Key={"jobId": job_id})
    item = response.get("Item", {})
    training_job_name = item.get("sageMakerJobName", None)
    if not training_job_name:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No matching job found in DynamoDB"})
        }

    # Attempt to stop the SageMaker job
    try:
        sagemaker.stop_training_job(TrainingJobName=training_job_name)
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    # Update DynamoDB status (you can call it "STOPPING")
    table.update_item(
        Key={"jobId": job_id},
        UpdateExpression="SET #s = :val",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":val": "STOPPING"}
    )

    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*"
        },
        "body": json.dumps({
            "message": "Stopping job",
            "jobId": job_id
        })
    }
