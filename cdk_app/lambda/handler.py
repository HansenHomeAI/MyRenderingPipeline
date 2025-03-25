import os
import json
import boto3
import uuid

sagemaker = boto3.client("sagemaker")
dynamodb = boto3.resource("dynamodb")
stepfunctions_client = boto3.client("stepfunctions")

def handler(event, context):
    # If invoked by Step Functions, we won't have the same "path" logic:
    # We'll parse "action" from event["action"] if it exists.
    
    # If invoked by API Gateway /start path, keep old logic. Or, unify them.

    # 1) Check if Step Functions passed us "action"
    if "action" in event:
        # Called from Step Functions
        action = event["action"]
        return do_stage_logic(event, action)
    else:
        # 2) Otherwise, fallback to the API Gateway route-based logic
        path = event["requestContext"]["resourcePath"] if "requestContext" in event else None
        method = event["httpMethod"] if "httpMethod" in event else None

        if path == "/stop" and method == "POST":
            return stop_job_logic(event)
        elif path == "/start" and method == "POST":
            return start_job_logic(event)
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid route or method."})
            }


def do_stage_logic(event, action):
        """
        Shared function that starts either Recon or Train job in SageMaker
        depending on 'action' ("RECON" or "TRAIN").
        """
        # Setup
        table_name = os.environ["STATUS_TABLE"]
        role_arn   = os.environ["SAGEMAKER_ROLE_ARN"]
        recon_bucket_name = "gabe-recon-renderingpipeline-bucket"  # adjust if needed
        output_bucket_name = os.environ["OUTPUT_BUCKET"]            # the final output bucket
        table = dynamodb.Table(table_name)
    
        # parse step function input
        input_payload = event.get("input", {})  # from "payload": {"input.$": "$"}
        job_id = input_payload.get("jobId", str(uuid.uuid4()))  # or generate new if none
    
        # parse containerName from input (or default)
        container_name = input_payload.get("containerName", "nerfstudio")
        train_command  = input_payload.get("trainCommand", "")
    
        account_id = "975050048887"
        region     = "us-west-2"
        ecr_uri    = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{container_name}:latest"
    
        # figure out which s3 input & output we want
        if action == "RECON":
            # store partial output to recon bucket
            output_s3_uri = f"s3://{recon_bucket_name}/recon-outputs/"
            job_prefix     = "recon"
        else:
            # final model output in final output bucket
            output_s3_uri = f"s3://{output_bucket_name}/models/"
            job_prefix     = "train"
    
        # always read user data from user-submissions or from recon bucket if second stage
        if action == "RECON":
            # read from user-submissions
            s3_archive_name = input_payload.get("s3ArchiveName", "my-training-data")
            input_s3_uri = f"s3://user-submissions/{s3_archive_name}"
        else:
            # read from recon bucket as input
            # (assuming the recon job dumped results to "s3://gabe-recon-renderingpipeline-bucket/recon-outputs/...") 
            # you might store the exact path in dynamo from the first job,
            # or do a known naming pattern, e.g. "jobId-RECON-Output"
            recon_output_key = f"{job_id}-RECON-output"
            input_s3_uri = f"s3://{recon_bucket_name}/recon-outputs/{recon_output_key}"
    
        training_job_name = f"{job_prefix}-job-{job_id}"
    
        # start SageMaker job
        sagemaker.create_training_job(
            TrainingJobName=training_job_name,
            AlgorithmSpecification={
                "TrainingImage": ecr_uri,
                "TrainingInputMode": "File",
                "ContainerEntrypoint": ["/bin/bash","-c",train_command]
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
            OutputDataConfig={"S3OutputPath": output_s3_uri},
            ResourceConfig={
                "InstanceType": "ml.p3.2xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
        )
    
        # store or update job status in DynamoDB
        table.put_item(
            Item={
                "jobId": job_id,
                "stage": action,
                "status": "IN_PROGRESS",
                "sageMakerJobName": training_job_name,
                "outputBucket": output_s3_uri,
            }
        )
    
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"{action} stage job started",
                "jobId": job_id,
                "containerUsed": ecr_uri,
                "inputData": input_s3_uri,
                "trainingJobName": training_job_name
            })
        }
    
    def start_job_logic(event):
    ...
    # parse user input
    recon_container = body.get("reconContainer", "colmap")
    train_container = body.get("trainContainer", "nerfstudio")

    job_id = str(uuid.uuid4())

    # Compose input for step function
    sfn_input = {
        "jobId": job_id,
        "reconContainer": recon_container,
        "trainContainer": train_container,
        # etc...
    }

    # start step function
    response = stepfunctions_client.start_execution(
        stateMachineArn="arn-of-your-state-machine",
        input=json.dumps(sfn_input)
    )

    # store job status in DB, etc.
    ...
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Step Functions pipeline started",
            "jobId": job_id,
            "executionArn": response["executionArn"]
        })
    }

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
