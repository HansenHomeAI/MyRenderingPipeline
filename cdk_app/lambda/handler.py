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
    if "action" in event:
        # Called from Step Functions
        action = event["action"]
        return do_stage_logic(event, action)
    else:
        # Otherwise, fallback to the API Gateway route-based logic
        path = event.get("requestContext", {}).get("resourcePath")
        method = event.get("httpMethod")
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
    if action == "UPDATE_TOKEN":
        table_name = os.environ["STATUS_TABLE"]
        table = dynamodb.Table(table_name)
        # Expecting jobId and taskToken in the event
        job_id = event.get("jobId")
        task_token = event.get("taskToken")
        if job_id and task_token:
            table.update_item(
                Key={"jobId": job_id},
                UpdateExpression="SET taskToken = :token",
                ExpressionAttributeValues={":token": task_token}
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Task token updated", "jobId": job_id})
            }
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing jobId or taskToken"})
            }
    # Otherwise, proceed with the regular RECON/ TRAIN logic:
    table_name = os.environ["STATUS_TABLE"]
    role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
    recon_bucket_name = "gabe-recon-renderingpipeline-bucket"
    output_bucket_name = os.environ["OUTPUT_BUCKET"]
    table = dynamodb.Table(table_name)
    
    input_payload = event.get("input", {})
    job_id = input_payload.get("jobId", str(uuid.uuid4()))
    container_name = input_payload.get("containerName", "nerfstudio")
    train_command = input_payload.get("trainCommand", "")
    
    account_id = "975050048887"
    region = "us-west-2"
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{container_name}:latest"
    
    if action == "RECON":
        output_s3_uri = f"s3://{recon_bucket_name}/recon-outputs/"
        job_prefix = "recon"
        s3_archive_name = input_payload.get("s3ArchiveName", "my-training-data")
        input_s3_uri = f"s3://user-submissions/{s3_archive_name}"
    else:
        output_s3_uri = f"s3://{output_bucket_name}/models/"
        job_prefix = "train"
        recon_output_key = f"{job_id}-RECON-output"
        input_s3_uri = f"s3://{recon_bucket_name}/recon-outputs/{recon_output_key}"
    
    training_job_name = f"{job_prefix}-job-{job_id}"
    
    sagemaker.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            "TrainingImage": ecr_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["/bin/bash", "-c", train_command]
        },
        RoleArn=role_arn,
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_s3_uri,
                    "S3DataDistributionType": "FullyReplicated"
                }
            }
        }],
        OutputDataConfig={"S3OutputPath": output_s3_uri},
        ResourceConfig={
            "InstanceType": "ml.p3.2xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 50
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600}
    )
    
    table.put_item(
        Item={
            "jobId": job_id,
            "stage": action,
            "status": "IN_PROGRESS",
            "sageMakerJobName": training_job_name,
            "outputBucket": output_s3_uri
        }
    )
    
    return {
        "statusCode": 200,
        "message": f"{action} stage job started",
        "jobId": job_id,
        "containerUsed": ecr_uri,
        "inputData": input_s3_uri,
        "trainingJobName": training_job_name
    }



def start_job_logic(event):
    # Parse the incoming event body from API Gateway
    body = {}
    if "body" in event:
        try:
            body = json.loads(event["body"])
        except Exception as e:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid JSON body", "detail": str(e)})
            }
    else:
        body = event

    # Extract parameters from the request
    recon_container = body.get("reconContainer", "colmap")
    train_container = body.get("trainContainer", "nerfstudio")
    s3_archive_name = body.get("s3ArchiveName", "my-training-data")
    train_command = body.get("trainCommand", "")
    
    # Generate a new job ID
    job_id = str(uuid.uuid4())
    
    # Compose input for the Step Functions state machine execution
    sfn_input = {
        "jobId": job_id,
        "reconContainer": recon_container,
        "trainContainer": train_container,
        "s3ArchiveName": s3_archive_name,
        "trainCommand": train_command
    }
    
    # Start execution of the state machine. Replace the placeholder ARN with your actual Step Functions ARN.
    response = stepfunctions_client.start_execution(
        stateMachineArn="arn:aws:states:us-west-2:975050048887:stateMachine:RenderingPipelineStateMachine39265931-vs5WJJdof6SW",
        input=json.dumps(sfn_input)
    )
    
    # Optionally, store the job status in DynamoDB
    table_name = os.environ["STATUS_TABLE"]
    table = dynamodb.Table(table_name)
    table.put_item(
        Item={
            "jobId": job_id,
            "status": "STEP_FUNCTION_STARTED",
            "reconContainer": recon_container,
            "trainContainer": train_container,
            "executionArn": response["executionArn"]
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
            "message": "Step Functions pipeline started",
            "jobId": job_id,
            "executionArn": response["executionArn"]
        })
    }

def stop_job_logic(event):
    table_name = os.environ["STATUS_TABLE"]
    table = dynamodb.Table(table_name)
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
    response = table.get_item(Key={"jobId": job_id})
    item = response.get("Item", {})
    training_job_name = item.get("sageMakerJobName", None)
    if not training_job_name:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No matching job found in DynamoDB"})
        }
    try:
        sagemaker.stop_training_job(TrainingJobName=training_job_name)
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
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
