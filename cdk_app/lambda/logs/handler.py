import os
import json
import boto3
import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

dynamodb = boto3.resource("dynamodb")
sagemaker = boto3.client("sagemaker")
stepfunctions = boto3.client('stepfunctions')  # added for callbacks

def handler(event, context):
    # Check if this is a DynamoDB Stream event (triggered by table updates)
    if "Records" in event:
        # Process each record in the batch
        for record in event["Records"]:
            if record["eventName"] in ["INSERT", "MODIFY"]:
                new_image = record["dynamodb"].get("NewImage", {})
                job_id = new_image.get("jobId", {}).get("S")
                status = new_image.get("status", {}).get("S")
                # Assume we've stored a taskToken in the DB when the Step Functions wait started.
                task_token = new_image.get("taskToken", {}).get("S")
                
                # Check if the update indicates that the Recon job is complete
                if status == "COMPLETED_RECON" and task_token:
                    try:
                        stepfunctions.send_task_success(
                            taskToken=task_token,
                            output=json.dumps({"status": status, "jobId": job_id})
                        )
                        print(f"Callback sent for job {job_id} with token {task_token}")
                    except Exception as e:
                        print(f"Error sending callback for job {job_id}: {str(e)}")
        # Return a simple acknowledgment for stream processing
        return {"status": "stream processed"}
    
    # Otherwise, assume this is an API Gateway invocation to get status/logs
    table_name = os.environ["STATUS_TABLE"]
    table = dynamodb.Table(table_name)
    
    job_id = None
    if "queryStringParameters" in event and event["queryStringParameters"]:
        job_id = event["queryStringParameters"].get("jobId")
    else:
        if "body" in event:
            try:
                body = json.loads(event["body"])
                job_id = body.get("jobId")
            except:
                pass

    if not job_id:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing jobId parameter."})
        }
    
    result = table.get_item(Key={"jobId": job_id})
    job_status_in_db = "UNKNOWN"
    if "Item" in result:
        job_status_in_db = result["Item"].get("status", "UNKNOWN")
    
    job_description = None
    try:
        job_name = f"nerf-training-{job_id}"
        job_description = sagemaker.describe_training_job(TrainingJobName=job_name)
    except sagemaker.exceptions.ClientError as e:
        job_description = {"error": str(e)}
    
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*"
        },
        "body": json.dumps({
            "dbStatus": job_status_in_db,
            "sageMakerJobDescription": job_description
        }, cls=DateTimeEncoder)
    }
