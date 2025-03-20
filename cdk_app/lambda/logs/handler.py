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

def handler(event, context):
    """
    Expects a query parameter or JSON body with {"jobId": "some-uuid"} 
    to retrieve the status/logs.
    """
    # Grab environment variables
    table_name = os.environ["STATUS_TABLE"]
    table = dynamodb.Table(table_name)

    # Basic event parsing (API Gateway can pass 'queryStringParameters' or 'body')
    job_id = None
    if "queryStringParameters" in event and event["queryStringParameters"]:
        job_id = event["queryStringParameters"].get("jobId")
    else:
        # Possibly a POST request with JSON
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

    # Query DynamoDB for job status
    result = table.get_item(Key={"jobId": job_id})
    job_status_in_db = "UNKNOWN"
    if "Item" in result:
        job_status_in_db = result["Item"].get("status", "UNKNOWN")

    # Optionally call SageMaker to get the actual status
    # In the main function, we use "nerf-training-{job_id}" 
    # so let's do that here:
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

