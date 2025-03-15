from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
    # We don't need a direct SageMaker import unless we're provisioning SageMaker resources directly
    # For now, the Lambda will call SageMaker with Boto3 at runtime
)
from constructs import Construct


class MyRenderingPipelineStack(Stack):
    """
    A single CDK Stack that includes:
      - Reference to an existing S3 bucket ('user-submissions') for input data
      - A new S3 bucket for storing trained model outputs
      - A DynamoDB table for tracking job statuses
      - A Lambda function that triggers SageMaker training
      - An API Gateway that invokes the Lambda
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # 1) REFERENCE EXISTING S3 BUCKET (user-submissions)
        # This will NOT modify or overwrite your existing bucket;
        # it simply creates a reference for our code to use.
        submissions_bucket = s3.Bucket.from_bucket_name(
            self,
            "UserSubmissionsBucket",
            "user-submissions"  # The name of your existing S3 bucket
        )

        # 2) CREATE A NEW BUCKET FOR OUTPUT DATA
        # We'll store final trained models or logs here
        output_bucket = s3.Bucket(
            self,
            "OutputBucket-RenderingPipeline",
            bucket_name="gabe-output-renderingpipeline-bucket",  # Must be globally unique!
            removal_policy=RemovalPolicy.DESTROY
        )

        # 3) CREATE A DYNAMODB TABLE FOR JOB STATUS
        # This table will track SageMaker job metadata (jobId, status, timestamps, etc.)
        table = dynamodb.Table(
            self,
            "DynamoDB-RenderingPipeline",
            table_name="DynamoDB-RenderingPipeline",  # example name
            partition_key=dynamodb.Attribute(
                name="jobId",
                type=dynamodb.AttributeType.STRING
            ),
            removal_policy=RemovalPolicy.DESTROY  # For easy cleanup
        )

        # 4) CREATE THE LAMBDA FUNCTION (TRIGGER SAGEMAKER)
        # We assume you'll place your Lambda code in a 'lambda' folder
        # that contains a 'handler.py' with a 'handler(event, context)' function.
        # e.g. cdk_app/lambda/handler.py
        training_lambda = _lambda.Function(
            self,
            "Lambda-RenderingPipeline",
            function_name="Lambda-RenderingPipeline",
            runtime=_lambda.Runtime.PYTHON_3_9,
            handler="handler.handler",  # "handler.py" file, "handler" function
            code=_lambda.Code.from_asset("lambda"),
            environment={
                "OUTPUT_BUCKET": output_bucket.bucket_name,
                "STATUS_TABLE": table.table_name,
                # Add any other environment variables if needed
            },
            timeout=None,  # default is short; consider a bigger timeout if needed
            memory_size=2048
        )

        # 4a) PERMISSIONS FOR THE LAMBDA TO CALL SAGEMAKER, DYNAMODB, AND S3
        # - The managed policy for AWSLambdaBasicExecutionRole covers basic logging, etc.
        # - We attach an inline policy for SageMaker: create training jobs, list jobs, etc.
        training_lambda.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
        )

        # For reading/writing from S3
        submissions_bucket.grant_read(training_lambda)
        output_bucket.grant_read_write(training_lambda)

        # For reading/writing job statuses in DynamoDB
        table.grant_read_write_data(training_lambda)

        # For SageMaker:
        training_lambda.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:ListTrainingJobs",
                    "sagemaker:CreateModel",
                    "sagemaker:DescribeModel",
                    "sagemaker:DeleteModel",
                    "ecr:GetAuthorizationToken",    # if pulling container images from ECR
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "logs:CreateLogGroup",         # typical logging
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                resources=["*"]  # for simplicity; consider restricting to specific ARNs
            )
        )

        # 5) CREATE AN API GATEWAY TO INVOKE THE LAMBDA
        api = apigw.LambdaRestApi(
            self,
            "APIGateway-RenderingPipeline",
            rest_api_name="APIGateway-RenderingPipeline",
            handler=training_lambda,
            deploy_options=apigw.StageOptions(stage_name="prod")
        )

        # If you need CORS configuration:
        # add_cors_options(api.root)

        # That’s it! We have a single Stack that references an existing bucket,
        # makes a new output bucket, a status-tracking DynamoDB table,
        # a Lambda to launch training jobs on SageMaker, and an API Gateway.
