from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
    CfnOutput,
)
from constructs import Construct

class MyRenderingPipelineStack(Stack):
    """
    A single CDK Stack that includes:
      - Reference to an existing S3 bucket ('user-submissions') for input data
      - A new S3 bucket for storing trained model outputs
      - A DynamoDB table for tracking job statuses
      - A Lambda function that triggers SageMaker training
      - A Lambda function that retrieves logs / job status
      - An API Gateway with:
         * Root route for the training Lambda
         * /logs route for the logs Lambda
         * CORS enabled for any origin (pre-production)
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # 1) REFERENCE EXISTING S3 BUCKET (user-submissions)
        submissions_bucket = s3.Bucket.from_bucket_name(
            self,
            "UserSubmissionsBucket",
            "user-submissions"  
        )

        # 2) CREATE A NEW BUCKET FOR OUTPUT DATA
        output_bucket = s3.Bucket(
            self,
            "OutputBucket-RenderingPipeline",
            bucket_name="gabe-output-renderingpipeline-bucket",
            removal_policy=RemovalPolicy.DESTROY
        )

        # 3) CREATE A DYNAMODB TABLE FOR JOB STATUS
        table = dynamodb.Table(
            self,
            "DynamoDB-RenderingPipeline",
            table_name="DynamoDB-RenderingPipeline",
            partition_key=dynamodb.Attribute(
                name="jobId",
                type=dynamodb.AttributeType.STRING
            ),
            removal_policy=RemovalPolicy.DESTROY
        )

        # 4) CREATE THE TRAINING LAMBDA (TRIGGERS SAGEMAKER)
        training_lambda = _lambda.Function(
            self,
            "Lambda-RenderingPipeline",
            function_name="Lambda-RenderingPipeline",
            runtime=_lambda.Runtime.PYTHON_3_9,
            handler="handler.handler",
            code=_lambda.Code.from_asset("lambda"),
            environment={
                "OUTPUT_BUCKET": output_bucket.bucket_name,
                "STATUS_TABLE": table.table_name,
                "SAGEMAKER_ROLE_ARN": "arn:aws:iam::975050048887:role/MySageMakerExecutionRole"
            },
            timeout=None,
            memory_size=2048
        )


        # 4a) PERMISSIONS FOR TRAINING LAMBDA
        training_lambda.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
        )
        submissions_bucket.grant_read(training_lambda)
        output_bucket.grant_read_write(training_lambda)
        table.grant_read_write_data(training_lambda)
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
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                resources=["*"]
            )
        )

        training_lambda.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=["arn:aws:iam::975050048887:role/MySageMakerExecutionRole"]
            )
        )

        # 5) CREATE THE LOGS LAMBDA (RETRIEVES SAGEMAKER/DB STATUS)
        logs_lambda = _lambda.Function(
            self,
            "Lambda-RenderingPipeline-Logs",
            function_name="Lambda-RenderingPipeline-Logs",
            runtime=_lambda.Runtime.PYTHON_3_9,
            handler="handler.handler",  # "handler.py" inside the 'logs' folder
            code=_lambda.Code.from_asset("lambda/logs"),
            environment={
                "STATUS_TABLE": table.table_name,
            },
            timeout=None,
            memory_size=1024
        )
        logs_lambda.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
        )
        # Logs Lambda can read from the table and describe SageMaker jobs
        table.grant_read_write_data(logs_lambda)
        logs_lambda.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:DescribeTrainingJob",
                    # if your logs function might call other sagemaker actions, list them here
                ],
                resources=["*"]
            )
        )

        # 6) CREATE AN API GATEWAY
        # We'll set up a root route for the training Lambda
        # and add an extra resource /logs for the logs Lambda
        # We'll also enable open CORS for now (*)
        api = apigw.RestApi(
            self,
            "APIGateway-RenderingPipeline",
            rest_api_name="APIGateway-RenderingPipeline",
            deploy_options=apigw.StageOptions(stage_name="prod"),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=apigw.Cors.DEFAULT_HEADERS,
            )
        )


        # Enable open CORS on entire API
        # for demonstration purposes—this is often more permissive than a production scenario
        api.add_gateway_response(
            "Default4XX",
            type=apigw.ResponseType.DEFAULT_4_XX,
            response_headers={
                "Access-Control-Allow-Origin": "'*'",
                "Access-Control-Allow-Headers": "'*'",
                "Access-Control-Allow-Methods": "'*'"
            }
        )
        api.add_gateway_response(
            "Default5XX",
            type=apigw.ResponseType.DEFAULT_5_XX,
            response_headers={
                "Access-Control-Allow-Origin": "'*'",
                "Access-Control-Allow-Headers": "'*'",
                "Access-Control-Allow-Methods": "'*'"
            }
        )

        # Root integration for training Lambda
        training_integration = apigw.LambdaIntegration(training_lambda)
        api.root.add_method("ANY", training_integration)

        # Create /start resource for training Lambda
        start_resource = api.root.add_resource("start")
        start_resource.add_method("ANY", training_integration)


        # /logs resource for logs Lambda
        logs_integration = apigw.LambdaIntegration(logs_lambda)
        logs_resource = api.root.add_resource("logs")
        logs_resource.add_method("ANY", logs_integration)
        
        # /stop resource for training Lambda
        stop_resource = api.root.add_resource("stop")
        stop_resource.add_method("ANY", training_integration)
        
        # Provide an output so you can easily find the API endpoint
        CfnOutput(
            self,
            "ApiEndpoint",
            value=api.url,
            description="API Gateway endpoint for MyRenderingPipeline"
        )
