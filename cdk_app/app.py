
# app.py
import os
import aws_cdk as cdk
from my_rendering_pipeline_stack import MyRenderingPipelineStack
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks



app = cdk.App()

MyRenderingPipelineStack(app, "MyRenderingPipelineStack", 
    env=cdk.Environment(
        account=os.environ["CDK_DEFAULT_ACCOUNT"],
        region=os.environ["CDK_DEFAULT_REGION"]
    )
)

app.synth()
