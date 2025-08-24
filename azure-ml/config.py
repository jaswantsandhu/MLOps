import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Azure workspace details can be provided via environment variables
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "3fc7fd13-533e-40a7-8e3d-f1fbf4204436")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "edu-demo")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "edu-demo")

# Initialize ML Client
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)
