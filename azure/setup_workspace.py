#!/usr/bin/env python3
"""
Create Azure ML workspace using Python SDK only (no Azure CLI required).
"""

import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.core.exceptions import ResourceNotFoundError

# Configuration
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP", "mhealth-rg")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME", "mhealth-ws")
location = os.getenv("AZURE_LOCATION", "westeurope")

if not subscription_id:
    print("❌ Error: AZURE_SUBSCRIPTION_ID not set")
    print("Get it from: https://portal.azure.com -> Subscriptions")
    print("Then: export AZURE_SUBSCRIPTION_ID='your-subscription-id'")
    exit(1)

print(f"🚀 Setting up Azure ML workspace...")
print(f"   Subscription: {subscription_id}")
print(f"   Resource Group: {resource_group}")
print(f"   Workspace: {workspace_name}")
print(f"   Location: {location}")

# Authenticate
print("\n📝 Authenticating with Azure...")
try:
    credential = DefaultAzureCredential()
    # Try to get a token to verify authentication
    credential.get_token("https://management.azure.com/.default")
    print("✅ Authenticated using DefaultAzureCredential")
except Exception:
    print("⚠️  DefaultAzureCredential failed, trying interactive login...")
    credential = InteractiveBrowserCredential()
    credential.get_token("https://management.azure.com/.default")
    print("✅ Authenticated using InteractiveBrowserCredential")

# Create resource group if it doesn't exist
print(f"\n📦 Checking resource group: {resource_group}...")
resource_client = ResourceManagementClient(credential, subscription_id)

try:
    resource_client.resource_groups.get(resource_group)
    print(f"✅ Resource group '{resource_group}' already exists")
except ResourceNotFoundError:
    print(f"📦 Creating resource group '{resource_group}'...")
    resource_client.resource_groups.create_or_update(
        resource_group,
        {"location": location}
    )
    print(f"✅ Resource group created")

# Create or get workspace
print(f"\n📦 Setting up workspace: {workspace_name}...")
try:
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    workspace = ml_client.workspaces.get(workspace_name)
    print(f"✅ Workspace '{workspace_name}' already exists")
except Exception:
    print(f"📦 Creating workspace '{workspace_name}'...")
    workspace = Workspace(
        name=workspace_name,
        location=location,
        display_name=workspace_name,
        description="Workspace for mHealth privacy experiments",
    )
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
    )
    ml_client.workspaces.begin_create(workspace).result()
    print(f"✅ Workspace created")

print(f"\n✅ Setup complete!")
print(f"\n📋 Workspace details:")
print(f"   Name: {workspace_name}")
print(f"   Resource Group: {resource_group}")
print(f"   Location: {location}")
print(f"   Portal URL: https://ml.azure.com/workspaces/{workspace_name}")

print(f"\n💡 You can now run experiments:")
print(f"   python azure/run_all_experiments.py")

