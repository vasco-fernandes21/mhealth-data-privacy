#!/bin/bash

# ============================================================================
# Azure ML Configuration
# ============================================================================

export AZURE_SUBSCRIPTION_ID="aef653af-c844-4925-aba1-df0f6ba17860"
export AZURE_RESOURCE_GROUP="Mestrado"
export AZURE_WORKSPACE_NAME="ML_Mestrado"
export AZURE_COMPUTE_NAME="gpu-cluster"
export AZURE_GPU_SKU="Standard_NC8as_T4_v3"
export AZURE_ENVIRONMENT="azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:1"
export AZURE_MAX_JOBS="10"

# ============================================================================
# Run
# ============================================================================

echo "✅ Azure ML Configuration:"
echo "   Subscription: $AZURE_SUBSCRIPTION_ID"
echo "   Resource Group: $AZURE_RESOURCE_GROUP"
echo "   Workspace: $AZURE_WORKSPACE_NAME"
echo "   GPU SKU: $AZURE_GPU_SKU"
echo "   Environment: $AZURE_ENVIRONMENT"
echo "   Max parallel jobs: $AZURE_MAX_JOBS"
echo ""

python azure/run_all_experiments.py