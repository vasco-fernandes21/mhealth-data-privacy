#!/bin/bash

set -e  # Exit on error

echo "="*80
echo "🔄 FULL RESET - Azure ML Experiments"
echo "="*80

# 1. Delete cluster
echo -e "\n🗑️  Step 1: Deleting old cluster 'gpu-cluster'..."
az ml compute delete --name gpu-cluster \
  --workspace-name ML_Mestrado \
  --resource-group Mestrado \
  --yes 2>/dev/null || echo "⚠️  Cluster already deleted or doesn't exist"

echo "⏳ Waiting for cleanup (15 seconds)..."
sleep 15

# 2. Verify deletion
echo -e "\n✅ Step 2: Verifying deletion..."
CLUSTERS=$(az ml compute list --workspace-name ML_Mestrado --resource-group Mestrado \
  --query "length(@)" -o tsv)
echo "   Remaining compute targets: $CLUSTERS"

# 3. Show job status before resubmit
echo -e "\n📊 Step 3: Current job status:"
FAILED=$(az ml job list --workspace-name ML_Mestrado --resource-group Mestrado \
  --query "[?status=='Failed'] | length(@)" -o tsv)
RUNNING=$(az ml job list --workspace-name ML_Mestrado --resource-group Mestrado \
  --query "[?status=='Running'] | length(@)" -o tsv)
COMPLETED=$(az ml job list --workspace-name ML_Mestrado --resource-group Mestrado \
  --query "[?status=='Completed'] | length(@)" -o tsv)

echo "   Failed: $FAILED"
echo "   Running: $RUNNING"
echo "   Completed: $COMPLETED"

# 4. Resubmit
echo -e "\n🚀 Step 4: Resubmitting all experiments..."
echo "   Starting new run..."
./azure/azure.sh

echo -e "\n"
echo "="*80
echo "✅ RESTART COMPLETE"
echo "="*80
echo ""
echo "📊 Monitor progress:"
echo "   az ml job list --workspace-name ML_Mestrado --resource-group Mestrado -o table"
echo ""