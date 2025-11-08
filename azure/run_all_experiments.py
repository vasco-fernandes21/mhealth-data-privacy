#!/usr/bin/env python3
"""
Run all experiments from YAML config on Azure ML with GPU.

Automatically submits all enabled experiments as parallel jobs.
"""

import sys
import yaml
from pathlib import Path
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute

# ============================================================================
# CONFIGURATION (hardcoded - use export before running)
# ============================================================================

subscription_id = "aef653af-c844-4925-aba1-df0f6ba17860"
resource_group = "Mestrado"
workspace_name = "ML_Mestrado"
compute_name = "gpu-cluster"

CONFIG_FILE = "experiments/scenarios/experiments.yaml"

GPU_SKU = "Standard_NC8as_T4_v3"
ENVIRONMENT = "azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:1"
MAX_PARALLEL_JOBS = 10

# ============================================================================
# INITIALIZE
# ============================================================================

def init_ml_client():
    """Initialize Azure ML client."""
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
        print("✅ Using DefaultAzureCredential")
    except Exception:
        print("🔐 Using InteractiveBrowserCredential...")
        credential = InteractiveBrowserCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    print(f"✅ Connected to: {workspace_name}")
    return ml_client

# ============================================================================
# GPU CLUSTER
# ============================================================================

def create_gpu_cluster(ml_client):
    """Create GPU cluster if it doesn't exist."""
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"✅ Cluster '{compute_name}' exists")
        print(f"   SKU: {compute.size}")
        print(f"   Max instances: {compute.max_instances}")
        return compute

    except Exception:
        print(f"📦 Creating cluster '{compute_name}'...")
        print(f"   SKU: {GPU_SKU}")

        compute = AmlCompute(
            name=compute_name,
            size=GPU_SKU,
            min_instances=0,
            max_instances=MAX_PARALLEL_JOBS,
            idle_time_before_scale_down=1800,
        )

        ml_client.compute.begin_create_or_update(compute).wait()
        print(f"✅ Cluster created")
        return compute

# ============================================================================
# LOAD CONFIG
# ============================================================================

def load_experiments(config_path):
    """Load experiments from YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiments = []
    for exp_name, exp_config in config.get("experiments", {}).items():
        if exp_config.get("enabled", True):
            experiments.append({
                "id": exp_name,
                "name": exp_config.get("name", exp_name),
                "dataset": exp_config.get("dataset"),
                "method": exp_config.get("method"),
                "seed": exp_config.get("seed"),
                "hyperparameters": exp_config.get("hyperparameters", {}),
            })

    return experiments

# ============================================================================
# BUILD COMMAND
# ============================================================================

def build_experiment_command(exp):
    """Build command for a single experiment."""
    cmd = (
        f"python experiments/run_experiments.py "
        f"--scenario {exp['method']} "
        f"--datasets {exp['dataset']} "
        f"--seed {exp['seed']} "
        f"--device cuda "
        f"--auto"
    )

    # Add hyperparameters if present
    hyperparams = exp.get("hyperparameters", {})
    if hyperparams:
        import json
        hp_json = json.dumps(hyperparams).replace('"', '\\"')
        cmd += f" --hyperparameters '{hp_json}'"

    return cmd

# ============================================================================
# SUBMIT JOBS
# ============================================================================

def submit_experiment(ml_client, exp, job_index, total):
    """Submit a single experiment job."""

    job_name = f"mhealth-{exp['id']}"
    display = f"[{job_index}/{total}] {exp['name']:<50}"

    print(f"  📤 {display}", end="", flush=True)

    try:
        job = command(
            code="./",
            command=build_experiment_command(exp),
            environment=ENVIRONMENT,
            compute=compute_name,
            experiment_name="mhealth-experiments",
            display_name=job_name,
        )

        submitted_job = ml_client.jobs.create_or_update(job)
        print(f" ✅")
        return (exp, submitted_job)

    except Exception as e:
        error_msg = str(e)[:80]
        print(f" ❌ {error_msg}")
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Submit all experiments to Azure ML."""

    print("="*80)
    print("🚀 Azure ML - Batch Experiment Runner")
    print("="*80)

    print("\n📋 Configuration:")
    print(f"   Workspace: {workspace_name}")
    print(f"   Resource Group: {resource_group}")
    print(f"   GPU SKU: {GPU_SKU}")
    print(f"   Environment: {ENVIRONMENT}")
    print(f"   Max parallel: {MAX_PARALLEL_JOBS}")

    # Initialize
    ml_client = init_ml_client()

    # Create GPU cluster
    print("\n📊 GPU Cluster:")
    create_gpu_cluster(ml_client)

    # Load config
    print(f"\n📋 Loading config: {CONFIG_FILE}")
    if not Path(CONFIG_FILE).exists():
        print(f"❌ Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    experiments = load_experiments(CONFIG_FILE)
    print(f"✅ Loaded {len(experiments)} experiments")

    # Group by scenario for summary
    scenarios = {}
    for exp in experiments:
        method = exp["method"]
        dataset = exp["dataset"]
        key = f"{method}_{dataset}"
        scenarios[key] = scenarios.get(key, 0) + 1

    print(f"\n📊 Breakdown:")
    for key, count in sorted(scenarios.items()):
        print(f"   {key}: {count} runs")

    # Submit experiments
    print(f"\n🚀 Submitting {len(experiments)} experiments\n")

    jobs = []
    failed = 0

    for idx, exp in enumerate(experiments, 1):
        result = submit_experiment(ml_client, exp, idx, len(experiments))
        if result:
            jobs.append(result)
        else:
            failed += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 Results: {len(jobs)}/{len(experiments)} submitted ✅")
    if failed > 0:
        print(f"           {failed} failed ❌")
    print(f"{'='*80}")

    if len(jobs) == 0:
        print("\n⚠️  No jobs were submitted")
        return 0

    print(f"\n💡 Monitor at: https://ml.azure.com/home")
    print(f"   Experiment: mhealth-experiments")

    print(f"\n📥 Commands:")
    print(f"  az ml job list --workspace-name {workspace_name} --experiment-name mhealth-experiments")
    print(f"  az ml job stream --name <job-id>")

    print(f"\n💾 Job IDs saved to: job_ids.txt")
    job_ids = {f"{exp['id']}": job.id for exp, job in jobs}
    with open("job_ids.txt", "w") as f:
        for exp_id, job_id in job_ids.items():
            f.write(f"{exp_id}: {job_id}\n")

    print(f"\n{'='*80}")
    return len(jobs)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success > 0 else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)