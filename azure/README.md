# Azure ML - Executar Experimentos com GPU

Setup simples para executar todos os experimentos no Azure ML com GPU.

## Setup (uma vez)

```bash
# 1. Instalar dependências
pip install -r azure/requirements.txt

# 2. Autenticar (abre browser)
python -c "from azure.identity import InteractiveBrowserCredential; InteractiveBrowserCredential().get_token('https://management.azure.com/.default')"

# 3. Obter Subscription ID do portal Azure
# https://portal.azure.com -> Subscriptions -> Copiar ID

# 4. Definir variáveis
export AZURE_SUBSCRIPTION_ID="seu-subscription-id"
export AZURE_RESOURCE_GROUP="mhealth-rg"
export AZURE_WORKSPACE_NAME="mhealth-ws"

# 5. Criar workspace (uma vez)
python azure/setup_workspace.py
```

## Executar Experimentos

```bash
# Executar todos os cenários (baseline, dp, fl, dp_fl)
python azure/run_all_experiments.py
```

Isto vai:
- Criar cluster GPU automaticamente (se não existir)
- Submeter 4 jobs (um por cenário)
- Cada job executa todos os experimentos do cenário
- Monitorizar em: https://ml.azure.com

## Custos

- GPU: `Standard_NC6s_v3` (1x NVIDIA V100) - ~$2.50/hora
- Cluster escala automaticamente (0-20 instâncias)
- Paga apenas quando está a executar

## Monitorizar

Aceder a: https://ml.azure.com
- Workspace: `mhealth-ws`
- Resource Group: `mhealth-rg`
- Experiment: `mhealth-experiments`
