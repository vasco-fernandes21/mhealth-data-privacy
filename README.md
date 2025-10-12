# Privacy-Preserving Health Data Analysis

Investigação do trade-off entre privacidade e precisão em sistemas de análise de dados de saúde móveis, aplicando técnicas de **Differential Privacy (DP)** e **Federated Learning (FL)** em dois datasets reais: **Sleep-EDF** e **WESAD**.

## 📋 Visão Geral

Este projeto implementa e compara três abordagens para análise de dados de saúde:

1. **Baseline**: Modelo LSTM padrão (sem técnicas de privacidade)
2. **Differential Privacy (DP)**: Modelo com ruído gaussiano para garantir privacidade
3. **Federated Learning (FL)**: Treino descentralizado simulando múltiplos dispositivos

## 🏗️ Estrutura do Projeto

```
mhealth-data-privacy/
├── src/                          # Código fonte modular
│   ├── preprocessing/            # Pré-processamento de dados
│   ├── models/                   # Arquiteturas de modelos
│   ├── privacy/                  # Implementações DP e FL
│   └── evaluation/               # Métricas e visualizações
├── notebooks/                    # Notebooks Jupyter/Colab
│   ├── 00_colab_setup.ipynb     # Template de setup
│   ├── 01_preprocess_sleep_edf.ipynb
│   ├── 02_preprocess_wesad.ipynb
│   ├── 03_train_baseline.ipynb
│   ├── 04_train_dp.ipynb
│   ├── 05_train_fl.ipynb
│   └── 06_analysis.ipynb
├── data/                         # Dados (não versionados)
├── models/                       # Modelos treinados (não versionados)
├── results/                      # Resultados e gráficos (não versionados)
├── setup.py                      # Configuração do package
├── requirements.txt              # Dependências
└── README.md                     # Este arquivo
```

## 🚀 Instalação

### Opção 1: Instalação Local

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/mhealth-data-privacy.git
cd mhealth-data-privacy

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar projeto como package
pip install -e .
```

### Opção 2: Google Colab (Recomendado)

Adicione no início de cada notebook:

```python
# Clonar repositório
!git clone https://github.com/seu-usuario/mhealth-data-privacy.git
%cd mhealth-data-privacy

# Instalar projeto
!pip install -e .

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

## 📊 Datasets

### Sleep-EDF
- **Fonte**: [PhysioNet](https://physionet.org/content/sleep-edfx/)
- **Descrição**: Dados de padrões de sono (EEG/EOG)
- **Classes**: 5 (W, N1, N2, N3, R)
- **Features**: 24 (tempo + frequência)

### WESAD
- **Fonte**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
- **Descrição**: Dados de resposta emocional a stress (ECG/EDA/Temperatura)
- **Classes**: 2 (stress vs. amusement)
- **Features**: 27 (tempo + frequência)

## 🔬 Workflow

### 1. Pré-processamento (executar 1x)

```bash
# Executar notebooks 01 e 02
# Dados processados são salvos no Google Drive
```

### 2. Treino de Modelos

```bash
# Baseline (notebook 03)
# DP (notebook 04)
# FL (notebook 05)
```

### 3. Análise de Resultados

```bash
# Comparação e visualizações (notebook 06)
```

## 📦 Uso do Código Modular

Após instalação com `pip install -e .`, você pode importar módulos:

```python
# Pré-processamento
from src.preprocessing.sleep_edf import preprocess_sleep_edf
from src.preprocessing.wesad import preprocess_wesad

# Modelos
from src.models.lstm_baseline import build_lstm_model, train_baseline

# Privacy
from src.privacy.dp_training import train_with_dp
from src.privacy.fl_training import train_with_fl

# Avaliação
from src.evaluation.metrics import evaluate_model, compute_metrics
from src.evaluation.visualization import plot_tradeoff_curve
```

## 🔐 Técnicas de Privacidade

### Differential Privacy (DP)
- **Biblioteca**: `tensorflow-privacy`
- **Parâmetros testados**: ε = 0.1, 1.0, 5.0, 10.0
- **Métrica**: Privacy budget (epsilon)

### Federated Learning (FL)
- **Biblioteca**: `Flower`
- **Configurações**: 3, 5, 10 clientes
- **Métrica**: Communication cost

## 📈 Resultados Esperados

- **Trade-off quantificado**: Accuracy vs. Privacy
- **Comparação entre datasets**: Validação de generalização
- **Recomendações práticas**: Valores ótimos de ε e número de clientes

## 🛠️ Tecnologias

- **Python**: 3.8+
- **ML Framework**: TensorFlow/Keras
- **Privacy**: TensorFlow Privacy, Flower
- **Signal Processing**: MNE, pyedflib, scipy
- **Visualização**: Matplotlib, Seaborn

## 📝 Citação

Se usar este código, por favor cite:

```bibtex
@mastersthesis{vasco2025privacy,
  title={Privacy-Preserving Health Data Analysis: Trade-offs between Privacy and Accuracy},
  author={Vasco},
  year={2025},
  school={Universidade}
}
```

## 📄 Licença

Este projeto é parte de uma dissertação de mestrado.

## 👤 Autor

**Vasco**  
Mestrado em Sistemas de Informação e Decisão em Medicina (SIDM)

## 🤝 Contribuições

Este é um projeto académico. Para questões ou sugestões, abra uma issue.

