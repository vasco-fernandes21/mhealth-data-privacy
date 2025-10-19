# 🎯 Baseline Final - Comparações Intra-Dataset

## 📊 Resumo Executivo

**Objetivo**: Baseline comum para comparações **intra-dataset**:
- Sleep-EDF: Baseline vs DP vs FL vs DP+FL
- WESAD: Baseline vs DP vs FL vs DP+FL

**Formatos mantidos**: Cada dataset usa seu formato otimizado original
**Split**: ✅ Ambos agora usam **subject-wise split** (crítico para FL)

---

## 🔬 SLEEP-EDF

### 📦 Características dos Dados

```python
# Formato
Shape: (453005, 24) epochs × features
Split: Subject-wise ✅ (implementado agora!)
Classes: 5 (W, N1, N2, N3, R)
Desbalanceamento: W=64%, outros <20%

# Processamento
- Features: Time + Frequency domain (8 per channel)
- Channels: 4 (EEG Fpz-Cz, EEG Pz-Oz, EOG H, EOG V)
- Epoch: 30 segundos
- Normalização: StandardScaler (train-only)
```

### 🎯 Baseline Recomendado

**Arquitetura:**
```python
class SleepEDFBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # Janelas de 10 épocas consecutivas
        # Input: (batch, 10, 24)
        
        self.lstm = nn.LSTM(
            input_size=24,      # Features por época
            hidden_size=128,    # Hidden units
            num_layers=2,       # Layers
            batch_first=True,
            bidirectional=False # Unidirecional
        )
        
        self.fc = nn.Linear(128, 5)  # 5 classes
```

**Hyperparâmetros:**
```python
config = {
    'window_size': 10,          # Épocas por janela
    'stride': 10,               # Sem overlap (épocas sequenciais)
    'hidden_size': 128,
    'num_layers': 2,
    'num_classes': 5,
    'dropout': 0.0,             # Sem dropout no baseline simples
    'learning_rate': 0.001,
    'batch_size': 64,
    'optimizer': 'Adam',
    'weight_decay': 0.0,
    'epochs': 100,
    'early_stopping': 5,
    
    # Para lidar com desbalanceamento
    'class_weights': True,      # Usar class weights
    'criterion': 'CrossEntropyLoss'
}
```

### ✅ Compatibilidade

| Técnica | Compatível | Notas |
|---------|-----------|-------|
| **Baseline** | ✅ | Implementado |
| **DP** | ✅ | Opacus sobre gradientes |
| **FL** | ✅ | Subject-wise split pronto |
| **DP+FL** | ✅ | Combinação de ambos |

### 📊 Métricas Esperadas

```python
# Baseline (estimativa)
Accuracy: ~75-85%
F1-Score (weighted): ~0.70-0.80

# Desafios:
- Desbalanceamento de classes
- N1 e N3 são minoritários
- Variabilidade inter-sujeito
```

---

## 💊 WESAD

### 📦 Características dos Dados

```python
# Formato
Shape: (1189, 14, 1920) windows × channels × timesteps
Split: Subject-wise ✅ (já implementado)
Classes: 2 (non-stress, stress)
Desbalanceamento: non-stress=70%, stress=30%

# Processamento
- Raw temporal data (14 sinais fisiológicos)
- Window: 60 segundos × 32 Hz = 1920 samples
- Overlap: 50%
- Normalização: Per-channel z-score (train-only)
- Subjects: 15 total → 9 train, 3 val, 3 test
```

### 🎯 Baseline Recomendado

**Arquitetura:**
```python
class WESADBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (batch, 14, 1920)
        
        self.lstm = nn.LSTM(
            input_size=14,       # Canais
            hidden_size=128,     # Hidden units (mesma capacidade que Sleep-EDF)
            num_layers=2,        # Layers
            batch_first=True,
            bidirectional=True   # Bidirectional para temporal
        )
        
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 classes
```

**Hyperparâmetros:**
```python
config = {
    'input_channels': 14,
    'hidden_size': 128,         # Alinhado com Sleep-EDF
    'num_layers': 2,
    'num_classes': 2,
    'bidirectional': True,
    'dropout': 0.3,             # Dropout para evitar overfit (dataset pequeno)
    'learning_rate': 0.001,
    'batch_size': 64,
    'optimizer': 'Adam',
    'weight_decay': 1e-4,       # L2 regularização
    'epochs': 100,
    'early_stopping': 8,
    
    # Para lidar com desbalanceamento
    'class_weights': True,
    'label_smoothing': 0.05,    # Suavização de labels
    'criterion': 'CrossEntropyLoss'
}
```

### ✅ Compatibilidade

| Técnica | Compatível | Notas |
|---------|-----------|-------|
| **Baseline** | ✅ | Já implementado |
| **DP** | ✅ | Opacus sobre gradientes |
| **FL** | ✅ | Subject-wise split pronto |
| **DP+FL** | ✅ | Combinação de ambos |

### 📊 Métricas Esperadas

```python
# Baseline (atual)
Accuracy: ~85-90%
F1-Score (weighted): ~0.80-0.85

# Vantagens:
- Binary classification mais simples
- Subject-wise split adequado
- Bom número de features temporais
```

---

## 🔄 Comparação de Arquiteturas

### Semelhanças (Base Comum)
```python
✅ LSTM com 2 layers
✅ Hidden size: 128
✅ Adam optimizer (lr=0.001)
✅ Batch size: 64
✅ Early stopping
✅ Class weights para desbalanceamento
✅ Subject-wise split
```

### Diferenças (Específicas por Dataset)

| Aspecto | Sleep-EDF | WESAD |
|---------|-----------|-------|
| **Input** | Feature vectors (24) | Temporal windows (14×1920) |
| **LSTM** | Unidirecional | Bidirectional |
| **Dropout** | Não | Sim (0.3) |
| **Regularização** | Não | L2 (1e-4) |
| **Classes** | 5 (multiclass) | 2 (binary) |
| **Samples** | 453k épocas | 1.2k janelas |

---

## 🚀 Pipeline de Experimentos

### 1. Baseline Puro
```bash
# Sleep-EDF
python src/train/sleep-edf/train_baseline.py

# WESAD  
python src/train/wesad/train_baseline.py
```

### 2. Baseline + Differential Privacy
```bash
# Sleep-EDF com DP
python src/train/sleep-edf/differential-privacy/train_dp.py --epsilon 10.0

# WESAD com DP
python src/train/wesad/differential-privacy/train_dp.py --epsilon 10.0

# Testar múltiplos ε: 1.0, 5.0, 10.0
```

### 3. Baseline + Federated Learning
```bash
# Sleep-EDF com FL
python src/train/sleep-edf/federated-learning/train_fl.py --num_clients 5

# WESAD com FL  
python src/train/wesad/federated-learning/train_fl.py --num_clients 3

# WESAD: 3 clients = 3 test subjects (LOSO)
# Sleep-EDF: 5 clients = distribuir subjects igualmente
```

### 4. Baseline + DP + FL
```bash
# Sleep-EDF com DP+FL
python src/train/sleep-edf/dp-fl/train_dp_fl.py --epsilon 5.0 --num_clients 5

# WESAD com DP+FL
python src/train/wesad/dp-fl/train_dp_fl.py --epsilon 5.0 --num_clients 3
```

---

## 📊 Tabela de Resultados (Template para Paper)

### Sleep-EDF

| Method | Accuracy | F1 (W) | F1 (N1) | F1 (N2) | F1 (N3) | F1 (R) | F1 (Weighted) | Time (s) |
|--------|----------|---------|---------|---------|---------|--------|---------------|----------|
| Baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + DP (ε=10) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + DP (ε=5) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + DP (ε=1) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + FL (5c) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + DP+FL | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### WESAD

| Method | Accuracy | Precision | Recall | F1 (non-stress) | F1 (stress) | F1 (Weighted) | Time (s) |
|--------|----------|-----------|--------|----------------|-------------|---------------|----------|
| Baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + DP (ε=10) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + DP (ε=5) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + DP (ε=1) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + FL (3c) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + DP+FL | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## ⚠️ AÇÃO NECESSÁRIA

### Sleep-EDF: Reprocessar Dados

```bash
# IMPORTANTE: Os dados atuais usam split random!
# Precisas reprocessar com subject-wise split:

cd /Users/vasco/Desktop/Mestrado/SIDM/mhealth-data-privacy

# Backup dos dados antigos (opcional)
mv data/processed/sleep-edf data/processed/sleep-edf.backup

# Reprocessar com subject-wise split
source venv/bin/activate
python -m src.preprocessing.sleep_edf \
    --data_dir data/raw/sleep-edf \
    --output_dir data/processed/sleep-edf

# Verificar que tem subject_splits
python -c "
import joblib
info = joblib.load('data/processed/sleep-edf/preprocessing_info.pkl')
print('✅ Subject splits:', 'subject_splits' in info)
print('Train subjects:', len(info.get('subject_splits', {}).get('train_subjects', [])))
"
```

### WESAD: Pronto ✅

WESAD já tem subject-wise split implementado e funcional!

---

## 🎯 Checklist Final

### Sleep-EDF
- ✅ Subject-wise split implementado no código
- ⚠️ **Precisa reprocessar dados**
- ✅ Baseline model pronto
- ⏳ DP implementation
- ⏳ FL implementation
- ⏳ DP+FL implementation

### WESAD
- ✅ Subject-wise split funcional
- ✅ Dados processados corretos
- ✅ Baseline model pronto
- ⏳ DP implementation
- ⏳ FL implementation
- ⏳ DP+FL implementation

---

## 📝 Próximos Passos

1. **Reprocessar Sleep-EDF** com subject-wise split
2. **Validar baselines** funcionais em ambos datasets
3. **Implementar DP** (Opacus) para ambos
4. **Implementar FL** (Flower) para ambos  
5. **Combinar DP+FL**
6. **Executar experimentos** completos
7. **Coletar métricas** para paper

---

## 💡 Considerações para o Paper

### Strengths do Setup
1. ✅ **Subject-wise split** evita data leakage
2. ✅ **Base comum** (LSTM, hyperparâmetros alinhados)
3. ✅ **Formatos otimizados** para cada dataset
4. ✅ **Compatibilidade** com todas as técnicas (DP, FL, DP+FL)

### Challenges
1. ⚠️ **Sleep-EDF**: Desbalanceamento severo (W=64%)
2. ⚠️ **WESAD**: Dataset pequeno (apenas 15 subjects)
3. ⚠️ **Trade-offs**: DP reduz accuracy, FL aumenta overhead

### Contribution
1. 🎯 Comparação rigorosa de DP vs FL vs DP+FL
2. 🎯 Avaliação em dois datasets reais de saúde
3. 🎯 Subject-wise evaluation (realista para produção)
4. 🎯 Análise de trade-offs privacy-utility


