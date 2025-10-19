# 🎯 Recomendações de Baseline para Paper Científico

## 📊 Análise dos Datasets

### Sleep-EDF
- **Samples**: 453,005 épocas (30s cada)
- **Formato**: Feature vectors (24 features)
- **Classes**: 5 (W, N1, N2, N3, R) - **DESBALANCEADO** (W=64%)
- **Tipo**: Série temporal → Features extraídas
- **Split**: Random (não subject-wise)

### WESAD
- **Samples**: 1,189 janelas (60s cada)
- **Formato**: Temporal windows (14 channels × 1920 timesteps)
- **Classes**: 2 (non-stress, stress) - **DESBALANCEADO** (non-stress=70%)
- **Tipo**: Série temporal raw
- **Split**: Subject-wise (LOSO-style) - **IDEAL para FL**

---

## 🚨 PROBLEMA CRÍTICO IDENTIFICADO

### ⚠️ Incompatibilidade de Formatos

**Sleep-EDF**: Features extraídas (24 dimensões)
**WESAD**: Temporal windows (14×1920 dimensões)

**Problema**: Formatos diferentes impossibilitam comparação justa entre baselines!

---

## ✅ SOLUÇÃO RECOMENDADA

### 🎯 Opção 1: Uniformizar para Temporal Windows (RECOMENDADO)

#### **Vantagens:**
- ✅ Compatível com FL (necessita temporal structure)
- ✅ Compatível com DP (adiciona ruído aos dados raw)
- ✅ Preserva informação temporal completa
- ✅ Permite usar mesma arquitetura (LSTM) para ambos
- ✅ Mais robusto e generalizável

#### **Implementação:**

**Sleep-EDF:**
```python
# Usar dados temporais RAW em vez de features
# Criar janelas de 30s × 100 Hz = 3000 samples
# Shape: (n_windows, n_channels, 3000)
# Channels: 4 (EEG Fpz-Cz, EEG Pz-Oz, EOG H, EOG V)
```

**WESAD:**
```python
# Manter formato atual
# Shape: (n_windows, 14, 1920)
# 14 channels de sinais fisiológicos
```

**Modelo Baseline:**
```python
# LSTM Bidirectional simples (compatível com ambos)
class SimpleBidirectionalLSTM(nn.Module):
    def __init__(self, input_channels, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels, 
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
```

---

## 📋 PIPELINE UNIFICADO PROPOSTO

### 1. Preprocessing (Temporal Windows)

#### Sleep-EDF:
```python
# Criar sleep_edf_temporal.py
def preprocess_sleep_edf_temporal(
    data_dir: str,
    window_size: int = 3000,  # 30s × 100 Hz
    overlap: float = 0.0,      # Sem overlap (épocas sequenciais)
    normalize: str = 'per-channel-zscore'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Output shape: (n_windows, 4, 3000)
    Channels: [EEG_Fpz-Cz, EEG_Pz-Oz, EOG_H, EOG_V]
    """
```

#### WESAD:
```python
# Manter processamento atual
# Output shape: (n_windows, 14, 1920)
# Já tem overlap de 50% e normalização per-channel
```

### 2. Model Architecture (Unificado)

```python
class BaselineLSTM(nn.Module):
    """
    Arquitetura unificada para ambos os datasets
    Compatível com: Baseline, DP, FL, DP+FL
    """
    def __init__(self, input_channels, hidden_size, num_classes):
        super().__init__()
        
        # Normalização per-channel (camada)
        self.channel_norm = nn.LayerNorm(input_channels)
        
        # LSTM Bidirectional
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classificação
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, channels, timesteps)
        x = x.permute(0, 2, 1)  # (batch, timesteps, channels)
        x = self.channel_norm(x)
        lstm_out, (hn, _) = self.lstm(x)
        x = torch.cat([hn[-2], hn[-1]], dim=1)  # Concat bidirectional
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 3. Training Loop (Unificado)

```python
def train_baseline_unified(
    dataset_name: str,  # 'sleep-edf' ou 'wesad'
    model_config: dict,
    train_config: dict
):
    """
    Loop de treino unificado para ambos os datasets
    Garante mesma base para comparação com DP/FL
    """
    
    # Configurações comuns
    - Optimizer: Adam(lr=0.001, weight_decay=1e-4)
    - Loss: CrossEntropyLoss com class_weights
    - Scheduler: ReduceLROnPlateau
    - Early Stopping: patience=8
    - Batch size: 64
    - Max epochs: 100
```

---

## 🎯 HYPERPARÂMETROS RECOMENDADOS

### Sleep-EDF:
```python
config = {
    'input_channels': 4,      # EEG + EOG
    'hidden_size': 128,       # LSTM hidden units
    'num_layers': 2,          # LSTM layers
    'num_classes': 5,         # W, N1, N2, N3, R
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'class_weights': True,    # Para desbalanceamento
}
```

### WESAD:
```python
config = {
    'input_channels': 14,     # Sinais fisiológicos
    'hidden_size': 128,       # Manter igual ao Sleep-EDF
    'num_layers': 2,          # Manter igual
    'num_classes': 2,         # non-stress, stress
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'class_weights': True,    # Para desbalanceamento
}
```

---

## 🔧 COMPATIBILIDADE COM DP/FL

### Differential Privacy (DP):
```python
# Adiciona ruído aos gradientes durante treino
# COMPATÍVEL com temporal windows ✅
# Funciona sobre os dados raw/normalizados

from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)
```

### Federated Learning (FL):
```python
# Treino distribuído por sujeitos
# COMPATÍVEL com subject-wise split ✅
# WESAD já tem split adequado

# Sleep-EDF precisa adaptar para subject-wise:
# - Identificar sujeitos por arquivo
# - Criar splits por sujeito (LOSO-style)
```

### DP + FL:
```python
# Combina ambas as técnicas
# COMPATÍVEL com setup proposto ✅
# Cada cliente (sujeito) treina com DP local
```

---

## 📊 MÉTRICAS DE AVALIAÇÃO (Unificadas)

```python
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, average='weighted'),
    'recall': recall_score(y_true, y_pred, average='weighted'),
    'f1_score': f1_score(y_true, y_pred, average='weighted'),
    'confusion_matrix': confusion_matrix(y_true, y_pred),
    
    # Específicas para paper
    'per_class_f1': f1_score(y_true, y_pred, average=None),
    'training_time': time_elapsed,
    'inference_time': avg_inference_time,
}
```

---

## 🚀 PLANO DE IMPLEMENTAÇÃO

### Fase 1: Preprocessing Unificado
1. ✅ Criar `sleep_edf_temporal.py` (janelas temporais)
2. ✅ Adaptar `wesad.py` para compatibilidade
3. ✅ Garantir mesma normalização (per-channel z-score)
4. ✅ Subject-wise split para ambos

### Fase 2: Baseline Unificado
1. ✅ Criar `train_baseline_unified.py`
2. ✅ Arquitetura LSTM comum
3. ✅ Hyperparâmetros alinhados
4. ✅ Treino e validação

### Fase 3: Extensões
1. ⏳ DP: Adicionar Opacus
2. ⏳ FL: Adicionar Flower
3. ⏳ DP+FL: Combinar ambos

### Fase 4: Experimentos
1. ⏳ Baseline puro
2. ⏳ Baseline + DP (vários ε)
3. ⏳ Baseline + FL (vários clientes)
4. ⏳ Baseline + DP + FL

---

## 📝 TABELA COMPARATIVA PARA PAPER

| Method | Sleep-EDF Acc | Sleep-EDF F1 | WESAD Acc | WESAD F1 | Overhead |
|--------|---------------|--------------|-----------|----------|----------|
| Baseline | TBD | TBD | TBD | TBD | 1.0x |
| + DP (ε=10) | TBD | TBD | TBD | TBD | ~1.2x |
| + DP (ε=5) | TBD | TBD | TBD | TBD | ~1.2x |
| + DP (ε=1) | TBD | TBD | TBD | TBD | ~1.2x |
| + FL (3 clients) | TBD | TBD | TBD | TBD | ~2.0x |
| + FL (5 clients) | TBD | TBD | TBD | TBD | ~3.0x |
| + DP+FL | TBD | TBD | TBD | TBD | ~2.5x |

---

## 🎯 CONCLUSÃO

### Recomendação Final:

**UNIFORMIZAR para Temporal Windows com LSTM Bidirectional**

**Razões:**
1. ✅ Compatível com TODAS as técnicas (Baseline, DP, FL, DP+FL)
2. ✅ Preserva informação temporal completa
3. ✅ Permite comparação justa entre datasets
4. ✅ Mesma arquitetura = mesma complexidade
5. ✅ Adequado para paper científico rigoroso

**Próximos Passos:**
1. Criar `sleep_edf_temporal.py` para janelas temporais
2. Criar `train_baseline_unified.py` com arquitetura comum
3. Validar resultados do baseline
4. Implementar DP/FL sobre baseline validado


