# 🚀 Análise e Otimização: Differential Privacy & Federated Learning

## 📊 ANÁLISE DA ESTRUTURA DOS DADOS PROCESSADOS

### Sleep-EDF
- **Shape**: (313,922 train, 66,753 val, 72,330 test) × 24 features
- **Formato**: Dados já extraídos (features temporais + frequência)
- **Tipo**: Dados tabulares 2D → Requer windowing para LSTM

### WESAD  
- **Shape**: (715 train, 237 val, 237 test) × 14 channels × 1920 timesteps
- **Formato**: Janelas temporais pré-processadas
- **Tipo**: Dados 3D prontos para LSTM

---

## 🔍 PROBLEMAS IDENTIFICADOS

### 1. **SLEEP-EDF: Windowing Redundante e Lento**
```python
# ❌ PROBLEMA: Cria windows em CADA época de treinamento
def create_windows(X, y, window_size):
    n_windows = n_samples - window_size + 1
    for i in range(n_windows):  # Loop Python lento!
        X_windows[i] = X[i:i+window_size]
        y_windows[i] = y[i+window_size-1]
```

**Impacto**: 
- ~300k windows criadas por época
- Loop Python não vetorizado
- Tempo desperdiçado: ~5-10 segundos por época

### 2. **DP: DataLoader Não Otimizado**
```python
# ❌ PROBLEMA: Sem workers paralelos, sem pin_memory
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**Impacto**:
- I/O síncrono (bloqueante)
- Sem prefetching de dados
- GPU fica ociosa esperando dados

### 3. **FL: Criação de Windows Repetida por Cliente**
```python
# ❌ PROBLEMA: Cada cliente recria windows
def client_fn(cid: str):
    X_client, y_client = client_datasets[int(cid)]
    # Windows criadas aqui = N_clients × tempo
```

**Impacto**:
- 3-5 clientes × tempo de windowing
- Memória duplicada
- Simulação mais lenta

### 4. **DP: Augmentation em Tempo de Treino**
```python
# ❌ PROBLEMA: Augmentation determinística a cada treino
def _augment_temporal(X, noise_std=0.01, max_time_shift=8):
    # Loop Python para time shifts
    for i in range(n_samples):
        if shift != 0:
            X_aug[i] = np.pad(...)  # Operação lenta
```

**Impacto**:
- Augmentation poderia ser pré-calculada
- Loops Python lentos
- Overhead em cada época

### 5. **FL: Sem Cache de Modelo Base**
```python
# ❌ PROBLEMA: Modelo criado N vezes
def client_fn(cid: str):
    model = SimpleLSTM(...).to(device)  # Nova inicialização
```

**Impacto**:
- Inicialização repetida
- Overhead de memória
- Tempo de setup por round

---

## ✅ OTIMIZAÇÕES PROPOSTAS

### 🎯 **1. PRÉ-PROCESSAR WINDOWS UMA VEZ (Sleep-EDF)**

#### Antes:
```python
# Dados salvos: X_train.npy (313922, 24)
# Windows criadas em runtime
```

#### Depois:
```python
# Salvar dados já em formato windowed
# X_train_windows.npy (313913, 10, 24)  ← Pronto para LSTM!
```

**Implementação**:
```python
def save_windowed_data(data_dir: str, window_size: int = 10):
    """Pré-processa e salva dados em formato windowed"""
    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    
    # Vetorização usando view + stride tricks
    from numpy.lib.stride_tricks import sliding_window_view
    X_windows = sliding_window_view(X_train, window_size, axis=0)
    X_windows = X_windows.transpose(0, 2, 1)  # (n_windows, window, features)
    y_windows = y_train[window_size-1:]  # Labels alinhados
    
    np.save(f"{data_dir}/X_train_windows.npy", X_windows)
    np.save(f"{data_dir}/y_train_windows.npy", y_windows)
```

**Ganho Esperado**: 
- ⚡ **5-10s por época** eliminados
- 📉 Redução de 30-40% no tempo total de treino
- 🔄 Uma única vez no pré-processamento

---

### 🎯 **2. OTIMIZAR DATALOADERS (DP & FL)**

#### Antes:
```python
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

#### Depois:
```python
train_loader = DataLoader(
    dataset, 
    batch_size=64, 
    shuffle=True,
    num_workers=4,           # ✅ Paralelo
    pin_memory=True,         # ✅ GPU transfer rápido
    persistent_workers=True, # ✅ Reutiliza workers
    prefetch_factor=2,       # ✅ Prefetch batches
    drop_last=True           # ✅ DP requer batch fixo
)
```

**Ganho Esperado**:
- ⚡ **20-30% mais rápido** por época
- 🎯 GPU utilização aumenta de ~60% → ~90%
- 📊 Throughput aumenta ~1.5×

---

### 🎯 **3. PRÉ-CALCULAR AUGMENTATION (DP)**

#### Antes:
```python
# Augmentation em cada época
X_train_aug = _augment_temporal(X_train, seed=SEED)
```

#### Depois:
```python
# Salvar augmented data uma vez
def save_augmented_data(data_dir: str, n_augmentations: int = 2):
    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    
    X_aug_list = [X_train]
    y_aug_list = [y_train]
    
    for i in range(n_augmentations):
        X_aug = _augment_temporal(X_train, seed=SEED+i)
        X_aug_list.append(X_aug)
        y_aug_list.append(y_train)
    
    X_all = np.concatenate(X_aug_list, axis=0)
    y_all = np.concatenate(y_aug_list, axis=0)
    
    np.save(f"{data_dir}/X_train_augmented.npy", X_all)
    np.save(f"{data_dir}/y_train_augmented.npy", y_all)
```

**Ganho Esperado**:
- ⚡ **2-3s por época** eliminados
- 🔄 Augmentation determinística garantida
- 💾 Trade-off: +2× storage

---

### 🎯 **4. CACHE DE WINDOWED DATA (FL)**

#### Antes:
```python
def partition_data(X_train, y_train, num_clients):
    # Windows criadas por cliente
    client_datasets = []
    for i in range(num_clients):
        X_windows, y_windows = create_windows(X_client, y_client)
        client_datasets.append((X_windows, y_windows))
```

#### Depois:
```python
# Usar dados pré-windowed
X_train_windows = np.load("X_train_windows.npy")
y_train_windows = np.load("y_train_windows.npy")

def partition_windowed_data(X_windows, y_windows, num_clients):
    # Particionar dados já processados
    dataset_size = len(X_windows)
    client_size = dataset_size // num_clients
    indices = np.random.permutation(dataset_size)
    
    client_datasets = []
    for i in range(num_clients):
        start = i * client_size
        end = start + client_size if i < num_clients - 1 else dataset_size
        client_datasets.append((X_windows[indices[start:end]], 
                               y_windows[indices[start:end]]))
    return client_datasets
```

**Ganho Esperado**:
- ⚡ **10-15s** eliminados por run
- 🚀 Inicialização de clientes ~5× mais rápida
- 💾 Memória mais eficiente (sem duplicação)

---

### 🎯 **5. OTIMIZAR TRAINING LOOPS (DP)**

#### Antes:
```python
def train_one_epoch_dp(model, loader, criterion, optimizer, device, privacy_engine):
    for batch_idx, (batch_X, batch_y) in enumerate(loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Impressão frequente (overhead)
        if (batch_idx + 1) % 3 == 0:
            print(f"Batch {batch_idx + 1}/{len(loader)} - Loss: {loss.item():.4f}")
```

#### Depois:
```python
def train_one_epoch_dp(model, loader, criterion, optimizer, device, privacy_engine):
    # Progress bar com tqdm (mais eficiente)
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch_X, batch_y in pbar:
        batch_X, batch_y = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # ✅ Mais rápido
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        # Update progress bar (sem overhead)
        pbar.set_postfix({'loss': loss.item()})
```

**Ganho Esperado**:
- ⚡ **5-10%** mais rápido por época
- 📊 Melhor visualização de progresso
- 🎯 `non_blocking=True` + `set_to_none=True`

---

### 🎯 **6. MIXED PRECISION TRAINING (DP & FL)**

```python
# Antes: FP32 padrão
model = SimpleLSTM(...).to(device)

# Depois: FP16/BF16 para aceleração
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_X, batch_y in loader:
    with autocast():  # ✅ Operações em FP16
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Ganho Esperado**:
- ⚡ **30-50%** mais rápido (se GPU suporta)
- 💾 50% menos memória GPU
- ⚠️ Requer GPU moderna (Volta+, M1/M2/M3)

---

### 🎯 **7. COMPILAÇÃO DE MODELO (PyTorch 2.0+)**

```python
# Antes
model = SimpleLSTM(...).to(device)

# Depois: torch.compile
model = SimpleLSTM(...).to(device)
model = torch.compile(model, mode='reduce-overhead')  # ✅ JIT compilation
```

**Ganho Esperado**:
- ⚡ **10-20%** mais rápido
- 🎯 Funciona bem com LSTMs
- ⚠️ Requer PyTorch 2.0+

---

## 📈 ESTIMATIVA DE GANHOS TOTAIS

### **Sleep-EDF DP**
| Otimização | Ganho | Impacto Cumulativo |
|------------|-------|-------------------|
| Pre-windowing | -5s/época | **30-40%** |
| DataLoader otimizado | -20% tempo | **+20%** |
| Training loop | -5% | **+5%** |
| Mixed precision | -30% (GPU) | **+30%** |
| **TOTAL** | | **🚀 50-60% mais rápido** |

### **WESAD DP**
| Otimização | Ganho | Impacto Cumulativo |
|------------|-------|-------------------|
| Pre-augmentation | -2s/época | **15-20%** |
| DataLoader otimizado | -20% | **+20%** |
| Training loop | -5% | **+5%** |
| Mixed precision | -30% (GPU) | **+30%** |
| **TOTAL** | | **🚀 45-55% mais rápido** |

### **Federated Learning**
| Otimização | Ganho | Impacto Cumulativo |
|------------|-------|-------------------|
| Pre-windowing (Sleep-EDF) | -10s setup | **20-25%** |
| DataLoader otimizado | -15% | **+15%** |
| Cached clients | -20% | **+20%** |
| **TOTAL** | | **🚀 40-50% mais rápido** |

---

## 🛠️ PLANO DE IMPLEMENTAÇÃO

### **Fase 1: Pré-processamento (Alta Prioridade)**
1. ✅ Adicionar função `save_windowed_data` em `preprocessing/sleep_edf.py`
2. ✅ Adicionar função `save_augmented_data` em `preprocessing/wesad.py`
3. ✅ Executar pré-processamento uma vez
4. ✅ Atualizar funções de carregamento

### **Fase 2: DataLoaders (Média Prioridade)**
1. ✅ Atualizar todos os DataLoaders com workers paralelos
2. ✅ Adicionar `pin_memory` e `persistent_workers`
3. ✅ Testar impacto na velocidade

### **Fase 3: Training Loops (Média Prioridade)**
1. ✅ Substituir prints por tqdm
2. ✅ Adicionar `non_blocking=True`
3. ✅ Usar `set_to_none=True` em `zero_grad()`

### **Fase 4: Advanced (Baixa Prioridade)**
1. ⚠️ Testar mixed precision (se GPU disponível)
2. ⚠️ Testar torch.compile (PyTorch 2.0+)
3. ⚠️ Profiling detalhado com `torch.profiler`

---

## 📝 RECOMENDAÇÕES ESPECÍFICAS

### **Para Differential Privacy**
- ✅ **Crítico**: Pre-windowing (Sleep-EDF) e DataLoader otimizado
- ✅ **Importante**: Pre-augmentation (WESAD)
- ⚠️ **Opcional**: Mixed precision (ganho depende de GPU)

### **Para Federated Learning**
- ✅ **Crítico**: Pre-windowing (Sleep-EDF)
- ✅ **Importante**: DataLoader com workers
- ⚠️ **Opcional**: Client caching (complexidade vs ganho)

### **Compatibilidade**
- ✅ Todas otimizações Fase 1-3 são compatíveis com Opacus (DP)
- ✅ Flower (FL) suporta todas otimizações propostas
- ⚠️ Mixed precision: testar compatibilidade com Opacus
- ⚠️ torch.compile: pode ter issues com DPLSTM

---

## 🎯 PRÓXIMOS PASSOS

1. **Implementar Fase 1** (pré-processamento)
   - Modificar `preprocessing/sleep_edf.py`
   - Modificar `preprocessing/wesad.py`
   - Executar pré-processamento

2. **Atualizar Scripts de Treino**
   - Modificar DataLoaders em todos os scripts
   - Atualizar training loops com tqdm

3. **Benchmarking**
   - Medir tempo antes/depois
   - Comparar métricas de performance
   - Documentar ganhos reais

4. **Validação**
   - Verificar que resultados são reprodutíveis
   - Confirmar que DP/FL ainda funcionam corretamente
   - Testar com diferentes configurações

---

## 📊 MÉTRICAS DE SUCESSO

- ⏱️ **Tempo de treino reduzido em 40-60%**
- 🎯 **GPU utilização > 85%**
- 🔄 **Reprodutibilidade mantida**
- ✅ **Métricas de performance inalteradas**
- 📈 **Throughput aumentado em 1.5-2×**


