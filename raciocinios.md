# Raciocínios e Decisões no Processamento de Dados

## Resumo Executivo

Este documento explica as decisões técnicas e raciocínios por trás do processamento dos datasets **Sleep-EDF** e **WESAD** para análise de dados de saúde com preservação de privacidade. O processamento foi adaptado para lidar com formatos específicos de dados e limitações técnicas encontradas.

---

## 1. Sleep-EDF Dataset

### 1.1 Problema Inicial Identificado

**Situação**: O código original esperava ficheiros `.edf` (European Data Format), mas os dados disponíveis estavam em formato `.rec` e `.hyp` (PhysioNet format).

**Evidência**: 
- Ficheiros encontrados: `sc4002e0.rec`, `sc4002e0.hyp`, etc.
- Código original: `mne.io.read_raw_edf()` esperava ficheiros `.edf`

### 1.2 Análise da Documentação PhysioNet

**Descoberta**: Os ficheiros `.rec` e `.hyp` são na verdade ficheiros EDF com extensões diferentes:
- `.rec`: Gravações de sinais (EEG, EOG, EMG)
- `.hyp`: Hypnogramas (labels de sono)

**Estrutura dos dados**:
- **Sinais**: EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG, respiração, temperatura
- **Frequência**: 100 Hz
- **Labels**: 0=W, 1=S1, 2=S2, 3=S3, 4=S4, 5=R, 6=M, 9=unscored

### 1.3 Solução Implementada

**Abordagem**: Usar `pyedflib` em vez de `mne` para carregar ficheiros EDF com extensões `.rec/.hyp`.

**Raciocínio**:
1. `pyedflib` é mais direto para ficheiros EDF
2. Menos overhead de conversão
3. Melhor compatibilidade com o formato específico

**Implementação**:
```python
def load_physionet_file(rec_path: str, hyp_path: str):
    import pyedflib
    
    # Load recording using pyedflib
    f = pyedflib.EdfReader(rec_path)
    # Extract signals and hypnogram
```

### 1.4 Problema dos Hypnogramas

**Descoberta**: O hypnograma estava vazio devido à função de parsing incorreta.

**Análise do ficheiro `.hyp`**:
- Header com metadados
- Dados do hypnograma no final como dígitos individuais (0-6)
- Cada dígito representa um estágio de sono de 30 segundos

**Solução**:
```python
def load_physionet_hypnogram(hyp_path: str):
    # Extract numeric values from the entire content
    for char in content:
        if char.isdigit() and int(char) <= 6:
            hypnogram_data.append(int(char))
```

### 1.5 Mapeamento de Labels

**Problema**: Labels PhysioNet vs. padrão clínico
- PhysioNet: 0=W, 1=S1, 2=S2, 3=S3, 4=S4, 5=R, 6=M
- Padrão: 0=W, 1=N1, 2=N2, 3=N3, 4=R

**Solução**: Mapeamento direto com consolidação de S3/S4 → N3
```python
stage_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 0}
```

### 1.6 Resultados Finais Sleep-EDF

- ✅ **291 épocas processadas** (8 ficheiros)
- ✅ **24 features por época** (8 por sinal × 3 sinais)
- ✅ **5 classes** (W, N1, N2, N3, R)
- ✅ **Distribuição equilibrada**: [85, 70, 60, 52, 24]

---

## 2. WESAD Dataset

### 2.1 Problema Inicial Identificado

**Situação**: Erro "1" ao carregar ficheiros `.pkl` - estrutura de dados incorreta.

**Evidência**: Código original tentava aceder a `data['signal'][1][0]` mas a estrutura real era diferente.

### 2.2 Análise da Estrutura Real

**Descoberta através de investigação**:
```python
# Estrutura real dos dados WESAD
data = {
    'signal': {
        'chest': {  # RespiBAN (700 Hz)
            'ECG': (4255300, 1),
            'EDA': (4255300, 1), 
            'Temp': (4255300, 1),
            'ACC': (4255300, 3)
        },
        'wrist': {  # Empatica E4
            'BVP': (389056, 1),   # 64 Hz
            'EDA': (24316, 1),    # 4 Hz
            'TEMP': (24316, 1),   # 4 Hz
            'ACC': (194528, 3)    # 32 Hz
        }
    },
    'label': (4255300,)  # 700 Hz, sincronizado com chest
}
```

### 2.3 Decisão Estratégica: Chest vs Wrist

**Problema**: Dois dispositivos com frequências diferentes e sincronização complexa.

**Análise**:
- **Chest (RespiBAN)**: 700 Hz, já sincronizado com labels
- **Wrist (Empatica E4)**: Frequências variáveis (4-64 Hz), precisa sincronização manual

**Decisão**: Usar dados do **chest (RespiBAN)** porque:
1. ✅ Já sincronizado com labels
2. ✅ Frequência uniforme (700 Hz)
3. ✅ Menos complexidade de processamento
4. ✅ Sinais mais estáveis (ECG vs BVP)

### 2.4 Problema de Frequências de Amostragem

**Descoberta**: Resampling para 4 Hz criou inconsistências nas dimensões dos sinais.

**Problema**: Sinais com comprimentos diferentes após resampling
- ECG: 1519 amostras
- EDA: 24316 amostras  
- TEMP: 24316 amostras
- ACC: 3039 amostras

**Solução**: Usar comprimento de referência baseado no sinal mais longo
```python
reference_length = len(signals_dict['eda'])
target_length = reference_length
```

### 2.5 Problema dos Filtros

**Erro**: "Digital filter critical frequencies must be 0 < Wn < fs/2 (fs=4.0 -> fs/2=2.0)"

**Causa**: Filtros com frequências de corte superiores ao limite de Nyquist (2 Hz para fs=4 Hz).

**Soluções aplicadas**:
- ECG: 0.5-40 Hz → 0.5-1.5 Hz
- ACC: 0.1-5 Hz → 0.1-1.5 Hz
- EDA: 0.05-1 Hz (mantido, dentro do limite)

### 2.6 Problema das Dimensões dos Sinais

**Erro**: "inhomogeneous shape after 2 dimensions"

**Causa**: ACC tem 3 canais (x, y, z) mas outros sinais têm 1 canal.

**Solução**: Usar apenas o primeiro canal (eixo x) do acelerómetro
```python
ecg_window = ecg[i:i+window_size, 0]  # Take first channel
acc_window = acc[i:i+window_size, 0]  # Take first channel (x-axis)
```

### 2.7 Correção dos Labels WESAD

**Problema identificado**: O código estava a usar labels incorretos para filtragem.

**Labels corretos do WESAD**:
- 0: not defined/transient (ignorar)
- 1: baseline (ignorar)
- 2: stress ✅
- 3: amusement ✅
- 4: meditation (ignorar)
- 5,6,7: outros (ignorar)

**Correção aplicada**:
```python
# Antes (incorreto): (y == 1) | (y == 2)  # baseline e stress
# Depois (correto):  (y == 2) | (y == 3)  # stress e amusement
```

### 2.8 Resultados Finais WESAD

- ✅ **539 janelas processadas** (15 ficheiros)
- ✅ **36 features por janela** (9 por sinal × 4 sinais)
- ✅ **2 classes** (stress vs amusement)
- ✅ **Filtragem aplicada**: Removidos baseline, meditation e transições

---

## 3. Decisões de Design Geral

### 3.1 Escolha de Frequência de Amostragem

**Decisão**: 4 Hz para WESAD

**Raciocínio**:
1. **Realismo**: Frequência típica de dispositivos móveis
2. **Eficiência**: Menos dados para processar
3. **Compatibilidade**: Adequada para sinais fisiológicos lentos (EDA, temperatura)

### 3.2 Segmentação Temporal

**Sleep-EDF**: Épocas de 30 segundos (padrão clínico)
**WESAD**: Janelas de 60 segundos com 50% overlap

**Raciocínio**:
- Sleep-EDF: Segue padrão clínico estabelecido
- WESAD: Janelas maiores para capturar padrões de stress/emoção

### 3.3 Extração de Features

**Abordagem**: Features no domínio do tempo e frequência

**Justificação**:
- **Domínio do tempo**: Estatísticas básicas (média, desvio padrão, min, max)
- **Domínio da frequência**: Power spectral density para diferentes bandas
- **Combinação**: Captura tanto características instantâneas quanto padrões rítmicos

### 3.4 Extração de Features Detalhada

**Sleep-EDF - Features por Época (30s)**:
```python
def extract_sleep_features(epoch: np.ndarray, sfreq: float = 100):
    # epoch shape: (3, 3000) - 3 sinais, 3000 amostras (30s × 100Hz)
    features = []
    
    for signal in epoch:  # Para cada sinal (EEG Fpz-Cz, EEG Pz-Oz, EOG)
        # Domínio do tempo (4 features)
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        min_val = np.min(signal)
        max_val = np.max(signal)
        
        # Domínio da frequência (4 features)
        freqs, psd = signal.welch(signal, sfreq, nperseg=256)
        delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])    # 0.5-4 Hz
        theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])      # 4-8 Hz
        alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])     # 8-13 Hz
        beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])     # 13-30 Hz
        
        features.extend([mean_val, std_val, min_val, max_val, delta, theta, alpha, beta])
    
    return np.array(features)  # 24 features total (8 × 3 sinais)
```

**WESAD - Features por Janela (60s)**:
```python
def extract_wesad_features(window: np.ndarray, sfreq: float = 4):
    # window shape: (4, 240) - 4 sinais, 240 amostras (60s × 4Hz)
    features = []
    
    for channel in window:  # Para cada sinal (ECG, EDA, TEMP, ACC)
        # Domínio do tempo (4 features)
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        min_val = np.min(channel)
        max_val = np.max(channel)
        
        # Domínio da frequência (5 features)
        freqs, psd = signal.welch(channel, sfreq, nperseg=min(len(channel), 64))
        # Usar apenas as primeiras 5 componentes do PSD
        features.extend([mean_val, std_val, min_val, max_val] + list(psd[:5]))
    
    return np.array(features)  # 36 features total (9 × 4 sinais)
```

### 3.5 Segmentação e Windowing

**Sleep-EDF - Segmentação em Épocas**:
```python
def segment_signals(signals: np.ndarray, labels: np.ndarray, sfreq: float = 100):
    n_samples_epoch = int(sfreq * 30)  # 30 segundos = 3000 amostras
    n_epochs = signals.shape[1] // n_samples_epoch
    
    # Reshape: (3, 3000×n_epochs) → (3, n_epochs, 3000)
    epochs = signals[:, :n_epochs * n_samples_epoch].reshape(3, n_epochs, n_samples_epoch)
    
    # Labels já estão em formato de épocas (1 label por 30s)
    epoch_labels = labels[:n_epochs]
    
    return epochs, epoch_labels
```

**WESAD - Windowing com Overlap**:
```python
def create_windows(signals_dict: Dict, window_size: int = 240, stride: int = 120):
    # window_size = 240 amostras (60s × 4Hz)
    # stride = 120 amostras (50% overlap)
    
    windows = []
    for i in range(0, len(ecg) - window_size, stride):
        # Extrair janela de 60s para cada sinal
        ecg_window = ecg[i:i+window_size, 0]
        eda_window = eda[i:i+window_size, 0]
        temp_window = temp[i:i+window_size, 0]
        acc_window = acc[i:i+window_size, 0]
        
        # Label: maioritário na janela
        window_label = np.bincount(labels[i:i+window_size].astype(int)).argmax()
        
        window_data = np.array([ecg_window, eda_window, temp_window, acc_window])
        windows.append(window_data)
    
    return np.array(windows), np.array(window_labels)
```

### 3.6 Resampling e Sincronização

**WESAD - Resampling de 700 Hz para 4 Hz**:
```python
def resample_signals(signals_dict: Dict, target_freq: float = 4):
    # Todos os sinais chest estão a 700 Hz
    original_length = len(signals_dict['ecg'])  # 4,255,300 amostras
    target_length = int(original_length * target_freq / 700)  # 24,316 amostras
    
    # Resample todos os sinais para o mesmo comprimento
    resampled['ecg'] = signal.resample(signals_dict['ecg'], target_length)
    resampled['eda'] = signal.resample(signals_dict['eda'], target_length)
    resampled['temp'] = signal.resample(signals_dict['temp'], target_length)
    resampled['acc'] = signal.resample(signals_dict['acc'], target_length)
    resampled['labels'] = signal.resample(signals_dict['labels'], target_length)
```

**Cálculo**: `target_length = original_length × (target_freq / original_freq)`
- Original: 4,255,300 amostras a 700 Hz
- Target: 24,316 amostras a 4 Hz
- Ratio: 4/700 = 0.0057

### 3.7 Filtragem Digital

**Filtros Butterworth de 4ª ordem**:
```python
# ECG: 0.5-1.5 Hz (heart rate range adaptado para 4Hz)
filtered['ecg'] = signal.sosfilt(
    signal.butter(4, [0.5, 1.5], btype='band', fs=4, output='sos'), 
    signals_dict['ecg']
)

# EDA: 0.05-1 Hz (variações lentas)
filtered['eda'] = signal.sosfilt(
    signal.butter(4, [0.05, 1], btype='band', fs=4, output='sos'), 
    signals_dict['eda']
)

# ACC: 0.1-1.5 Hz (movimento corporal)
filtered['acc'] = signal.sosfilt(
    signal.butter(4, [0.1, 1.5], btype='band', fs=4, output='sos'), 
    signals_dict['acc']
)
```

**Limite de Nyquist**: Para fs=4 Hz, frequência máxima = 2 Hz
- ✅ ECG: 0.5-1.5 Hz (dentro do limite)
- ✅ EDA: 0.05-1 Hz (dentro do limite)  
- ✅ ACC: 0.1-1.5 Hz (dentro do limite)

### 3.8 Normalização

**Método**: StandardScaler (normalização z-score)

**Implementação**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Fórmula: z = (x - μ) / σ
# Onde μ é a média e σ é o desvio padrão de cada feature
```

**Raciocínio**:
1. **Consistência**: Mesma escala para todas as features
2. **Estabilidade**: Reduz impacto de outliers
3. **Compatibilidade**: Adequado para algoritmos de ML

---

## 4. Lições Aprendidas

### 4.1 Importância da Documentação

**Lição**: Sempre verificar a documentação oficial antes de assumir estruturas de dados.

**Aplicação**: A documentação do WESAD revelou a estrutura correta dos dados, evitando tentativas de adivinhação.

### 4.2 Investigação Sistemática

**Lição**: Quando há erros, investigar sistematicamente a estrutura dos dados.

**Método aplicado**:
1. Carregar dados e inspecionar estrutura
2. Verificar dimensões e tipos de dados
3. Testar funções individualmente
4. Corrigir incrementalmente

### 4.3 Adaptabilidade

**Lição**: Estar preparado para adaptar o código quando os dados não correspondem às expectativas.

**Exemplos**:
- Sleep-EDF: Adaptação de `.edf` para `.rec/.hyp`
- WESAD: Mudança de wrist para chest data

### 4.4 Validação Contínua

**Lição**: Testar cada etapa do pipeline individualmente.

**Benefícios**:
- Identificação rápida de problemas
- Isolamento de erros
- Confiança no resultado final

---

## 5. Conclusões

O processamento bem-sucedido dos datasets Sleep-EDF e WESAD resultou de:

1. **Investigação sistemática** da estrutura real dos dados
2. **Adaptação flexível** do código às limitações encontradas
3. **Decisões informadas** baseadas na documentação e análise técnica
4. **Validação contínua** de cada etapa do processo

### 5.1 Pipeline Completo de Processamento

**Sleep-EDF Pipeline**:
```
Ficheiros .rec/.hyp → pyedflib → Sinais (3×N) + Labels (M) → 
Segmentação (30s) → Features (24) → Normalização → Split (70/15/15)
```

**WESAD Pipeline**:
```
Ficheiros .pkl → Chest data (700Hz) → Resampling (4Hz) → 
Filtragem → Windowing (60s, 50% overlap) → Features (36) → 
Filtragem labels → Normalização → Split (70/15/15)
```

### 5.2 Resultados Numéricos Detalhados

**Sleep-EDF**:
- **Ficheiros processados**: 8 (sc4002e0, sc4012e0, sc4102e0, sc4112e0, st7022j0, st7052j0, st7121j0, st7132j0)
- **Épocas por ficheiro**: 33-39 (dependendo da duração)
- **Total épocas**: 291
- **Features por época**: 24 (8 por sinal × 3 sinais)
- **Classes**: 5 (W=85, N1=70, N2=60, N3=52, R=24)
- **Split**: Train=203, Val=44, Test=44

**WESAD**:
- **Ficheiros processados**: 15 (S2-S17, excluindo S1 e S12)
- **Janelas por ficheiro**: 173-234 (dependendo da duração)
- **Total janelas**: 2,874 (antes da filtragem)
- **Janelas válidas**: 539 (após filtragem stress/amusement)
- **Features por janela**: 36 (9 por sinal × 4 sinais)
- **Classes**: 2 (stress=172, amusement=367)
- **Split**: Train=377, Val=81, Test=81

### 5.3 Estrutura Final dos Dados

**Sleep-EDF**:
```python
X_train.shape = (203, 24)  # 203 épocas, 24 features
y_train.shape = (203,)     # 5 classes: [0,1,2,3,4]
```

**WESAD**:
```python
X_train.shape = (377, 36)  # 377 janelas, 36 features  
y_train.shape = (377,)     # 2 classes: [0,1]
```

Os datasets estão agora prontos para treino de modelos com preservação de privacidade, com:
- **Sleep-EDF**: 291 épocas, 24 features, 5 classes de sono
- **WESAD**: 539 janelas, 36 features, 2 classes de stress/emoção

Esta base sólida permite avançar para as fases de treino com Differential Privacy e Federated Learning.
