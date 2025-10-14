# Pré-processamento: Sleep-EDF e WESAD

## 1. SLEEP-EDF

### 1.1 Carregamento dos dados

Sleep-EDF vem em ficheiros `.rec` (gravações) e `.hyp` (hypnogramas) em formato EDF. Usa a biblioteca `mne`:

```python
import mne
from mne.io import read_raw_edf

# Carregar ficheiro de gravação
raw = read_raw_edf('ficheiro.rec')
print(raw.info)  # Ver canais disponíveis

# Carregar hypnograma (labels de sono)
hypno = read_raw_edf('ficheiro.hyp')
# Ou usar pyedflib para maior controlo
import pyedflib
f = pyedflib.EdfReader('ficheiro.hyp')
hypno_data = f.readSignal(0)  # Sinal do hypnograma
```

**Alternativa com pyedflib (mais direto):**

```python
import pyedflib
import numpy as np

# Ler sinais do ficheiro .rec
f = pyedflib.EdfReader('ficheiro.rec')
n_signals = f.signals_in_file

# Ver nomes dos sinais disponíveis
signal_labels = f.getSignalLabels()
print(signal_labels)  # Ver quais sinais existem

# Ler sinais específicos
eeg_fpz_cz = f.readSignal(0)  # Geralmente canal 0
eeg_pz_oz = f.readSignal(1)   # Geralmente canal 1
eog = f.readSignal(2)          # Geralmente canal 2
```

**Sinais disponíveis:**
- `Fpz-Cz`: EEG (prefrontal-central) — **USAR ESTE**
- `Pz-Oz`: EEG (parietal-occipital) — **USAR ESTE**
- `ROC-LOC`: EOG (electrooculograma) — **USAR ESTE**
- `EMG chin`: Eletromiograma (queixo) — opcional

**Labels (fases de sono):**
- W: Acordado
- N1, N2, N3: NREM (sono não-REM)
- R: REM (movimento rápido dos olhos)

### 1.2 Filtragem

Aplicar bandpass filter de 3ª ordem Butterworth com frequências de corte 0.5-32 Hz para EEG e 0.5-10 Hz para EOG.

```python
# EEG: 0.5-32 Hz
eeg_filtered = raw.filter(l_freq=0.5, h_freq=32.0, method='iir', 
                          iir_params=dict(order=3, ftype='butter'))

# EOG: 0.5-10 Hz
eog_filtered = raw.filter(l_freq=0.5, h_freq=10.0, method='iir',
                          iir_params=dict(order=3, ftype='butter'))

# EMG (opcional): 10-100 Hz
emg_filtered = raw.filter(l_freq=10.0, h_freq=100.0)
```

### 1.3 Segmentação em épocas

Sleep-EDF usa épocas de **30 segundos** (padrão clínico).

```python
# Frequência de amostragem típica: 100 Hz
# 30 segundos = 3000 amostras
sfreq = raw.info['sfreq']  # Verificar
epoch_duration = 30  # segundos
n_samples_epoch = int(sfreq * epoch_duration)

# Extrair sinais
eeg_fpz_cz = raw.get_data(picks='Fpz-Cz')[0]
eeg_pz_oz = raw.get_data(picks='Pz-Oz')[0]
eog = raw.get_data(picks='ROC-LOC')[0]

# Segmentar em épocas
n_epochs = len(eeg_fpz_cz) // n_samples_epoch
epochs_eeg_1 = [eeg_fpz_cz[i*n_samples_epoch:(i+1)*n_samples_epoch] 
                for i in range(n_epochs)]
epochs_eeg_2 = [eeg_pz_oz[i*n_samples_epoch:(i+1)*n_samples_epoch] 
                for i in range(n_epochs)]
epochs_eog = [eog[i*n_samples_epoch:(i+1)*n_samples_epoch] 
              for i in range(n_epochs)]
```

### 1.4 Extração de features (por época)

Para cada época de 30s, extrair características no domínio do tempo e frequência:

```python
from scipy import signal
import numpy as np

def extract_features_epoch(epoch, sfreq=100):
    # Domínio do tempo
    mean_val = np.mean(epoch)
    std_val = np.std(epoch)
    max_val = np.max(epoch)
    min_val = np.min(epoch)
    
    # Domínio da frequência (power spectral density)
    freqs, psd = signal.welch(epoch, sfreq, nperseg=256)
    
    # Bandas de frequência
    delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
    theta = np.mean(psd[(freqs >= 4) & (freqs < 8)])
    alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)])
    beta = np.mean(psd[(freqs >= 13) & (freqs < 30)])
    
    return [mean_val, std_val, max_val, min_val, delta, theta, alpha, beta]

# Para cada época, extrair features de ambos EEG + EOG
features = []
labels = []
for i in range(n_epochs):
    feat_eeg1 = extract_features_epoch(epochs_eeg_1[i])
    feat_eeg2 = extract_features_epoch(epochs_eeg_2[i])
    feat_eog = extract_features_epoch(epochs_eog[i])
    features.append(feat_eeg1 + feat_eeg2 + feat_eog)  # 24 features total
```

### 1.5 Normalização

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
```

### 1.6 Split treino/validação/teste

```python
from sklearn.model_selection import train_test_split

# 70% treino, 15% validação, 15% teste
X_train, X_temp, y_train, y_temp = train_test_split(
    features_normalized, labels, test_size=0.3, random_state=42, stratify=labels)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```

---

## 2. WESAD

### 2.1 Carregamento dos dados

WESAD vem em ficheiros `.pkl` (pickle). Cada ficheiro contém dados de um participante.

```python
import pickle
import pandas as pd

# Carregar ficheiro de um participante
with open('S2.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Estrutura típica: data['signal'][device][signal_type]
# Devices: 0=chest (RespiBAN), 1=wrist (Empatica E4)
# Signal types: 0=ECG, 1=EDA, 2=EMG, 3=Temp, 4=Resp, 5=Acc
```

### 2.2 Extração de sinais

O RespiBAN (chest) fornece: ECG, EDA, EMG, respiração, temperatura e aceleração em 700 Hz. O Empatica E4 (wrist) fornece: BVP (64 Hz), EDA (4 Hz), temperatura (4 Hz) e aceleração (32 Hz).

**Recomendação**: Usar dados do **Empatica E4 (wrist)** porque tem taxa de amostragem mais baixa (mais realista para dispositivos móveis):

```python
# Extrair sinais do wrist (device=1)
ecg_bvp = data['signal'][1][0]  # BVP (blood volume pulse, substitui ECG)
eda = data['signal'][1][1]       # EDA
temp = data['signal'][1][2]      # Temperatura
acc = data['signal'][1][3]       # Aceleração

# Extrair labels
# data['label']: Array com (timestamp, label) para cada amostra
# Labels: 0=baseline, 1=stress, 2=amusement, 3=transition
labels_raw = data['label']
labels = labels_raw[:, 1]  # Extrair só o label (coluna 1)
```

### 2.3 Downsample e sincronização de frequências

Como os sinais têm diferentes frequências de amostragem (BVP 64Hz, EDA 4Hz, Temp 4Hz), precisas de resample para a mesma frequência.

```python
from scipy import signal as sig

# Resample tudo para 4 Hz (frequência mais baixa, mais comum em mobile)
target_freq = 4
sfreq_bvp = 64
sfreq_eda = 4
sfreq_temp = 4

# BVP precisa de downsample
n_samples_target = int(len(ecg_bvp) * target_freq / sfreq_bvp)
bvp_resampled = sig.resample(ecg_bvp, n_samples_target)
eda_resampled = eda[:n_samples_target]  # Já está a 4 Hz
temp_resampled = temp[:n_samples_target]  # Já está a 4 Hz

# Agora todos têm o mesmo comprimento e frequência de amostragem
labels_resampled = labels[:n_samples_target]
```

### 2.4 Filtragem

```python
# BVP/ECG: 5-15 Hz (coração bate 60-100 bpm)
bvp_filtered = sig.butter(order=4, Wn=[5, 15], fs=4, btype='band')
bvp_filt = sig.sosfilt(bvp_filtered, bvp_resampled)

# EDA: 0.05-1 Hz (varia lentamente com stress)
eda_filtered = sig.butter(order=4, Wn=[0.05, 1], fs=4, btype='band')
eda_filt = sig.sosfilt(eda_filtered, eda_resampled)

# Temperatura: sem filtragem rigorosa (varia muito lentamente)
temp_filt = temp_resampled
```

### 2.5 Segmentação em janelas

Com 4 Hz, criar janelas de ~60 segundos (240 amostras):

```python
window_size = 240  # 60 segundos a 4 Hz
stride = 120      # Overlap de 50%

windows_bvp = []
windows_eda = []
windows_temp = []
windows_labels = []

for i in range(0, len(bvp_filt) - window_size, stride):
    windows_bvp.append(bvp_filt[i:i+window_size])
    windows_eda.append(eda_filt[i:i+window_size])
    windows_temp.append(temp_filt[i:i+window_size])
    # Label: maioritário na janela
    window_label = np.bincount(labels_resampled[i:i+window_size]).argmax()
    windows_labels.append(window_label)
```

### 2.6 Extração de features (por janela)

```python
def extract_features_window(signal_data, sfreq=4):
    # Domínio do tempo
    mean_val = np.mean(signal_data)
    std_val = np.std(signal_data)
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    
    # Domínio da frequência
    freqs, psd = signal.welch(signal_data, sfreq, nperseg=min(len(signal_data), 64))
    
    return [mean_val, std_val, min_val, max_val] + list(psd[:5])  # 9 features

# Extrair features para cada janela
features = []
for i in range(len(windows_bvp)):
    feat_bvp = extract_features_window(windows_bvp[i])
    feat_eda = extract_features_window(windows_eda[i])
    feat_temp = extract_features_window(windows_temp[i])
    features.append(feat_bvp + feat_eda + feat_temp)  # 27 features total

features = np.array(features)
labels = np.array(windows_labels)
```

### 2.7 Remover transições e baseline

WESAD tem labels: 0=baseline, 1=stress, 2=amusement, 3=transition. Remove transições:

```python
# Manter apenas: stress (1) e amusement (2), relabel como binary (0 vs 1)
valid_idx = (labels != 0) & (labels != 3)  # Remove baseline e transition
features_filtered = features[valid_idx]
labels_filtered = labels[valid_idx] - 1  # Relabel: 1→0 (stress), 2→1 (amusement)
```

### 2.8 Normalização

```python
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features_filtered)
```

### 2.9 Split treino/validação/teste

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    features_normalized, labels_filtered, test_size=0.3, random_state=42, stratify=labels_filtered)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```

---

## Resumo Comparativo

| Passo | Sleep-EDF | WESAD |
|-------|-----------|-------|
| **Entrada** | Ficheiros `.edf` (EEG, EOG) | Ficheiros `.pkl` por participante |
| **Filtragem** | 0.5-32 Hz (EEG), 0.5-10 Hz (EOG) | 5-15 Hz (BVP), 0.05-1 Hz (EDA) |
| **Frequência amostragem** | ~100 Hz | 4 Hz (após resample) |
| **Segmentação** | 30s épocas | 60s janelas com 50% overlap |
| **Features por época/janela** | 24 (8 por sinal × 3 sinais) | 27 (9 por sinal × 3 sinais) |
| **Labels** | 5 classes (W, N1, N2, N3, R) | 2 classes (stress vs. amusement) |
| **Normalização** | StandardScaler | StandardScaler |

---
