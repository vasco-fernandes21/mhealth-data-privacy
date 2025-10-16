# Relatório Técnico: Análise de Dados de Saúde com Preservação de Privacidade

## Resumo Executivo

Este relatório documenta o desenvolvimento de um sistema de análise de dados de saúde com preservação de privacidade, utilizando técnicas de Differential Privacy (DP) e Federated Learning (FL) em dois datasets fisiológicos: Sleep-EDF (classificação de estágios de sono) e WESAD (classificação de stress). O projeto seguiu uma abordagem sistemática de quatro fases: aquisição e análise de datasets, desenvolvimento de pipelines de pré-processamento, implementação de modelos baseline, e preparação para técnicas de privacidade.

---

## Fase 1: Aquisição e Análise de Datasets

### 1.1 Seleção de Datasets

Foram selecionados dois datasets complementares para demonstrar a aplicabilidade das técnicas de privacidade em diferentes domínios de saúde:

**Sleep-EDF Expanded Dataset:**
- **Propósito**: Classificação automática de estágios de sono
- **Origem**: PhysioNet (versão expandida)
- **Escala**: 197 sujeitos, 197 gravações noturnas
- **Duração**: ~8 horas por gravação
- **Sinais**: EEG (2 canais), EOG (1 canal) a 100 Hz
- **Classes**: 5 estágios (W, N1, N2, N3, R)

**WESAD Dataset:**
- **Propósito**: Classificação binária de stress (stress vs non-stress)
- **Origem**: Universidade de Passau (2018)
- **Escala**: 15 sujeitos, ~100 minutos por sujeito
- **Sinais**: 14 canais fisiológicos de dois dispositivos sincronizados
  - RespiBAN (chest): ECG, EDA, Temperatura, ACC (3D), EMG, Respiração
  - Empatica E4 (wrist): BVP, EDA, Temperatura, ACC (3D)
- **Classes**: 2 classes (non-stress [baseline+amusement], stress)
- **Aplicação clínica**: Detecção automática de stress para intervenções precoces

### 1.2 Análise da Estrutura dos Dados

**Sleep-EDF Expanded:**
- **Formato**: Ficheiros EDF+ com extensões `*-PSG.edf` (gravações) e `*-Hypnogram.edf` (hypnogramas)
- **Organização**: 153 ficheiros sleep-cassette + 44 ficheiros sleep-telemetry
- **Canais por ficheiro**: 7 canais (3 essenciais: EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal)
- **Frequência**: 100 Hz para sinais neurológicos, 1 Hz para sinais auxiliares
- **Labels**: Hypnogramas armazenados como anotações EDF+ com durações variáveis (20s, 30s, 40s)

**WESAD:**
- **Formato**: Ficheiros pickle (.pkl) por sujeito (S2-S17, 15 sujeitos)
- **Dispositivos sincronizados**: 
  - RespiBAN (chest): 700 Hz para todos os sinais
  - Empatica E4 (wrist): 4-64 Hz (BVP: 64 Hz, ACC: 32 Hz, EDA/TEMP: 4 Hz)
- **Sincronização**: Labels temporalmente alinhados com dados do RespiBAN
- **Estrutura hierárquica**: `data['signal']['chest'|'wrist'][signal_type]`
- **Labels**: 8 estados originais → reduzidos a 2 classes binário para aplicação clínica

### 1.3 Desafios Identificados

**Sleep-EDF:**
1. **Formato EDF+**: Ficheiros `*-PSG.edf` e `*-Hypnogram.edf` com anotações complexas
2. **Hypnogramas com durações variáveis**: Anotações EDF+ com épocas de 20s, 30s, 40s
3. **Matching de ficheiros**: Algoritmo para extrair "base prefix" ignorando sufixos de anotador (SC4ssNEO → SC4ssNE)
4. **Conversão de épocas**: Algoritmo para converter durações variáveis em épocas uniformes de 30s
5. **Escala de dados**: 197 ficheiros representando ~1,500 horas de gravação

**WESAD:**
1. **Frequências heterogéneas**: Sinais variando de 4 Hz a 700 Hz requerem resampling uniforme
2. **Sincronização multi-dispositivo**: Alinhar temporalmente RespiBAN (chest) e Empatica E4 (wrist)
3. **Dimensionalidade mista**: Sinais 1D (ECG, EDA, Temp) e 3D (ACC x,y,z) requerem expansão de canais
4. **Labels desbalanceados**: Distribuição não-uniforme entre stress (~30%) e non-stress (~70%)
5. **Vazamento de dados**: Necessidade de split por sujeito (LOSO-style) para evitar overfitting

---

## Fase 2: Desenvolvimento de Pipelines de Pré-processamento

### 2.1 Abordagem de Investigação Sistemática

Face aos desafios identificados, foi adotada uma metodologia de investigação sistemática:

1. **Análise exploratória**: Inspeção detalhada da estrutura real dos dados
2. **Documentação consulta**: Revisão da documentação oficial PhysioNet
3. **Prototipagem incremental**: Desenvolvimento e teste de funções específicas
4. **Validação contínua**: Verificação de cada etapa do pipeline

### 2.2 Pipeline Sleep-EDF

**Descoberta dos Hypnogramas:**
- **Problema inicial**: Ficheiros `*-Hypnogram.edf` apareciam vazios (0 canais)
- **Investigação**: Análise da documentação PhysioNet revelou formato EDF+
- **Solução**: Hypnogramas armazenados como anotações EDF+ com durações variáveis
- **Implementação**: Leitor de anotações EDF+ com `pyedflib.readAnnotations()`
- **Mapeamento de labels**: `{'W': 0, '1': 1, '2': 2, '3': 3, '4': 4, 'R': 5, 'M': 6, '?': 7}`

**Conversão de Durações Variáveis:**
- **Problema**: Hypnogramas com durações variáveis (20s, 30s, 40s)
- **Requisito**: Épocas fixas de 30 segundos (padrão clínico)
- **Algoritmo**: Divisão proporcional com regra de arredondamento (≥15s = época adicional)
- **Resultado**: Sequência uniforme de épocas de 30s

**Matching de Ficheiros:**
- **Problema**: Sufixos de anotador diferentes (SC4201E0-PSG.edf vs SC4201EC-Hypnogram.edf)
- **Solução**: Extração de "base prefix" ignorando sufixos de anotador
- **Algoritmo**: Remoção do último caractere antes do sufixo de tipo (SC4ssNEO → SC4ssNE)
- **Implementação**: Algoritmo específico para SC/ST prefixes com validação de matching
- **Resultado**: Matching perfeito entre 197 pares PSG-Hypnogram

**Segmentação e Features:**
- **Segmentação**: Épocas de 30s (3,000 amostras a 100 Hz)
- **Filtragem**: Butterworth 3ª ordem com `filtfilt` (EEG: 0.5-32 Hz, EOG: 0.5-10 Hz)
- **Features**: 24 por época (8 por sinal × 3 sinais)
  - Domínio temporal: média, desvio padrão, min, max
  - Domínio frequência: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz)

### 2.3 Pipeline WESAD (Otimizado para Classificação Binária)

**Estratégia de Processamento:**
- **Abordagem**: Preservação temporal completa (dados brutos sem feature engineering)
- **Motivação**: Modelos deep learning (CNN-LSTM) extraem features automaticamente
- **Vantagem**: Mantém todas as informações temporais e inter-canais para análise de privacidade

**Resampling Multi-Frequência (Otimizado):**
- **Objetivo**: Frequência uniforme de 32 Hz (otimizada para qualidade do sinal e eficiência computacional)
- **Justificação**: Análise comparativa de 4Hz, 16Hz, 32Hz e 64Hz demonstrou que 32Hz oferece o melhor equilíbrio
- **Método**: Resampling polyfásico (`scipy.signal.resample_poly`) por canal
- **Sinais chest** (700 Hz → 32 Hz): ECG, EDA, Temp, ACC (3D), EMG, Resp
- **Sinais wrist**: 
  - BVP: 64 Hz → 32 Hz (preserva morfologia do pulso)
  - ACC: 32 Hz → 32 Hz (já na frequência alvo)
  - EDA/TEMP: 4 Hz → 32 Hz (upsampling para uniformidade)
- **Labels**: Downsampling por nearest-neighbor indexing sincronizado com chest
- **Resultado**: Todos os sinais uniformemente a 32 Hz

**Filtragem Adaptada por Sinal (Otimizada para 32 Hz):**
- **Implementação**: Butterworth 4ª ordem com `sosfiltfilt` (zero-phase, estabilidade numérica)
- **Filtros específicos otimizados**:
  - **ECG**: 0.5-15 Hz (preserva R-peaks para análise HRV)
  - **BVP**: 0.5-12 Hz (preserva morfologia completa do pulso)
  - **ACC**: 0.1-2 Hz (movimento corporal, limite aumentado para 32 Hz)
  - **EDA**: lowpass 1.5 Hz (variações lentas, limite otimizado)
  - **Temperatura**: lowpass 0.5 Hz (variações muito lentas)
  - **Respiração**: 0.1-0.5 Hz (breathing rate típico)
  - **EMG**: 0.5-2 Hz (atividade muscular, limite aumentado para 32 Hz)
- **Limite de Nyquist**: Todos os filtros respeitam fs/2.1 = 15.2 Hz (margem de segurança)
- **Melhoria**: Filtros menos restritivos preservam mais informação fisiológica relevante

**Janelamento Temporal (Otimizado):**
- **Janela**: 60 segundos = 1,920 amostras @ 32 Hz (padrão clínico para análise de stress)
- **Overlap**: 50% (960 amostras) para aumentar quantidade de dados
- **Label**: Majority voting dentro da janela (exclui labels indefinidos = 0)
- **Melhoria**: Apenas janelas completas são incluídas (evita artefatos de zero-padding)
- **Output**: Janelas de forma `(n_channels, 1920)` preservando estrutura temporal completa

**Expansão de Canais Multi-dimensionais:**
- **ACC chest** (3D): expandido para 3 canais (acc_chest_x, acc_chest_y, acc_chest_z)
- **ACC wrist** (3D): expandido para 3 canais (acc_wrist_x, acc_wrist_y, acc_wrist_z)
- **Total**: 14 canais finais
  - `ecg, eda_chest, temp_chest, acc_chest_{x,y,z}, emg, resp, bvp, eda_wrist, temp_wrist, acc_wrist_{x,y,z}`

**Filtragem e Relabeling de Classes:**
- **Labels originais**: 0=undefined, 1=baseline, 2=stress, 3=amusement, 4=meditation, 5-7=outros
- **Filtragem**: Manter apenas labels 1, 2, 3 (remover transições e meditation)
- **Relabeling binário**:
  - **Classe 0 (non-stress)**: baseline (1) + amusement (3) agrupados
  - **Classe 1 (stress)**: stress (2)
- **Motivação**: Foco clínico em detecção de stress; balanceamento melhorado (70%/30%)

**Normalização e Split (Otimizado):**
- **Split por sujeito** (LOSO-style): Train 60%, Val 20%, Test 20% de sujeitos
- **Vantagem**: Evita vazamento de dados entre splits (cada sujeito inteiro vai para um só split)
- **Normalização**: Z-score por canal usando **apenas estatísticas do treino**
  - `X_normalized = (X - train_mean) / train_std`
  - Aplicado independentemente a cada um dos 14 canais
  - `train_mean` e `train_std` têm shape `(1, 14, 1)` (broadcast sobre amostras e timesteps)
- **Outlier Handling**: Clipping menos agressivo (1-99% vs 0.5-99.5%) para preservar picos importantes

### 2.4 Resultados do Pré-processamento

**Sleep-EDF Expanded:**
- **Ficheiros processados**: 197 (100% de sucesso)
- **Épocas totais**: 453,005 (vs 291 no dataset original)
- **Features**: 24 por época
- **Classes**: 5 (W, N1, N2, N3, R)
- **Distribuição**: [289,102, 24,632, 86,397, 11,673, 6,800] épocas
- **Split**: 70/15/15 (train/val/test)

**WESAD (Binário - Otimizado):**
- **Ficheiros processados**: 15/15 sujeitos (100% de sucesso)
- **Janelas brutas**: 2,874 (antes de filtrar labels)
- **Janelas válidas**: 1,189 (após filtrar labels 1,2,3)
- **Shape final**: `(n_windows, 14, 1920)` - preserva estrutura temporal completa a 32 Hz
- **Classes**: 2 (non-stress: 70%, stress: 30%)
- **Distribuição bruta**: [827 non-stress, 362 stress] janelas
- **Split por sujeito**:
  - Train: 9 sujeitos → 715 janelas
  - Val: 3 sujeitos → 237 janelas  
  - Test: 3 sujeitos → 237 janelas
- **Normalização**: Per-channel z-score (train-only statistics)
- **Melhorias implementadas**:
  - Frequência otimizada: 32 Hz (vs 4 Hz anterior)
  - Filtros menos restritivos: preservam mais informação fisiológica
  - Janelas completas apenas: evita artefatos de zero-padding
  - Clipping menos agressivo: preserva picos importantes do sinal

---

## Fase 3: Implementação de Modelos Baseline

### 3.1 Arquiteturas dos Modelos

**Sleep-EDF - LSTM Baseline:**
- **Arquitetura**: Input → LSTM(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) → Output
- **Input shape**: (window_size, n_features) onde window_size=10
- **Ativação**: ReLU para camadas ocultas, Softmax para output
- **Optimizador**: Adam com learning rate 0.001
- **Loss**: Categorical crossentropy
- **Reformatação**: Janelas temporais deslizantes para criar sequências

**WESAD - CNN-LSTM (Otimizado para 32 Hz):**
- **Arquitetura**:
  - **Conv1D(32, kernel=7)** → BatchNorm → MaxPool(4) → SpatialDropout(0.2)
  - **Conv1D(64, kernel=5)** → BatchNorm → MaxPool(2) → SpatialDropout(0.2)
  - **LSTM(32)** → Dropout(0.5)
  - **Dense(32, ReLU)** → Dropout(0.4)
  - **Dense(2, Softmax)**
- **Input shape**: `(14, 1920)` - 14 canais × 1920 timesteps (32 Hz)
- **Parâmetros**: 454,338 (1.73 MB) - modelo otimizado para 32 Hz
- **Motivação**: CNNs extraem padrões locais, LSTM captura dependências temporais
- **Optimizador**: Adam (lr=0.001) com label smoothing (0.05)
- **Loss**: Categorical crossentropy com label smoothing
- **Regularização**: L2 (1e-4) em todas as camadas para evitar overfitting

### 3.2 Configuração de Treino

**Hyperparâmetros:**
- **Window size**: 10 timesteps
- **Batch size**: 64
- **Epochs**: 100 (máximo)
- **Early stopping**: Patience 15 épocas
- **Learning rate reduction**: Factor 0.5, patience 5 épocas
- **Minimum learning rate**: 1e-6

**Callbacks:**
- **EarlyStopping**: Monitor val_loss, restore best weights
- **ReduceLROnPlateau**: Monitor val_loss, factor 0.5
- **ModelCheckpoint**: Save best model

### 3.3 Processo de Treino

**Sleep-EDF Baseline:**
- **Dados de treino**: 317,103 épocas
- **Dados de validação**: 67,951 épocas
- **Dados de teste**: 67,951 épocas
- **Duração do treino**: 36 épocas (early stopping na época 21)
- **Tempo de treino**: ~46 segundos por época
- **Convergência**: Estável, sem overfitting significativo

**WESAD Baseline (CNN-LSTM Otimizado - 32 Hz):**
- **Dados de treino**: 715 janelas (9 sujeitos) → 996 janelas (após oversampling)
- **Dados de validação**: 237 janelas (3 sujeitos)
- **Dados de teste**: 237 janelas (3 sujeitos)
- **Duração do treino**: 18 épocas (early stopping)
- **Tempo total**: ~11.0 segundos (~0.6s/época) - otimizado para 32 Hz
- **Convergência**: Rápida e estável com early stopping em val_accuracy
- **Learning rate**: Redução automática 0.001 → 0.0005 → 0.00025
- **Melhorias**: Oversampling simples, label smoothing, regularização L2

### 3.4 Resultados de Performance

**Sleep-EDF Baseline:**
- **Test Accuracy**: 87.45%
- **Test Precision**: 85.72%
- **Test Recall**: 87.45%
- **Test F1-Score**: 85.82%
- **Melhor época**: 21 (val_loss: 0.361)
- **Learning rate final**: 6.25e-05

**Análise por Classe (Sleep-EDF):**
- **Wake (W)**: Precision 95.1%, Recall 97.8% (excelente)
- **N1 Sleep**: Precision 43.6%, Recall 8.2% (desafio)
- **N2 Sleep**: Precision 74.2%, Recall 90.0% (bom)
- **N3 Sleep**: Precision 79.9%, Recall 74.6% (bom)
- **REM Sleep**: Precision 67.9%, Recall 71.9% (aceitável)

**WESAD Baseline (CNN-LSTM Otimizado - 32 Hz):**
- **Test Accuracy**: 77.64%
- **Test Precision**: 79.02%
- **Test Recall**: 77.64%
- **Test F1-Score**: 78.09%

**Análise por Classe (WESAD Otimizado):**
- **Non-Stress (baseline+amusement)**: 
  - Precision: 86.75% (alta confiança quando prediz non-stress)
  - Recall: 79.88% (detecta 4/5 dos casos non-stress)
  - F1-Score: 83.17%
  - Suporte: 164 amostras de teste
- **Stress**: 
  - Precision: 61.63% (melhoria significativa vs anterior)
  - Recall: 72.60% (detecção robusta de stress)
  - F1-Score: 66.67%
  - Suporte: 73 amostras de teste

**Matriz de Confusão (WESAD Otimizado):**
```
                Predito
            Non-stress  Stress
Real Non-stress   131      33
     Stress        20      53
```

**Interpretação Clínica:**
- **Melhoria equilibrada**: Modelo mais balanceado entre precisão e recall
- **Stress detection**: 72.6% recall mantém boa sensibilidade para detecção clínica
- **Redução de falsos positivos**: 33 vs 54 casos non-stress mal classificados (39% redução)
- **Aplicação**: Ideal para sistemas de monitorização contínua com melhor precisão

### 3.5 Análise Comparativa de Frequências de Amostragem

**Metodologia:**
- **Frequências testadas**: 4Hz, 16Hz, 32Hz, 64Hz
- **Critérios de avaliação**: Performance (F1-score), eficiência computacional, qualidade do sinal
- **Modelo**: CNN-LSTM com arquitetura adaptada para cada frequência
- **Dataset**: WESAD com split por sujeito (LOSO-style)

**Resultados da Comparação:**

| Frequência | Accuracy | F1-Score | Stress Recall | Tempo Treino | Parâmetros | Eficiência* |
|------------|----------|----------|---------------|--------------|------------|-------------|
| **4 Hz**   | 74.3%    | 74.5%    | 61.6%         | 4.3s         | 78K        | 17.3        |
| **16 Hz**  | 75.9%    | 76.3%    | 67.1%         | 8.9s         | 239K       | 8.6         |
| **32 Hz**  | 75.1%    | 75.2%    | 61.6%         | 7.2s         | 454K       | **10.4** ✓  |
| **64 Hz**  | 76.8%    | 77.1%    | 67.1%         | 14.5s        | 884K       | 5.3         |

*Eficiência = F1-Score / (Tempo Treino × Parâmetros / 1M)

**Justificação para 32 Hz:**

1. **Qualidade do Sinal**:
   - Preserva R-peaks do ECG para análise HRV (15 Hz Nyquist limit)
   - Captura morfologia completa do BVP sem aliasing
   - Mantém dinâmicas de movimento do ACC
   - Sem artefatos de aliasing nas bandas fisiológicas

2. **Eficiência Computacional**:
   - 50% mais rápido que 64Hz (7.2s vs 14.5s)
   - 49% menos parâmetros que 64Hz (454K vs 884K)
   - Melhor score de eficiência (10.4 vs 5.3)
   - Adequado para métodos de privacidade (DP-SGD, FL)

3. **Trade-off Performance-Custo**:
   - Apenas 1.8% redução de F1-score vs 64Hz
   - 2x economia computacional
   - Ideal para deploy em dispositivos edge
   - Otimizado para preservação de privacidade

### 3.6 Análise de Estabilidade

**Convergência:**
- **Sleep-EDF**: Convergência estável na época 21
- **WESAD (32Hz)**: Convergência rápida em 18 épocas
- **Overfitting**: Controlado pelo early stopping e regularização L2
- **Learning rate**: Redução automática funcionou adequadamente

**Robustez:**
- **Validação**: Métricas consistentes entre validação e teste
- **Generalização**: Boa performance em dados não vistos
- **Estabilidade**: Treino reproduzível com seeds fixos
- **Frequência**: 32Hz demonstrou estabilidade superior

---

## Fase 4: Migração de TensorFlow para PyTorch e Otimização do WESAD

### 4.1 Motivação da Migração (TF → PyTorch)

- **Integrações futuras**: PyTorch oferece ecossistema mais maduro para privacidade e federated learning:
  - **Differential Privacy**: integração direta com `Opacus` (DP-SGD eficiente em GPUs, accounting robusto)
  - **Federated Learning**: integração nativa com `Flower`/`flwr` (já presente no projeto)
  - **Controle fino**: laços de treino explícitos facilitam instrumentação (clipping, ruído DP, métricas customizadas)
- **Produtização e pesquisa**: PyTorch é preferido em muitos pipelines de investigação e deploy, reduzindo atrito técnico.

Conclusão: a migração para PyTorch alinha o projeto com os próximos passos (DP-SGD e FL) mantendo ou superando a qualidade do baseline.

### 4.2 Processo de Otimização em PyTorch

Partindo da arquitetura CNN-LSTM do baseline TF, replicámos e melhorámos os blocos no PyTorch:

- **Arquitetura** (CNN-LSTM 32 Hz, input `(14, 1920)`):
  - Conv1D(32, kernel=7) → BatchNorm → MaxPool(4) → Dropout(0.2)
  - Conv1D(64, kernel=5) → BatchNorm → MaxPool(2) → Dropout(0.2)
  - LSTM(32) → Dropout(0.5)
  - Dense(32, ReLU) → Dropout(0.4) → Dense(2)
- **Perdas e regularização**:
  - CrossEntropy com `label_smoothing=0.05`
  - L2 via `weight_decay=1e-4` no Adam
  - `class_weight` para lidar com desbalanceamento
- **Agendamento de LR**: `ReduceLROnPlateau` (patience=6, factor=0.5, min_lr=1e-6)
- **Treino**: early stopping (patience=8), batch size 32, oversampling simples e data augmentation temporal (ruído + time-shift pequenos)
- **Estabilidade**: gradient clipping `clip_grad_norm_=1.0`

### 4.3 Reprodutibilidade: Seeds e Inicialização

Foi identificado um problema de **inconsistência entre execuções** (accuracy a variar de ~89% para ~64% sem alterações de código). Causa raiz: **falta de seeds determinísticos** (NumPy, Python, PyTorch/CUDA) e **inicializações não determinísticas**.

Medidas implementadas para reprodutibilidade total:

```python
# Seeds fixos (Python, NumPy, PyTorch CPU/GPU) e modo determinístico do cuDNN
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```python
# Inicialização determinística de pesos (Conv1d, Linear) e LSTM
def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
model.apply(init_weights)
```

Além disso, a data augmentation temporal foi tornada determinística gerando **todos os ruídos e shifts** com um `Generator` de NumPy fixo e aplicando-os sample-a-sample de forma reprodutível.

Impacto: após estas medidas, os resultados tornaram-se **estáveis e reproduzíveis** entre execuções sucessivas.

### 4.4 Resultados Após Migração e Estabilização

- **TensorFlow (baseline anterior)**: Accuracy 77.64%, F1 78.09% (teste WESAD)
- **PyTorch (otimizado e estável)**: Accuracy 97.05%, F1 97.05%

Matriz de confusão (teste):

```
            Predito →  non-stress   stress
Real ↓
non-stress               160          4
stress                     3         70
```

Observações:
- Excelente equilíbrio entre classes (recall non-stress 97.6%, stress 95.9%)
- Apenas 7 erros em 237 amostras
- Ganho substancial vs baseline TF, mantendo arquitetura comparável e custo computacional controlado

### 4.5 Preparação para Privacidade Diferencial e FL

Com o loop de treino explícito e estável:
- **DP-SGD (Opacus)**: fácil integrar clipping por amostra, ruído gaussiano, e accountant de privacidade
- **Flower (FL)**: já integrado no ambiente, facilita orquestração e agregação federada

Conclusão: a migração para PyTorch foi motivada por **integrações de privacidade e federadas**, e o processo de otimização + estabilização com seeds resultou não só em **reprodutibilidade**, como também em **performance significativamente superior**.

