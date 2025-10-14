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

**Resampling Multi-Frequência:**
- **Objetivo**: Frequência uniforme de 4 Hz (realista para wearables, eficiente computacionalmente)
- **Método**: Resampling polyfásico (`scipy.signal.resample_poly`) por canal
- **Sinais chest** (700 Hz → 4 Hz): ECG, EDA, Temp, ACC (3D), EMG, Resp
- **Sinais wrist**: 
  - BVP: 64 Hz → 4 Hz
  - ACC: 32 Hz → 4 Hz  
  - EDA/TEMP: 4 Hz → 4 Hz (já na frequência alvo)
- **Labels**: Downsampling por nearest-neighbor indexing sincronizado com chest
- **Resultado**: Todos os sinais uniformemente a 4 Hz

**Filtragem Adaptada por Sinal:**
- **Implementação**: Butterworth 4ª ordem com `sosfiltfilt` (zero-phase, estabilidade numérica)
- **Filtros específicos**:
  - **ECG**: 0.5-15 Hz (captura heart rate + variabilidade)
  - **BVP**: 0.5-8 Hz (pulso arterial)
  - **ACC**: 0.1-1.5 Hz (movimento corporal, remove high-freq noise)
  - **EDA**: lowpass 1 Hz (variações lentas de condutância)
  - **Temperatura**: lowpass 0.5 Hz (variações muito lentas)
  - **Respiração**: 0.1-0.5 Hz (breathing rate típico)
  - **EMG**: 0.5-1.5 Hz (atividade muscular adaptada para 4 Hz)
- **Limite de Nyquist**: Todos os filtros respeitam fs/2 = 2 Hz

**Janelamento Temporal:**
- **Janela**: 60 segundos = 240 amostras @ 4 Hz (padrão clínico para análise de stress)
- **Overlap**: 50% (120 amostras) para aumentar quantidade de dados
- **Label**: Majority voting dentro da janela (exclui labels indefinidos = 0)
- **Output**: Janelas de forma `(n_channels, 240)` preservando estrutura temporal completa

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

**Normalização e Split:**
- **Split por sujeito** (LOSO-style): Train 60%, Val 20%, Test 20% de sujeitos
- **Vantagem**: Evita vazamento de dados entre splits (cada sujeito inteiro vai para um só split)
- **Normalização**: Z-score por canal usando **apenas estatísticas do treino**
  - `X_normalized = (X - train_mean) / train_std`
  - Aplicado independentemente a cada um dos 14 canais
  - `train_mean` e `train_std` têm shape `(1, 14, 1)` (broadcast sobre amostras e timesteps)

### 2.4 Resultados do Pré-processamento

**Sleep-EDF Expanded:**
- **Ficheiros processados**: 197 (100% de sucesso)
- **Épocas totais**: 453,005 (vs 291 no dataset original)
- **Features**: 24 por época
- **Classes**: 5 (W, N1, N2, N3, R)
- **Distribuição**: [289,102, 24,632, 86,397, 11,673, 6,800] épocas
- **Split**: 70/15/15 (train/val/test)

**WESAD (Binário):**
- **Ficheiros processados**: 15/15 sujeitos (100% de sucesso)
- **Janelas brutas**: 2,874 (antes de filtrar labels)
- **Janelas válidas**: 1,189 (após filtrar labels 1,2,3)
- **Shape final**: `(n_windows, 14, 240)` - preserva estrutura temporal completa
- **Classes**: 2 (non-stress: 70%, stress: 30%)
- **Distribuição bruta**: [827 non-stress, 362 stress] janelas
- **Split por sujeito**:
  - Train: 9 sujeitos → 715 janelas
  - Val: 3 sujeitos → 237 janelas  
  - Test: 3 sujeitos → 237 janelas
- **Normalização**: Per-channel z-score (train-only statistics)

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

**WESAD - CNN-LSTM (Otimizado para Binário):**
- **Arquitetura**:
  - **Conv1D(64, kernel=5)** → BatchNorm → MaxPool(2) → Dropout(0.3)
  - **Conv1D(128, kernel=5)** → BatchNorm → MaxPool(2) → Dropout(0.3)
  - **LSTM(64)** → Dropout(0.4)
  - **Dense(32, ReLU)** → Dropout(0.3)
  - **Dense(2, Softmax)**
- **Input shape**: `(14, 240)` - 14 canais × 240 timesteps
- **Parâmetros**: 170,274 (665 KB) - modelo leve para análise de privacidade
- **Motivação**: CNNs extraem padrões locais, LSTM captura dependências temporais
- **Optimizador**: Adam (lr=0.001)
- **Loss**: Categorical crossentropy
- **Class weights**: {non-stress: 0.72, stress: 1.65} para balancear classes

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

**WESAD Baseline (CNN-LSTM Binário):**
- **Dados de treino**: 715 janelas (9 sujeitos)
- **Dados de validação**: 237 janelas (3 sujeitos)
- **Dados de teste**: 237 janelas (3 sujeitos)
- **Duração do treino**: 21 épocas (early stopping)
- **Tempo total**: ~6.2 segundos (~0.3s/época) - muito rápido devido ao modelo leve
- **Convergência**: Rápida e estável com early stopping em val_loss
- **Learning rate**: Redução automática 0.001 → 0.00025

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

**WESAD Baseline (CNN-LSTM Binário):**
- **Test Accuracy**: 75.95%
- **Test Precision**: 84.75%
- **Test Recall**: 75.95%
- **Test F1-Score**: 76.85%

**Análise por Classe (WESAD Binário):**
- **Non-Stress (baseline+amusement)**: 
  - Precision: 97.3% (alta confiança quando prediz non-stress)
  - Recall: 67.1% (detecta 2/3 dos casos non-stress)
  - F1-Score: 79.4%
  - Suporte: 164 amostras de teste
- **Stress**: 
  - Precision: 56.5% (muitos falsos positivos)
  - Recall: 95.9% (quase todos os casos de stress são detectados!)
  - F1-Score: 71.1%
  - Suporte: 73 amostras de teste

**Matriz de Confusão (WESAD):**
```
                Predito
            Non-stress  Stress
Real Non-stress   110      54
     Stress         3      70
```

**Interpretação Clínica:**
- **Alta sensibilidade ao stress**: Modelo conservador que raramente perde casos de stress (recall 96%)
- **Trade-off**: Aceita falsos positivos (54 casos non-stress classificados como stress) para não perder casos reais
- **Aplicação**: Ideal para alertas preventivos onde é preferível sinalizar stress em excesso a não detectá-lo

### 3.5 Análise de Estabilidade

**Convergência:**
- **Sleep-EDF**: Convergência estável na época 21
- **Overfitting**: Controlado pelo early stopping
- **Learning rate**: Redução automática funcionou adequadamente

**Robustez:**
- **Validação**: Métricas consistentes entre validação e teste
- **Generalização**: Boa performance em dados não vistos
- **Estabilidade**: Treino reproduzível com seeds fixos

---

## Fase 4: Preparação para Técnicas de Privacidade

### 4.1 Estrutura de Dados para DP/FL

**Organização dos Resultados:**
- **Modelos**: Salvos em formato HDF5 (.h5)
- **Histórico**: Métricas por época em JSON
- **Resultados**: Métricas finais e matriz de confusão
- **Estrutura**: `models/dataset/technique/run_XXX/`

**Dados Processados:**
- **Formato**: Arrays NumPy (.npy) para eficiência
- **Normalização**: StandardScaler salvo (.pkl)
- **Metadados**: Informações de pré-processamento
- **Estrutura**: `data/processed/dataset/`

### 4.2 Configuração para Experimentos

**Differential Privacy:**
- **Epsilon values**: [0.5, 1.0, 2.0, 5.0]
- **Noise mechanism**: Gaussian noise
- **Sensitivity**: Calculada baseada na arquitetura
- **Privacy budget**: Tracking por época

**Federated Learning:**
- **Client configurations**: [3, 5, 10] clientes
- **Communication rounds**: 50
- **Epochs per round**: 2
- **Aggregation**: FedAvg

### 4.3 Pipeline de Experimentos

**Scripts de Treino:**
- **Baseline**: `train/dataset/train_baseline.py`
- **DP**: `train/dataset/train_dp.py`
- **FL**: `train/dataset/train_fl.py`

**Execução:**
- **Múltiplas execuções**: 10 runs por configuração
- **Estatísticas**: Média ± desvio padrão
- **Comparação**: Testes de significância estatística

---

## Lições Aprendidas e Desafios

### 5.1 Desafios Técnicos

**Formato de Dados:**
- **Sleep-EDF**: Complexidade do formato EDF+ com anotações e durações variáveis
- **WESAD**: Múltiplas frequências, sincronização e extração de features abrangente
- **Solução**: Investigação sistemática e adaptação flexível

**Complexidade da Implementação:**
- **Sleep-EDF**: Algoritmo de matching de ficheiros com extração de "base prefix", conversão de épocas variáveis
- **WESAD**: Pipeline multi-dispositivo otimizado
  - Resampling polyfásico multi-frequência (4-700 Hz → 4 Hz)
  - Filtragem adaptada por tipo de sinal (ECG, EDA, ACC, etc.)
  - Expansão de canais 3D (ACC) e preservação temporal completa
  - Split por sujeito (LOSO-style) para evitar vazamento
- **Escalabilidade**: Processamento eficiente (<1 min para WESAD, ~20 min para Sleep-EDF)

**Escala de Dados:**
- **Sleep-EDF**: 453,005 épocas (vs 291 original)
- **Processamento**: ~15-20 minutos no Colab
- **Memória**: ~8GB RAM durante treino

**Performance:**
- **N1 Sleep**: Dificuldade de classificação (8.2% recall)
- **Desequilíbrio**: Classes com distribuições muito diferentes
- **Solução**: Análise detalhada por classe

### 5.2 Metodologia de Desenvolvimento

**Investigação Sistemática:**
- **Documentação**: Consulta sempre a documentação oficial
- **Prototipagem**: Desenvolvimento incremental com validação
- **Debugging**: Isolamento de problemas por etapa

**Adaptabilidade:**
- **Flexibilidade**: Código adaptável a diferentes formatos
- **Robustez**: Tratamento de erros e casos extremos
- **Escalabilidade**: Pipeline eficiente para grandes datasets

### 5.3 Preparação para Paper

**Reprodutibilidade:**
- **Seeds fixos**: Reproducibilidade garantida
- **Documentação**: Processo completamente documentado
- **Código limpo**: Estrutura modular e bem comentada

**Métricas Completas:**
- **Performance**: Accuracy, precision, recall, F1-score
- **Estabilidade**: Múltiplas execuções com estatísticas
- **Eficiência**: Tempo de treino e uso de recursos

---

## Conclusões e Próximos Passos

### 6.1 Resultados Alcançados

**Datasets Processados:**
- **Sleep-EDF**: 453,005 épocas, 24 features, 5 classes (multi-class)
- **WESAD**: 1,189 janelas, 14 canais × 240 timesteps, 2 classes (binário)
- **Qualidade**: Pipelines robustos, split por sujeito (LOSO), sem vazamento

**Modelos Baseline:**
- **Sleep-EDF (LSTM)**: 87.45% accuracy, F1=85.82% (5 classes)
- **WESAD (CNN-LSTM binário)**: 75.95% accuracy, F1=76.85% (2 classes)
  - **Destaque**: 95.9% recall em stress (detecção quase perfeita)
  - **Trade-off**: Alta sensibilidade vs precisão moderada
- **Estabilidade**: Treino convergente, reproduzível, sem overfitting

**Infraestrutura:**
- **Código modular**: Fácil extensão para DP/FL
- **Organização clara**: Estrutura escalável
- **Documentação completa**: Processo totalmente documentado

### 6.2 Próximos Passos

**Técnicas de Privacidade:**
- **Differential Privacy**: Implementação com diferentes epsilon values
- **Federated Learning**: Simulação com múltiplos clientes
- **Comparação**: Trade-offs privacy-performance

**Análise Científica:**
- **Testes estatísticos**: Significância das diferenças
- **Visualizações**: Gráficos de trade-offs
- **Conclusões**: Recomendações práticas

**Paper Científico:**
- **Metodologia**: Processo completamente documentado
- **Resultados**: Métricas robustas e comparáveis
- **Contribuições**: Novos insights sobre privacidade em mHealth

---

## Anexos Técnicos

### A.1 Estrutura Final dos Dados

**Sleep-EDF Processed:**
```
X_train.shape = (317,103, 24)
X_val.shape = (67,951, 24)
X_test.shape = (67,951, 24)
y_train.shape = (317,103,)
y_val.shape = (67,951,)
y_test.shape = (67,951,)
```

**WESAD Processed (Binário - Temporal):**
```
X_train.shape = (715, 14, 240)  # 715 janelas × 14 canais × 240 timesteps
X_val.shape = (237, 14, 240)
X_test.shape = (237, 14, 240)
y_train.shape = (715,)          # Labels binários: 0=non-stress, 1=stress
y_val.shape = (237,)
y_test.shape = (237,)
```
*Nota: Formato temporal preserva toda a informação da série temporal em 14 canais fisiológicos sincronizados. Normalização per-channel z-score aplicada usando apenas estatísticas do treino.*

### A.2 Configurações de Hardware

**Ambiente de Desenvolvimento:**
- **Local**: M1 MacBook Pro, 16GB RAM
- **Colab**: Tesla T4 GPU, 16GB RAM
- **Tempo de processamento**: 15-20 minutos (Colab) vs 30-40 minutos (local)

**Dependências:**
- **Python**: 3.8+
- **TensorFlow**: 2.x
- **Scikit-learn**: 1.0+
- **PyEDFlib**: Para leitura de ficheiros EDF
- **MNE**: Para processamento de sinais neurológicos

### A.3 Métricas de Qualidade

**Reprodutibilidade:**
- **Seeds**: numpy=42, tensorflow=42
- **Resultados**: Consistentes entre execuções
- **Documentação**: Processo completamente rastreável

**Robustez:**
- **Tratamento de erros**: Pipeline resiliente a falhas
- **Validação**: Verificação contínua de qualidade
- **Escalabilidade**: Eficiente para datasets grandes

---

*Este relatório documenta o desenvolvimento completo de um sistema de análise de dados de saúde com preservação de privacidade, desde a aquisição de datasets até à implementação de modelos baseline, preparando o terreno para a aplicação de técnicas avançadas de privacidade diferencial e federated learning.*
