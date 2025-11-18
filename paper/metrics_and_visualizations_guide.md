# Métricas e Visualizações para Paper IEEE - mHealth Data Privacy

## 1. Métricas Principais (Core Metrics)

### 1.1 Métricas de Utilidade (Utility Metrics)

#### **Accuracy (Acurácia)**
- **O que medir**: Accuracy média, std, min, max (3 runs)
- **Como reportar**: 
  - Tabela comparativa: Baseline vs DP vs FL vs DP+FL
  - Por dataset (Sleep-EDF, WESAD)
  - Por nível de ruído (DP: σ=0.6, 1.0, 2.0)
  - Por número de clientes (FL: 3, 5, 10)
- **Formato IEEE**: `Accuracy = 85.53% ± 0.12%` (mean ± std)

#### **F1-Score (Weighted)**
- **O que medir**: F1-score médio, std (3 runs)
- **Por quê**: Melhor para datasets desbalanceados (WESAD tem 70% non-stress, 30% stress)
- **Como reportar**: Tabela e gráficos de barras com error bars

#### **Precision e Recall (Per-Class)**
- **O que medir**: Precision e Recall por classe
- **Por quê**: 
  - Sleep-EDF: 5 classes (W, N1, N2, N3, R) - identificar classes problemáticas
  - WESAD: 2 classes (non-stress, stress) - importante para detecção de stress
- **Como reportar**: 
  - Heatmaps de precision/recall por classe
  - Tabelas detalhadas para análise de confusão

#### **Confusion Matrix**
- **O que medir**: Matriz de confusão completa
- **Como reportar**: 
  - Heatmaps normalizados (percentagens)
  - Uma por método (Baseline, DP, FL, DP+FL)
  - Comparação lado a lado

### 1.2 Métricas de Privacidade (Privacy Metrics)

#### **Epsilon (ε) - Privacy Budget**
- **O que medir**: 
  - Epsilon final após treino completo
  - Epsilon por epoch (para análise de consumo)
  - Epsilon por nível de ruído (σ)
- **Como reportar**:
  - Tabela: `ε = 2.45 ± 0.12` (mean ± std)
  - Gráfico: Epsilon vs Noise Multiplier
  - Gráfico: Epsilon consumption over epochs
- **Valores esperados**:
  - σ=0.6: ε ≈ 1.5-3.0
  - σ=1.0: ε ≈ 3.0-6.0
  - σ=2.0: ε ≈ 6.0-12.0

#### **Delta (δ)**
- **O que medir**: Delta fixo usado (tipicamente 1e-5)
- **Como reportar**: Mencionar na metodologia: `δ = 10⁻⁵`

#### **Noise Multiplier (σ)**
- **O que medir**: Valores testados (0.6, 1.0, 2.0)
- **Como reportar**: Tabela de trade-off: σ vs Accuracy vs ε

### 1.3 Métricas de Eficiência (Efficiency Metrics)

#### **Training Time**
- **O que medir**: Tempo total de treino (segundos/horas)
- **Como reportar**: 
  - Tabela comparativa
  - Gráfico de barras: Baseline vs DP vs FL vs DP+FL
  - Overhead de privacidade: `DP overhead = 1.3x baseline time`

#### **Communication Rounds (FL)**
- **O que medir**: Número de rounds até convergência
- **Como reportar**: 
  - Tabela: Rounds por número de clientes
  - Gráfico: Convergence curves (accuracy vs rounds)

#### **Model Size / Parameters**
- **O que medir**: Número de parâmetros do modelo
- **Como reportar**: Mencionar na metodologia (já padronizado: LSTM-only ~50K params)

## 2. Análises Estatísticas (Statistical Analysis)

### 2.1 Agregação de Múltiplas Runs

#### **Métricas Agregadas**
- **Mean ± Standard Deviation**: Para todas as métricas principais
- **Confidence Intervals**: 95% CI quando relevante
- **Min/Max**: Para mostrar variabilidade

#### **Testes Estatísticos**
- **T-test ou Wilcoxon**: Comparar Baseline vs DP, Baseline vs FL, etc.
- **ANOVA**: Comparar múltiplos níveis de ruído (σ)
- **Effect Size**: Cohen's d para magnitude das diferenças

### 2.2 Análise de Trade-off Privacidade-Utilidade

#### **Privacy-Utility Trade-off Curves**
- **Eixo X**: Epsilon (ε)
- **Eixo Y**: Accuracy ou F1-Score
- **Curvas**: Uma por método (DP, FL, DP+FL)
- **Interpretação**: Mostrar o "custo" de privacidade em termos de utilidade

#### **Pareto Frontier**
- **Eixo X**: Privacy (1/ε ou -log(ε))
- **Eixo Y**: Utility (Accuracy)
- **Mostrar**: Qual método oferece melhor trade-off

## 3. Visualizações Essenciais (Essential Visualizations)

### 3.1 Figuras Principais (Main Figures)

#### **Figure 1: Performance Comparison (Bar Chart)**
- **Tipo**: Bar chart com error bars
- **Eixos**: 
  - X: Métodos (Baseline, DP-σ0.6, DP-σ1.0, DP-σ2.0, FL-3c, FL-5c, FL-10c, DP+FL)
  - Y: Accuracy (%) ou F1-Score (%)
- **Barras**: Uma cor por dataset (Sleep-EDF, WESAD)
- **Error bars**: ±1 std (3 runs)
- **Subfigures**: (a) Sleep-EDF, (b) WESAD

#### **Figure 2: Privacy-Utility Trade-off**
- **Tipo**: Scatter plot ou line plot
- **Eixos**:
  - X: Epsilon (ε) - log scale
  - Y: Accuracy (%) ou F1-Score (%)
- **Pontos**: Diferentes métodos (DP, FL, DP+FL)
- **Cores**: Diferentes níveis de ruído ou número de clientes
- **Linhas**: Trend lines ou Pareto frontier

#### **Figure 3: Confusion Matrices**
- **Tipo**: Heatmap grid (2x4 ou 4x2)
- **Layout**: 
  - Linhas: Datasets (Sleep-EDF, WESAD)
  - Colunas: Métodos (Baseline, DP, FL, DP+FL)
- **Cores**: Normalizado (0-100%) com colormap divergente
- **Anotações**: Valores percentuais em cada célula

#### **Figure 4: Training Convergence**
- **Tipo**: Line plots
- **Eixos**:
  - X: Epochs ou Communication Rounds
  - Y: Validation Accuracy
- **Linhas**: Diferentes métodos
- **Subfigures**: (a) Sleep-EDF, (b) WESAD
- **Shaded areas**: ±1 std (3 runs)

#### **Figure 5: Per-Class Performance**
- **Tipo**: Grouped bar chart
- **Eixos**:
  - X: Classes (W, N1, N2, N3, R para Sleep-EDF; non-stress, stress para WESAD)
  - Y: F1-Score por classe
- **Barras**: Diferentes métodos (Baseline, DP, FL, DP+FL)
- **Subfigures**: (a) Sleep-EDF, (b) WESAD

### 3.2 Figuras Suplementares (Supplementary Figures)

#### **Figure S1: Epsilon Consumption Over Training**
- **Tipo**: Line plot
- **Eixos**:
  - X: Epochs
  - Y: Cumulative Epsilon (ε)
- **Linhas**: Diferentes níveis de ruído (σ=0.6, 1.0, 2.0)

#### **Figure S2: Communication Efficiency (FL)**
- **Tipo**: Bar chart
- **Eixos**:
  - X: Número de clientes (3, 5, 10)
  - Y: Rounds até convergência ou Total communication cost
- **Barras**: Diferentes datasets

#### **Figure S3: Training Time Comparison**
- **Tipo**: Bar chart
- **Eixos**:
  - X: Métodos
  - Y: Training time (seconds ou hours)
- **Barras**: Normalizado ao baseline (overhead %)

#### **Figure S4: Noise Level Sensitivity**
- **Tipo**: Line plot
- **Eixos**:
  - X: Noise Multiplier (σ)
  - Y: Accuracy (%)
- **Linhas**: Diferentes datasets
- **Error bars**: ±1 std

## 4. Tabelas Essenciais (Essential Tables)

### 4.1 Tabela Principal: Performance Summary

| Method | Dataset | Accuracy (%) | F1-Score (%) | Precision (%) | Recall (%) | ε | Training Time (s) |
|--------|---------|-------------|--------------|---------------|------------|---|-------------------|
| Baseline | Sleep-EDF | 85.53 ± 0.12 | 85.03 ± 0.15 | 84.64 ± 0.18 | 85.53 ± 0.12 | ∞ | 45.2 ± 2.1 |
| DP (σ=0.6) | Sleep-EDF | 78.21 ± 0.45 | 77.89 ± 0.52 | ... | ... | 2.45 ± 0.12 | 58.7 ± 3.2 |
| DP (σ=1.0) | Sleep-EDF | 72.34 ± 0.67 | 71.95 ± 0.71 | ... | ... | 4.89 ± 0.23 | 58.9 ± 3.1 |
| DP (σ=2.0) | Sleep-EDF | 65.12 ± 0.89 | 64.78 ± 0.92 | ... | ... | 9.67 ± 0.45 | 59.1 ± 3.3 |
| FL (3c) | Sleep-EDF | 84.12 ± 0.28 | 83.67 ± 0.31 | ... | ... | ∞ | 52.3 ± 2.8 |
| FL (5c) | Sleep-EDF | 83.45 ± 0.35 | 83.01 ± 0.38 | ... | ... | ∞ | 61.2 ± 3.5 |
| FL (10c) | Sleep-EDF | 82.78 ± 0.42 | 82.34 ± 0.45 | ... | ... | ∞ | 78.9 ± 4.2 |
| DP+FL (σ=0.3, 10c) | Sleep-EDF | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Nota**: Repetir para WESAD

### 4.2 Tabela: Per-Class Performance (Sleep-EDF)

| Method | W | N1 | N2 | N3 | R |
|--------|---|---|---|---|---|
| Baseline | F1: 0.XX | F1: 0.XX | F1: 0.XX | F1: 0.XX | F1: 0.XX |
| DP (σ=0.6) | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

### 4.3 Tabela: Privacy Budget Analysis

| Method | Noise (σ) | Epsilon (ε) | Delta (δ) | Privacy Guarantee |
|--------|-----------|-------------|-----------|-------------------|
| DP (σ=0.6) | 0.6 | 2.45 ± 0.12 | 1e-5 | (2.45, 1e-5)-DP |
| DP (σ=1.0) | 1.0 | 4.89 ± 0.23 | 1e-5 | (4.89, 1e-5)-DP |
| DP (σ=2.0) | 2.0 | 9.67 ± 0.45 | 1e-5 | (9.67, 1e-5)-DP |
| DP+FL (σ=0.3) | 0.3 | ... | 1e-5 | ... |
| ... | ... | ... | ... | ... |

### 4.4 Tabela: Statistical Significance Tests

| Comparison | Dataset | p-value | Effect Size (Cohen's d) | Significant? |
|------------|---------|---------|------------------------|-------------|
| Baseline vs DP (σ=0.6) | Sleep-EDF | < 0.001 | 0.XX | ✓ |
| Baseline vs DP (σ=1.0) | Sleep-EDF | < 0.001 | 0.XX | ✓ |
| Baseline vs FL (3c) | Sleep-EDF | 0.023 | 0.XX | ✓ |
| DP (σ=0.6) vs DP (σ=1.0) | Sleep-EDF | < 0.001 | 0.XX | ✓ |
| ... | ... | ... | ... | ... |

## 5. Métricas Específicas para Dados de Saúde (Health-Specific Metrics)

### 5.1 Sleep-EDF (Classificação de Estágios de Sono)

#### **Cohen's Kappa**
- **O que medir**: Concordância entre predições e ground truth
- **Por quê**: Importante para dados médicos, considera acaso
- **Interpretação**: 
  - κ > 0.8: Excelente
  - κ > 0.6: Bom
  - κ < 0.6: Moderado/Pobre

#### **Per-Class Sensitivity (Recall)**
- **Foco especial**: N1 (estágio mais difícil de detectar)
- **Reportar**: Tabela detalhada de recall por estágio

#### **Transition Accuracy**
- **O que medir**: Precisão nas transições entre estágios
- **Por quê**: Transições são clinicamente relevantes

### 5.2 WESAD (Detecção de Stress)

#### **Sensitivity (Recall) para Stress**
- **O que medir**: Taxa de detecção de stress (classe minoritária)
- **Por quê**: Falsos negativos são críticos em saúde
- **Reportar**: Destaque especial na tabela

#### **Specificity para Non-Stress**
- **O que medir**: Taxa de correta identificação de não-stress
- **Por quê**: Evitar alarmes falsos

#### **AUC-ROC**
- **O que medir**: Área sob a curva ROC
- **Por quê**: Melhor métrica para classificação binária desbalanceada
- **Reportar**: Tabela adicional com AUC-ROC

## 6. Análises Comparativas Adicionais (Additional Comparative Analyses)

### 6.1 Comparação com State-of-the-Art

#### **Baseline Comparison Table**
- Comparar com trabalhos anteriores:
  - Sleep-EDF: Outros métodos de classificação de sono
  - WESAD: Outros métodos de detecção de stress
- **Métricas**: Accuracy, F1-Score, número de parâmetros

### 6.2 Ablation Studies

#### **Component Analysis**
- **O que testar**: 
  - Impacto de cada componente (DP, FL, DP+FL)
  - Impacto do número de clientes (FL)
  - Impacto do nível de ruído (DP)
- **Como reportar**: Tabela ou gráfico de barras

### 6.3 Robustness Analysis

#### **Sensitivity to Hyperparameters**
- **O que medir**: Variação de performance com mudanças em:
  - Learning rate
  - Batch size
  - Max grad norm (DP)
- **Como reportar**: Heatmap ou tabela

## 7. Recomendações de Implementação

### 7.1 Scripts de Análise Sugeridos

1. **`scripts/analyze_results.py`**
   - Carregar `experiments/results_log.json`
   - Calcular métricas agregadas (mean, std, min, max)
   - Gerar tabelas LaTeX
   - Gerar gráficos (matplotlib/seaborn)

2. **`scripts/generate_figures.py`**
   - Figuras principais (Figures 1-5)
   - Figuras suplementares (Figures S1-S4)
   - Exportar em alta resolução (PDF, PNG 300dpi)

3. **`scripts/statistical_tests.py`**
   - T-tests, Wilcoxon, ANOVA
   - Effect sizes (Cohen's d)
   - Gerar tabela de significância

### 7.2 Formato de Dados para Análise

#### **Estrutura JSON Sugerida**
```json
{
  "experiment_id": "baseline_sleep_edf_run1",
  "dataset": "sleep-edf",
  "method": "baseline",
  "seed": 42,
  "metrics": {
    "accuracy": 0.8553,
    "f1_score": 0.8503,
    "precision": 0.8464,
    "recall": 0.8553,
    "precision_per_class": [0.XX, ...],
    "recall_per_class": [0.XX, ...],
    "f1_per_class": [0.XX, ...],
    "confusion_matrix": [[...], ...],
    "cohens_kappa": 0.XX,
    "auc_roc": null  // apenas para WESAD
  },
  "privacy": {
    "epsilon": null,  // ou valor se DP
    "delta": 1e-5,
    "noise_multiplier": null
  },
  "efficiency": {
    "training_time_seconds": 45.2,
    "total_epochs": 16,
    "communication_rounds": null  // apenas para FL
  },
  "training_history": {
    "epoch": [1, 2, ...],
    "train_loss": [...],
    "train_acc": [...],
    "val_loss": [...],
    "val_acc": [...],
    "epsilon_history": [...]  // apenas para DP
  }
}
```

## 8. Checklist para Paper IEEE

### 8.1 Seção de Resultados (Results Section)

- [ ] Tabela principal de performance (Accuracy, F1, Precision, Recall)
- [ ] Tabela de privacy budget (ε, δ)
- [ ] Tabela de per-class performance
- [ ] Tabela de significância estatística
- [ ] Figura 1: Performance comparison (bar chart)
- [ ] Figura 2: Privacy-utility trade-off
- [ ] Figura 3: Confusion matrices
- [ ] Figura 4: Training convergence
- [ ] Figura 5: Per-class performance

### 8.2 Seção de Discussão (Discussion Section)

- [ ] Análise do trade-off privacidade-utilidade
- [ ] Comparação com state-of-the-art
- [ ] Análise de limitações
- [ ] Implicações clínicas (para dados de saúde)

### 8.3 Apêndices/Supplementary Material

- [ ] Tabelas detalhadas completas
- [ ] Figuras suplementares (S1-S4)
- [ ] Resultados de todos os runs individuais
- [ ] Código de análise (se permitido)

## 9. Referências de Formato IEEE

### 9.1 Formato de Números

- **Percentagens**: `85.53%` (2 casas decimais)
- **Epsilon**: `ε = 2.45` (2 casas decimais) ou notação científica se muito pequeno
- **Estatísticas**: `85.53 ± 0.12` (mean ± std)
- **p-values**: `p < 0.001` ou `p = 0.023`

### 9.2 Formato de Figuras

- **Resolução**: Mínimo 300 DPI
- **Formato**: PDF (vetorial) ou PNG (raster)
- **Tamanho**: Ajustar para coluna única (3.5") ou dupla (7")
- **Fontes**: Times New Roman ou Arial, tamanho legível (≥8pt)
- **Legendas**: Descritivas, auto-contidas

### 9.3 Formato de Tabelas

- **Cabeçalhos**: Claros e descritivos
- **Unidades**: Sempre especificadas
- **Casas decimais**: Consistentes (2-3 casas)
- **Notação**: IEEE standard (ε, δ, σ)

## 10. Próximos Passos

1. **Executar todos os experimentos** (baseline, DP, FL, DP+FL)
2. **Implementar scripts de análise** (`scripts/analyze_results.py`)
3. **Gerar todas as figuras** (`scripts/generate_figures.py`)
4. **Calcular testes estatísticos** (`scripts/statistical_tests.py`)
5. **Criar tabelas LaTeX** para inclusão no paper
6. **Revisar e validar** todas as métricas

---

**Nota**: Este documento serve como guia. Adapte conforme necessário para o escopo específico do paper e requisitos do IEEE.

