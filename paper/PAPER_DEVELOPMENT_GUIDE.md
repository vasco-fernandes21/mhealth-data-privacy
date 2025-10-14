# Paper Development Guide

## 📋 Estrutura do Paper

### ✅ Secções Completas (Baseadas no Trabalho Existente)

1. **Introduction** - Completa
2. **Related Work** - Completa
3. **Methodology** (Baseline):
   - ✅ Datasets (WESAD, Sleep-EDF)
   - ✅ Preprocessing Pipeline
   - ✅ Baseline Model Architectures
   - ✅ Baseline Results

### 🔄 Secções a Implementar (Próximas 14 Semanas)

4. **Privacy-Preserving Techniques** - Framework criado, implementação pendente
5. **Experimental Setup** - Framework criado
6. **Results and Analysis** - Pendente de resultados experimentais
7. **Discussion** - Framework criado
8. **Conclusions** - Framework criado

---

## 🗓️ Plano de Desenvolvimento (14 Semanas)

### **Fase 1: Semanas 1-2 - Revisão de Literatura**
- [x] Princípios de Federated Learning
- [x] Princípios de Differential Privacy
- [x] Requisitos GDPR
- [ ] Completar seção Related Work com mais referências específicas
- [ ] Estudar TensorFlow Federated
- [ ] Estudar PyTorch Opacus

**Deliverables:**
- Bibliografia atualizada (references.bib)
- Notas sobre FL/DP aplicados a mHealth

---

### **Fase 2: Semanas 3-5 - Preparação de Datasets e Ambiente**

**Semana 3: Ambiente Experimental**
- [ ] Instalar TensorFlow Federated
- [ ] Instalar PyTorch + Opacus
- [ ] Criar notebooks Jupyter para experimentação
- [ ] Testar ambiente com modelos baseline

**Semana 4: Adaptação de Dados para FL**
- [ ] Implementar split por sujeito para simular clientes FL
- [ ] Criar função de distribuição de dados (IID vs non-IID)
- [ ] Configurar DataLoaders para FL
- [ ] Testar com 3, 5, 10 clientes

**Semana 5: Preparação para DP**
- [ ] Adaptar pipeline para batching compatível com Opacus
- [ ] Implementar privacy accounting
- [ ] Definir grid de epsilon values
- [ ] Preparar baseline de comparação

**Deliverables:**
- Código de preparação de dados para FL/DP
- Notebooks de teste

---

### **Fase 3: Semanas 6-8 - Implementação FL e DP**

**Semana 6: Federated Learning**
- [ ] Implementar FedAvg com TensorFlow Federated
- [ ] Configurar servidor e clientes
- [ ] Implementar métricas de convergência
- [ ] Testar com WESAD (3, 5, 10 clientes)

**Semana 7: Differential Privacy**
- [ ] Implementar DP-SGD com Opacus
- [ ] Configurar privacy engine
- [ ] Sweep de epsilon values (0.5, 1.0, 2.0, 5.0, 10.0)
- [ ] Implementar privacy accounting

**Semana 8: Híbrido FL + DP**
- [ ] Combinar FL e DP
- [ ] Implementar DP local em cada cliente
- [ ] Testar combinação
- [ ] Comparar com FL-only e DP-only

**Deliverables:**
- Código FL completo
- Código DP completo
- Código híbrido
- Primeiros resultados

---

### **Fase 4: Semanas 9-10 - Execução de Testes**

**Semana 9: Experimentos FL**
- [ ] Executar FL com WESAD (3, 5, 10 clientes)
- [ ] Executar FL com Sleep-EDF
- [ ] Coletar métricas: accuracy, communication overhead, time
- [ ] Gerar plots de convergência

**Semana 10: Experimentos DP**
- [ ] Executar DP-SGD com todos os epsilon values
- [ ] Testar em WESAD e Sleep-EDF
- [ ] Coletar métricas: accuracy, epsilon, delta, time
- [ ] Gerar plots accuracy vs. epsilon

**Deliverables:**
- Tabelas de resultados (CSV/JSON)
- Figuras (convergência, trade-offs)
- Logs de execução

---

### **Fase 5: Semanas 11-12 - Análise de Resultados**

**Semana 11: Análise Quantitativa**
- [ ] Compilar todos os resultados
- [ ] Criar tabelas comparativas (Baseline vs FL vs DP vs Híbrido)
- [ ] Análise estatística (significância das diferenças)
- [ ] Análise per-class (impacto em minority classes)

**Semana 12: Visualizações e Interpretação**
- [ ] Gerar todos os gráficos para o paper
- [ ] Criar figuras de arquitetura (FL, DP)
- [ ] Análise de trade-offs
- [ ] Preparar discussion points

**Deliverables:**
- Todas as tabelas finais
- Todas as figuras
- Análise estatística
- Rascunho da seção Results

---

### **Fase 6: Semanas 13-14 - Redação Final**

**Semana 13: Escrita do Paper**
- [ ] Completar seção Results com dados reais
- [ ] Completar seção Discussion
- [ ] Completar seção Conclusions
- [ ] Revisar Abstract com resultados finais
- [ ] Revisar Introduction para alinhamento

**Semana 14: Revisão e Finalização**
- [ ] Revisão completa do paper
- [ ] Verificar consistência de números/tabelas
- [ ] Verificar formatação IEEE
- [ ] Spell check e grammar check
- [ ] Preparar versão final

**Deliverables:**
- Paper completo (paper.tex)
- Todos os assets (figuras, tabelas)
- Código final documentado

---

## 📊 Tabelas e Figuras a Criar

### Tabelas

1. **Table 1: Baseline Results** ✅ (framework criado)
   - WESAD: 77.64% accuracy
   - Sleep-EDF: 87.45% accuracy

2. **Table 2: FL Communication Overhead** (a criar)
   - Clientes vs Bytes/Round vs Total vs Time

3. **Table 3: DP Results** (a criar)
   - Epsilon vs Accuracy vs Delta vs F1 vs Time

4. **Table 4: Comparative Analysis** (a criar)
   - Baseline vs FL vs DP vs Hybrid

5. **Table 5: Per-Class Analysis** (a criar)
   - Impact on minority classes (stress, N1 sleep)

### Figuras

1. **Figure 1: System Architecture** (a criar)
   - Overall architecture showing FL/DP components

2. **Figure 2: FL Convergence** (a criar)
   - Accuracy vs Rounds for different client configurations

3. **Figure 3: DP Trade-off** (a criar)
   - Accuracy vs Epsilon (privacy budget)

4. **Figure 4: Comparative Plot** (a criar)
   - Multi-dimensional comparison (accuracy, privacy, efficiency)

---

## 🔧 Código a Desenvolver

### Estrutura de Diretórios Proposta

```
src/
├── privacy/
│   ├── federated_learning/
│   │   ├── fl_wesad.py
│   │   ├── fl_sleep_edf.py
│   │   └── utils.py
│   ├── differential_privacy/
│   │   ├── dp_wesad.py
│   │   ├── dp_sleep_edf.py
│   │   └── privacy_accountant.py
│   └── hybrid/
│       ├── fl_dp_wesad.py
│       └── fl_dp_sleep_edf.py
├── evaluation/
│   ├── metrics.py
│   ├── visualization.py
│   └── statistical_tests.py
└── experiments/
    ├── run_all_experiments.py
    ├── config.yaml
    └── README.md

notebooks/
├── 04_federated_learning.ipynb
├── 05_differential_privacy.ipynb
├── 06_hybrid_fl_dp.ipynb
└── 07_results_analysis.ipynb

results/
├── federated_learning/
│   ├── wesad/
│   └── sleep-edf/
├── differential_privacy/
│   ├── wesad/
│   └── sleep-edf/
└── comparative/
    └── summary.json
```

---

## ✅ Checklist de Implementação

### Federated Learning

- [ ] Client data partitioning (subject-wise)
- [ ] FedAvg implementation
- [ ] Secure aggregation (optional)
- [ ] Communication tracking
- [ ] Convergence monitoring
- [ ] Multi-client configurations (3, 5, 10)
- [ ] IID vs non-IID analysis

### Differential Privacy

- [ ] DP-SGD integration
- [ ] Privacy engine setup
- [ ] Epsilon sweep (0.5, 1.0, 2.0, 5.0, 10.0)
- [ ] Privacy accounting (moments accountant)
- [ ] Gradient clipping
- [ ] Noise calibration
- [ ] Privacy budget tracking

### Evaluation

- [ ] Accuracy metrics (per-class, overall)
- [ ] Privacy metrics (epsilon, delta for DP; qualitative for FL)
- [ ] Efficiency metrics (time, communication, memory)
- [ ] Statistical significance tests
- [ ] Visualization functions (convergence plots, trade-off curves)
- [ ] Comparative analysis

---

## 📝 Notas de Desenvolvimento

### Pontos-Chave do Trabalho Existente (a Integrar)

1. **Baseline Performance:**
   - WESAD: 77.64% accuracy, 78.09% F1-score
   - Stress detection: 72.6% recall (clinically important)
   - Sleep-EDF: 87.45% accuracy

2. **Otimizações de Preprocessing:**
   - 32 Hz sampling (balance entre qualidade e eficiência)
   - Filtros otimizados para cada sinal
   - Janelas completas apenas (sem zero-padding)
   - Clipping menos agressivo (1-99%)

3. **Arquitetura CNN-LSTM:**
   - 454,338 parâmetros (1.73 MB)
   - Regularização L2, label smoothing
   - Adequada para edge deployment

### Estas otimizações são FUNDAMENTAIS para:
- **FL**: Menor modelo = menos comunicação
- **DP**: Modelo eficiente reduz privacy budget consumption
- **Edge deployment**: Viável para dispositivos móveis

---

## 🎯 Objetivos de Cada Seção

### Results (a completar com dados reais)

**Federated Learning:**
- Demonstrar convergência em X rounds
- Mostrar degradação mínima vs baseline (target: < 5%)
- Quantificar communication overhead
- Comparar 3, 5, 10 clientes

**Differential Privacy:**
- Caracterizar trade-off accuracy-epsilon
- Identificar epsilon "sweet spot" (target: epsilon=2-5 com < 10% accuracy loss)
- Mostrar impacto em minority classes
- Documentar computational overhead

**Híbrido:**
- Demonstrar defense-in-depth
- Comparar custo vs benefício

### Discussion

**Focar em:**
1. Privacy-utility trade-offs (quantitativos)
2. GDPR compliance (qualitativo)
3. Practical recommendations (aplicáveis)
4. Limitations (honestos)

### Conclusions

**Mensagens-chave:**
1. FL mantém accuracy com privacy prática
2. DP oferece garantias formais com custo aceitável (epsilon moderado)
3. Híbrido é viável para high-risk applications
4. Técnicas são compatíveis com GDPR
5. Implementação é feasible com frameworks open-source

---

## 📚 Próximos Passos Imediatos

1. **Esta Semana:**
   - [ ] Instalar TensorFlow Federated
   - [ ] Instalar PyTorch Opacus
   - [ ] Criar notebook de teste para FL
   - [ ] Criar notebook de teste para DP

2. **Próxima Semana:**
   - [ ] Implementar FL básico com WESAD
   - [ ] Implementar DP básico com WESAD
   - [ ] Primeiros resultados preliminares

3. **Milestone 1 (Semana 5):**
   - [ ] FL funcional
   - [ ] DP funcional
   - [ ] Baseline comparisons funcionais

---

## 🔗 Recursos Úteis

### Tutoriais
- TensorFlow Federated: https://www.tensorflow.org/federated/tutorials/tutorials_overview
- Opacus: https://opacus.ai/tutorials/
- FL + DP: https://blog.openmined.org/federated-learning-differential-privacy/

### Papers de Referência
- FedAvg: McMahan et al. (2017)
- DP-SGD: Abadi et al. (2016)
- FL in Healthcare: Rieke et al. (2020)

### Datasets
- WESAD: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
- Sleep-EDF: https://physionet.org/content/sleep-edfx/1.0.0/

---

*Este guia será atualizado conforme o projeto progride.*

