# Privacy-Preserving Health Data Analysis: Resumo Executivo

## Objetivo

Investigar e quantificar o **trade-off entre privacidade e precisão** em sistemas de análise de dados de saúde móveis, aplicando técnicas de privacidade diferencial (DP) e aprendizagem federada (FL) em dois contextos reais diferentes.

---

## Procedimento

### Fase 1: Datasets
- **Sleep-EDF**: Dados de padrões de sono (EEG/EOG)
- **WESAD**: Dados de resposta emocional a stress (ECG/EDA/Temperatura)

### Fase 2: Modelos (3 por dataset)
1. **Baseline**: Modelo padrão (sem técnicas de privacidade)
2. **DP**: Modelo com Differential Privacy (ruído aplicado)
3. **FL**: Modelo com Federated Learning (treino descentralizado)

### Fase 3: Treino e Avaliação
Para cada combinação (modelo + dataset):
- Métrica de **performance**: Accuracy, Precision, Recall, F1
- Métrica de **privacidade**: Epsilon (DP), Communication cost (FL)

### Fase 4: Comparação
- Tabelas e gráficos de trade-off (Accuracy vs. Privacy)
- Análise por dataset e por técnica
- Identificar qual abordagem oferece melhor equilíbrio

---

## Resultados Esperados

### 1. Trade-off Quantificado
**Exemplo de achado**: 
- DP com ε=1: Accuracy ↓ 15%, mas privacidade forte
- DP com ε=10: Accuracy ↓ 5%, privacidade moderada
- FL com 5 clientes: Accuracy ↓ 3%, privacidade via descentralização

### 2. Consistência entre Datasets
- **Se DP afeta Sleep-EDF em X%, também afeta WESAD em ~X%** → Padrão generalível
- Validação de que técnicas funcionam em diferentes contextos de saúde

### 3. Recomendações Práticas
- Qual técnica é mais adequada para apps móveis de saúde?
- DP é melhor para dados muito sensíveis (Sleep → diagnósticos)
- FL é melhor para dados distribuídos (múltiplos wearables)
- Valores ótimos de epsilon/clientes para aplicações reais

### 4. Implicações para Sistemas de Informação
- GDPR compliance vs. performance
- Viabilidade de implementação em dispositivos móveis
- Carga computacional (privacidade tem custo)

---

## Conclusões Principais (Esperadas)

1. **Trade-off é real mas quantificável**: Não é binário (privacidade OU performance), mas contínuo
2. **Técnica varia com contexto**: DP melhor para dados centralizados; FL para cenários distribuídos
3. **Datasets reais validam generalização**: Mesmos padrões em Sleep-EDF e WESAD
4. **Aplicável a mobile health**: Com calibração correta, privacidade é viável sem perda crítica de precisão

---

## Output Final

- **Paper académico** com literature review, metodologia, resultados e discussão
- **Tabelas de resultados** (6 linhas de baseline vs. técnicas)
- **Gráficos de trade-off** (Accuracy vs. Privacy)
- **Recomendações** para design de sistemas móveis de saúde