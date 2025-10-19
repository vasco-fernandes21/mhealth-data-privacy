# Otimizações Implementadas no Pré-processamento de Dados

## Resumo das Melhorias de Performance

Este documento descreve as otimizações implementadas no sistema de pré-processamento de dados para os datasets Sleep-EDF e WESAD, visando acelerar significativamente o processo de preparação de dados para machine learning.

## 🚀 Otimizações Implementadas

### 1. **Processamento Paralelo (Multiprocessing)**
- **Sleep-EDF**: Processa múltiplos arquivos EDF simultaneamente usando `multiprocessing.Pool`
- **WESAD**: Processa múltiplos arquivos pickle em paralelo
- **Impacto**: Redução drástica no tempo de processamento proporcional ao número de núcleos disponíveis
- **Configuração**: 4-8 workers por padrão (ajustável)

### 2. **Operações Vetoriais Otimizadas**
- **Filtros**: Pré-computação de coeficientes de filtro Butterworth para evitar recálculos
- **Extração de Características**: Vetorização completa usando NumPy para operações em lote
- **Aplicação de Filtros**: Uso de `sosfiltfilt` para melhor estabilidade numérica
- **Impacto**: Redução significativa no tempo de cálculo por arquivo

### 3. **Processamento em Lotes (Batch Processing)**
- **Sleep-EDF**: Processa épocas em lotes de 100 para reduzir uso de memória
- **Otimização de Memória**: Evita carregar todos os dados simultaneamente
- **Impacto**: Menor consumo de RAM e melhor escalabilidade

### 4. **Sistema de Cache Inteligente**
- **Verificação de Arquivos**: Detecta se dados já foram processados
- **Hash de Integridade**: Verifica se dados de entrada mudaram usando hash MD5
- **Cache Persistente**: Salva informações de cache em disco
- **Impacto**: Evita reprocessamento desnecessário de dados idênticos

### 5. **Otimização de E/S**
- **Compressão Numpy**: Usa compressão automática nos arrays `.npy`
- **Joblib**: Serialização eficiente de objetos Python
- **Impacto**: Menor espaço em disco e I/O mais rápido

## 📊 Melhorias de Performance Esperadas

### Sleep-EDF Dataset:
- **Antes**: ~30-45 minutos (processamento sequencial)
- **Depois**: ~5-8 minutos (com 8 workers paralelos)
- **Melhoria**: **75-85% de redução no tempo**

### WESAD Dataset:
- **Antes**: ~15-20 minutos (processamento sequencial)
- **Depois**: ~3-5 minutos (com 8 workers paralelos)
- **Melhoria**: **70-80% de redução no tempo**

## 🛠️ Como Usar as Otimizações

### Execução Básica:
```bash
cd /Users/vasco/Desktop/Mestrado/SIDM/mhealth-data-privacy
python src/preprocessing/preprocess_all.py
```

### Com Cache (recomendado):
```bash
# Primeira execução - processa e cria cache
python src/preprocessing/preprocess_all.py

# Execuções subsequentes - usa cache automaticamente
python src/preprocessing/preprocess_all.py
```

### Forçar Reprocessamento:
```python
from src.preprocessing.sleep_edf import preprocess_sleep_edf
from src.preprocessing.wesad import preprocess_wesad_temporal

# Sleep-EDF com reprocessamento forçado
preprocess_sleep_edf(
    data_dir='data/raw/sleep-edf',
    output_dir='data/processed/sleep-edf',
    force_reprocess=True
)

# WESAD com reprocessamento forçado
preprocess_wesad_temporal(
    data_dir='data/raw/wesad',
    output_dir='data/processed/wesad',
    force_reprocess=True
)
```

## 🔧 Configurações Ajustáveis

### Número de Workers:
```python
# Ajustar número de processos paralelos
preprocess_sleep_edf(..., n_workers=6)  # 6 workers
preprocess_wesad_temporal(..., n_workers=6)  # 6 workers
```

### Tamanho do Batch (Sleep-EDF):
- Modificado internamente para 100 épocas por batch
- Otimizado para equilíbrio memória/performance

## 📋 Características Técnicas

### Sleep-EDF:
- **Arquivos**: ~80 arquivos EDF pareados (PSG + Hypnogram)
- **Épocas**: ~15.000-20.000 épocas de 30s cada
- **Características**: 24 características por época (8 por canal × 3 canais)
- **Formato**: Arrays NumPy comprimidos

### WESAD:
- **Arquivos**: 15 arquivos pickle (um por sujeito)
- **Janelas**: ~8.000-10.000 janelas de 60s cada
- **Características**: 17 sinais × 1920 amostras por janela
- **Formato**: Arrays 3D otimizados para CNN/LSTM

## ✅ Verificação de Funcionamento

Para verificar se as otimizações estão funcionando:

1. **Monitorar logs**: Verificar mensagens de "PARALLEL" e "cache hit/miss"
2. **Comparar tempos**: Medir tempo de execução antes/depois
3. **Verificar arquivos**: Confirmar criação de arquivos `.cache_info.pkl`
4. **Testar paralelismo**: Usar `htop` ou `top` para ver múltiplos processos Python

## 🚨 Considerações Importantes

1. **Uso de CPU**: Múltiplos processos podem usar 100% da CPU disponível
2. **Memória**: Batch processing reduz pico de memória, mas ainda requer RAM suficiente
3. **Cache**: Sistema de cache acelera execuções subsequentes significativamente
4. **Compatibilidade**: Todas as otimizações são retrocompatíveis com código existente

## 🎯 Próximas Otimizações Potenciais

- [ ] Implementar processamento GPU com CuPy/Numba
- [ ] Usar HDF5 para armazenamento mais eficiente
- [ ] Otimizar ainda mais a extração de características com JAX
- [ ] Implementar pipeline de streaming para datasets muito grandes

---

*Implementado por: Sistema de Otimização de Pré-processamento - 2025*

