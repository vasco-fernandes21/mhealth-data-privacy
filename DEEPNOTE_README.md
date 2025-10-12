# 🚀 mHealth Data Privacy - Deepnote Setup

Este projeto foi adaptado para funcionar perfeitamente no **Deepnote**! 

## 📋 **O que foi adaptado:**

### ✅ **Notebooks Adaptados:**
- `00_deepnote_setup.ipynb` - Setup inicial (renomeado de colab_setup)
- `01_preprocess_sleep_edf.ipynb` - Pré-processamento Sleep-EDF
- `02_preprocess_wesad.ipynb` - Pré-processamento WESAD  
- `03_train_baseline.ipynb` - Treino baseline
- `04_train_dp.ipynb` - Treino com Differential Privacy
- `05_train_fl.ipynb` - Treino com Federated Learning
- `06_analysis.ipynb` - Análise final

### 🔄 **Principais Mudanças:**
- **Paths**: `/content/drive/MyDrive/mhealth-data` → `./data`
- **Storage**: Google Drive → Storage local do Deepnote
- **Upload**: Instruções para usar o sistema de upload do Deepnote
- **Setup**: Removido clone do Git (assume que já está no projeto)

## 🎯 **Como usar no Deepnote:**

### **1. Criar Projeto**
1. Criar novo projeto no Deepnote
2. **Opção A**: Clone automático (recomendado)
   - Execute `00_deepnote_setup.ipynb` - ele clona automaticamente
3. **Opção B**: Clone manual
   ```bash
   git clone https://github.com/vasco-fernandes21/mhealth-data-privacy.git
   cd mhealth-data-privacy
   ```
4. O projeto estará pronto para usar!

### **2. Executar Pipeline**
```bash
# Execute os notebooks em sequência:
0. 00_organize_data.ipynb       # Organizar dados (se necessário)
1. 00_deepnote_setup.ipynb      # Setup inicial
2. 01_preprocess_sleep_edf.ipynb # Pré-processamento Sleep-EDF
3. 02_preprocess_wesad.ipynb     # Pré-processamento WESAD
4. 03_train_baseline.ipynb       # Treino baseline
5. 04_train_dp.ipynb            # Treino DP
6. 05_train_fl.ipynb            # Treino FL
7. 06_analysis.ipynb            # Análise final
```

### **3. Upload de Dados**
- Use o sistema de upload do Deepnote
- Coloque os datasets em `./data/raw/sleep-edf/` e `./data/raw/wesad/`
- Ou arraste e solte os arquivos diretamente

**📋 Tipos de arquivos esperados:**
- **Sleep-EDF**: `.rec` (gravações), `.hyp` (anotações), `RECORDS`
- **WESAD**: `.pkl` (dados sincronizados - apenas estes arquivos!)

### **4. Problema: Estrutura "Achatada"**
**Se os arquivos ficaram todos no mesmo nível** (sem pastas):
1. Execute `00_organize_data.ipynb` primeiro
2. Este notebook organiza automaticamente os arquivos
3. Depois execute o setup normal

## 📁 **Estrutura de Dados:**

```
./data/
├── raw/
│   ├── sleep-edf/          # Dados originais Sleep-EDF (.rec, .hyp)
│   └── wesad/              # Dados originais WESAD (.pkl)
├── processed/
│   ├── sleep-edf/          # Dados processados Sleep-EDF
│   └── wesad/              # Dados processados WESAD
├── models/                 # Modelos treinados
└── results/                # Resultados e visualizações
```

## ⚡ **Vantagens do Deepnote:**

### ✅ **Storage Integrado**
- Dados ficam salvos no projeto
- Não precisa montar drives externos
- Persistência entre sessões

### ✅ **Interface Moderna**
- Interface mais limpa que Colab
- Melhor organização de arquivos
- Upload de arquivos mais fácil

### ✅ **Colaboração**
- Múltiplos usuários no mesmo projeto
- Versionamento automático
- Histórico de execuções

### ✅ **Performance**
- GPU disponível (Deepnote Pro)
- Recursos dedicados
- Menos limitações de tempo

## 🔧 **Configuração GPU:**

Para usar GPU no Deepnote:
1. Upgrade para Deepnote Pro
2. Selecione GPU nas configurações
3. O notebook detectará automaticamente

## 📊 **Resultados:**

Todos os resultados são salvos em:
- **Modelos**: `./data/models/`
- **Resultados**: `./data/results/`
- **Visualizações**: PNG files nos resultados
- **Relatórios**: JSON files com métricas

## 📁 **Estrutura Final no Deepnote:**

```
mhealth-data-privacy/
├── notebooks/
├── src/
├── data/
│   └── raw/
│       ├── sleep-edf/
│       │   ├── sc4001e0.rec    # Recording file
│       │   ├── sc4001e0.hyp    # Hypnogram file
│       │   ├── sc4002e0.rec    # Recording file
│       │   ├── sc4002e0.hyp    # Hypnogram file
│       │   ├── RECORDS         # List of all files
│       │   └── ...
│       └── wesad/
│           ├── S2.pkl          # Subject 2 data (synchronized)
│           ├── S3.pkl          # Subject 3 data (synchronized)
│           ├── S4.pkl          # Subject 4 data (synchronized)
│           ├── S5.pkl          # Subject 5 data (synchronized)
│           └── ...             # S6-S17 (S1 and S12 missing)
└── requirements.txt
```

## 🚨 **Troubleshooting:**

### **Problema**: Módulos não encontrados
**Solução**: Execute primeiro o `00_deepnote_setup.ipynb` (ele clona o repositório automaticamente)

### **Problema**: Repositório não encontrado
**Solução**: O notebook de setup clona automaticamente, ou clone manualmente:
```bash
git clone https://github.com/vasco-fernandes21/mhealth-data-privacy.git
cd mhealth-data-privacy
```

### **Problema**: Dados não encontrados
**Solução**: Upload dos datasets para `./data/raw/`

### **Problema**: GPU não detectada
**Solução**: Normal no plano gratuito - CPU funciona perfeitamente (apenas mais lento)

## 🖥️ **CPU-Only Environment (Plano Gratuito):**

### **✅ Funciona perfeitamente:**
- **Pré-processamento**: 5-15 minutos
- **Treino baseline**: 30-60 minutos
- **DP training**: 45-90 minutos
- **FL training**: 20-40 minutos
- **Análise**: 5-10 minutos

### **⚡ Otimizações automáticas:**
- TensorFlow configurado para usar todos os cores
- Threading otimizado para CPU
- Memória gerenciada automaticamente

## 💡 **Dicas:**

1. **Execute sempre o setup primeiro** (notebook 00)
2. **Upload dos dados** antes de executar pré-processamento
3. **Use GPU** para treino mais rápido
4. **Salve frequentemente** - Deepnote mantém histórico
5. **Colabore** - partilhe o projeto com colegas

## 🎉 **Pronto para usar!**

O projeto está 100% adaptado para Deepnote. Basta criar o projeto, fazer upload dos dados e executar os notebooks em sequência!

---

**Desenvolvido por**: Eduardo Carvalho, Filipe Correia, Vasco Fernandes  
**Adaptado para**: Deepnote  
**Versão**: 1.0
